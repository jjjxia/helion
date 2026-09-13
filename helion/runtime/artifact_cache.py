from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import enum
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import threading
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast
import uuid

import torch
from torch._inductor.codecache import PyCodeCache

from ..autotuner.base_cache import helion_key
from ..autotuner.base_cache import should_skip_cache
from ..autotuner.base_cache import torch_key_wrapper
from ..autotuner.base_cache import triton_key_wrapper
from ..autotuner.local_cache import get_helion_cache_dir
from ..autotuner.local_cache import helion_triton_cache_dir
from ..language.constexpr import ConstExpr

if TYPE_CHECKING:
    from .kernel import BoundKernel
    from .kernel import Kernel


log: logging.Logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_ENV_VAR = "HELION_KERNEL_ARTIFACT_CACHE"
_CACHE_SUBDIR = "kernel_artifacts"
_KNOWN_COMPILER_MODULES = ("helion", "torch", "triton")


class _Uncacheable(TypeError):
    pass


@dataclasses.dataclass(frozen=True)
class ArtifactCacheRequest:
    key: str
    payload: dict[str, object]
    input_identity: dict[str, object]
    device_index: int


@dataclasses.dataclass(frozen=True)
class ArtifactCacheHit:
    key: str
    input_identity: dict[str, object]
    custom_key: object
    run: Callable[..., Any]

    def matches(self, kernel: Kernel[Any], args: tuple[object, ...]) -> bool:
        try:
            custom_key = (
                None
                if kernel._key_fn is None
                else _canonical_value(kernel._key_fn(*args))
            )
            return (
                _input_identity(args) == self.input_identity
                and custom_key == self.custom_key
            )
        except _Uncacheable:
            return False


def enabled() -> bool:
    """Whether the experimental pre-bind artifact cache is enabled."""
    return os.environ.get(_ENV_VAR, "").strip().lower() not in {
        "",
        "0",
        "false",
    }


def prepare_request(
    kernel: Kernel[Any], args: tuple[object, ...]
) -> ArtifactCacheRequest | None:
    """Build the strict pre-bind lookup key, or return ``None`` if unsupported."""
    if not enabled() or should_skip_cache():
        return None
    if (
        kernel.settings.backend != "triton"
        or kernel.settings.force_autotune
        or kernel.settings.distributed
        or kernel._declares_process_group
        or len(kernel.configs) != 1
        or len(args) != kernel._num_params
    ):
        return None
    try:
        input_identity = _input_identity(args)
        device_index, hardware = _hardware_identity(args)
        payload: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "kernel": _kernel_identity(kernel),
            "config": _config_identity(kernel.configs[0].config),
            "settings": _settings_identity(kernel),
            "input": input_identity,
            "custom_key": (
                None
                if kernel._key_fn is None
                else _canonical_value(kernel._key_fn(*args))
            ),
            "hardware": hardware,
            "compiler": {
                "helion": helion_key(),
                "torch": torch_key_wrapper(),
                "triton": triton_key_wrapper(),
            },
            "python": {
                "cache_tag": sys.implementation.cache_tag,
                "version": list(sys.version_info[:3]),
            },
            "reset_generation": kernel._artifact_cache_generation,
        }
        return ArtifactCacheRequest(
            key=_stable_hash(payload),
            payload=payload,
            input_identity=input_identity,
            device_index=device_index,
        )
    except (_Uncacheable, OSError, TypeError, ValueError):
        log.debug(
            "Kernel artifact cache bypass for %s",
            kernel.name,
            exc_info=True,
        )
        return None


def load(kernel: Kernel[Any], request: ArtifactCacheRequest) -> ArtifactCacheHit | None:
    """Load a validated generated module for ``request`` without binding."""
    entry_path, module_path = _entry_paths(request.key)
    try:
        entry = json.loads(entry_path.read_text())
        if (
            not isinstance(entry, dict)
            or entry.get("schema_version") != _SCHEMA_VERSION
            or _canonical_bytes(entry.get("payload"))
            != _canonical_bytes(request.payload)
        ):
            return None
        source_sha256 = entry.get("source_sha256")
        pycodecache_key = entry.get("pycodecache_key")
        kernel_name = entry.get("kernel_name")
        if (
            not isinstance(source_sha256, str)
            or not isinstance(pycodecache_key, str)
            or not pycodecache_key.isalnum()
            or kernel_name != kernel.name
            or not module_path.is_file()
            or _file_sha256(module_path) != source_sha256
        ):
            return None
        if "TRITON_CACHE_DIR" not in os.environ:
            os.environ["TRITON_CACHE_DIR"] = helion_triton_cache_dir(
                request.device_index
            )
        module = PyCodeCache.load_by_key_path(pycodecache_key, str(module_path))
        run = getattr(module, kernel.name)
        if not callable(run):
            return None
    except Exception:
        log.debug(
            "Kernel artifact cache read failed for %s",
            kernel.name,
            exc_info=True,
        )
        return None
    return ArtifactCacheHit(
        request.key,
        request.input_identity,
        request.payload["custom_key"],
        run,
    )


def store(
    kernel: Kernel[Any],
    bound: BoundKernel[Any],
    request: ArtifactCacheRequest,
) -> bool:
    """Atomically persist a successfully executed fixed-config Triton module."""
    if not enabled() or should_skip_cache():
        return False
    # Runtime-value classifiers are discovered only by tracing. They cannot be
    # reproduced safely by the pre-bind lookup, so keep those kernels on the
    # normal path. Tensor shape/stride and scalar hl.specialize facts are already
    # represented exactly by ``input_identity``.
    if bound.env.runtime_input_specializations:
        return False
    config = bound._config
    if config is None:
        return False
    source_value = bound.get_cached_path(config)
    if source_value is None:
        return False
    source_path = Path(source_value)
    try:
        source_sha256 = _file_sha256(source_path)
        pycodecache_key = source_path.stem
        if not pycodecache_key.isalnum():
            return False
        entry_path, module_path = _entry_paths(request.key)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_copy(source_path, module_path)
        entry = {
            "schema_version": _SCHEMA_VERSION,
            "payload": request.payload,
            "kernel_name": kernel.name,
            "pycodecache_key": pycodecache_key,
            "source_sha256": source_sha256,
        }
        _atomic_write(entry_path, _canonical_bytes(entry) + b"\n")
    except OSError:
        log.debug(
            "Kernel artifact cache write failed for %s",
            kernel.name,
            exc_info=True,
        )
        return False
    return True


def _entry_paths(key: str) -> tuple[Path, Path]:
    root = get_helion_cache_dir() / _CACHE_SUBDIR / f"v{_SCHEMA_VERSION}"
    return root / f"{key}.json", root / f"{key}.py"


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write(destination: Path, data: bytes) -> None:
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_bytes(data)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_value(value: object, active: set[int] | None = None) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        number = value
        assert isinstance(number, float)
        return {"float": number.hex()}
    if isinstance(value, enum.Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _canonical_value(value.value, active),
        }
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value)}
    if isinstance(value, torch.device):
        return {"torch_device": str(value)}
    if isinstance(value, Path):
        return {"path": str(value)}
    if inspect.ismodule(value):
        module = value
        assert isinstance(module, ModuleType)
        return _module_identity(module)
    if inspect.isfunction(value) or inspect.isclass(value) or inspect.isbuiltin(value):
        return _callable_identity(value, active)

    seen = set() if active is None else active
    identity = id(value)
    if identity in seen:
        raise _Uncacheable("recursive dependency")
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            items = []
            for key, item in value.items():
                canonical_key = _canonical_value(key, seen)
                canonical_item = _canonical_value(item, seen)
                items.append([canonical_key, canonical_item])
            items.sort(key=lambda item: _canonical_bytes(item[0]))
            return {"mapping": items}
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return {
                "sequence_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "items": [_canonical_value(item, seen) for item in value],
            }
        if isinstance(value, (set, frozenset)):
            items = [_canonical_value(item, seen) for item in value]
            items.sort(key=_canonical_bytes)
            return {
                "set_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "items": items,
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {
                    field.name: _canonical_value(getattr(value, field.name), seen)
                    for field in dataclasses.fields(value)
                },
            }
    finally:
        seen.remove(identity)
    raise _Uncacheable(
        f"unsupported dependency {type(value).__module__}.{type(value).__qualname__}"
    )


def _module_identity(module: ModuleType) -> dict[str, object]:
    name = module.__name__
    root = name.partition(".")[0]
    if root in _KNOWN_COMPILER_MODULES or root in sys.stdlib_module_names:
        return {"module": name}
    # ``module.attr`` can observe arbitrary mutable module state. Import the
    # callable/constant directly if it needs to participate in this cache;
    # otherwise there is no bounded pre-bind dependency projection.
    raise _Uncacheable(f"external module dependency {name} is not supported")


def _callable_identity(
    value: object, active: set[int] | None = None
) -> dict[str, object]:
    module_name = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        raise _Uncacheable("callable has no stable identity")
    identity: dict[str, object] = {
        "callable": f"{module_name}.{qualname}",
    }
    root = module_name.partition(".")[0]
    if root in _KNOWN_COMPILER_MODULES or root in sys.stdlib_module_names:
        return identity
    source = inspect.getsourcefile(cast("Any", value))
    if source is None:
        raise _Uncacheable(f"callable {module_name}.{qualname} has no source")
    identity["source_sha256"] = _file_sha256(Path(source))
    if not inspect.isfunction(value):
        return identity

    seen = set() if active is None else active
    value_id = id(value)
    if value_id in seen:
        identity["recursive"] = True
        return identity
    seen.add(value_id)
    try:
        closure = inspect.getclosurevars(value)
        identity["dependencies"] = {
            "globals": {
                name: _canonical_value(item, seen)
                for name, item in sorted(closure.globals.items())
            },
            "nonlocals": {
                name: _canonical_value(item, seen)
                for name, item in sorted(closure.nonlocals.items())
            },
            "unbound_names": sorted(closure.unbound),
        }
    finally:
        seen.remove(value_id)
    return identity


def _kernel_identity(kernel: Kernel[Any]) -> dict[str, object]:
    source_path = inspect.getsourcefile(kernel.fn)
    if source_path is None:
        source_sha256 = hashlib.sha256(kernel.kernel_source().encode()).hexdigest()
    else:
        source_sha256 = _file_sha256(Path(source_path))
    closure = inspect.getclosurevars(kernel.fn)
    dependencies = {
        "globals": {
            name: _canonical_value(value)
            for name, value in sorted(closure.globals.items())
        },
        "nonlocals": {
            name: _canonical_value(value)
            for name, value in sorted(closure.nonlocals.items())
        },
    }
    key_identity = (
        None if kernel._key_fn is None else _callable_identity(kernel._key_fn)
    )
    return {
        "module": kernel.fn.__module__,
        "qualname": kernel.fn.__qualname__,
        "source_sha256": source_sha256,
        "dependencies": dependencies,
        # ``inspect.getclosurevars`` reports attribute names as unbound. The
        # containing module/callable identities above cover their definitions;
        # retain the names so changes in resolution still perturb the key.
        "unbound_names": sorted(closure.unbound),
        "key": key_identity,
    }


def _config_identity(config: Mapping[str, object]) -> object:
    result = _canonical_value(config)
    advanced_controls_file = config.get("advanced_controls_file")
    if advanced_controls_file is None:
        return result
    if not isinstance(advanced_controls_file, str):
        raise _Uncacheable("advanced_controls_file must be a path string")
    path = Path(advanced_controls_file)
    if not path.is_file():
        raise _Uncacheable("advanced_controls_file is not readable")
    return {
        "config": result,
        "advanced_controls_sha256": _file_sha256(path),
    }


def _settings_identity(kernel: Kernel[Any]) -> object:
    settings = kernel.settings.to_dict()
    # This seed is time-derived by default. It cannot affect a single explicit
    # config, while including it would make every fresh process miss.
    settings.pop("autotune_random_seed", None)
    return _canonical_value(settings)


def _input_identity(args: tuple[object, ...]) -> dict[str, object]:
    storage_labels: dict[tuple[str, int | None, int], int] = {}
    tensor_labels: dict[int, int] = {}
    arguments: list[object] = []
    storage_aliases: list[int | None] = []
    tensor_aliases: list[int | None] = []
    for value in args:
        if type(value) in (torch.Tensor, torch.nn.Parameter):
            tensor = value
            assert isinstance(tensor, torch.Tensor)
            if tensor.layout is not torch.strided:
                raise _Uncacheable("only strided tensors are supported")
            try:
                shape = [int(dim) for dim in tensor.shape]
                stride = [int(dim) for dim in tensor.stride()]
                storage_offset = int(tensor.storage_offset())
                storage = tensor.untyped_storage()
            except (RuntimeError, TypeError, ValueError) as error:
                raise _Uncacheable(
                    "symbolic or inaccessible tensor metadata"
                ) from error
            device = tensor.device
            storage_key = (device.type, device.index, storage._cdata)
            storage_label = storage_labels.setdefault(storage_key, len(storage_labels))
            tensor_label = tensor_labels.setdefault(id(tensor), len(tensor_labels))
            arguments.append(
                {
                    "kind": "tensor",
                    "type": f"{type(tensor).__module__}.{type(tensor).__qualname__}",
                    "shape": shape,
                    "stride": stride,
                    "dtype": str(tensor.dtype),
                    "device": str(device),
                    "storage_offset": storage_offset,
                    "requires_grad": tensor.requires_grad,
                }
            )
            storage_aliases.append(storage_label)
            tensor_aliases.append(tensor_label)
        elif isinstance(value, ConstExpr):
            arguments.append(
                {"kind": "constexpr", "value": _canonical_value(value.value)}
            )
            storage_aliases.append(None)
            tensor_aliases.append(None)
        elif value is None or type(value) in (
            bool,
            int,
            float,
            str,
            torch.dtype,
            torch.device,
        ):
            arguments.append(
                {
                    "kind": "value",
                    "type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "value": _canonical_value(value),
                }
            )
            storage_aliases.append(None)
            tensor_aliases.append(None)
        else:
            raise _Uncacheable(
                f"unsupported input {type(value).__module__}.{type(value).__qualname__}"
            )
    return {
        "arguments": arguments,
        "storage_aliases": storage_aliases,
        "tensor_aliases": tensor_aliases,
    }


def _hardware_identity(args: tuple[object, ...]) -> tuple[int, dict[str, object]]:
    devices: set[torch.device] = set()
    for value in args:
        if type(value) in (torch.Tensor, torch.nn.Parameter):
            tensor = cast("torch.Tensor", value)
            devices.add(tensor.device)
    if len(devices) != 1:
        raise _Uncacheable("exactly one tensor device is required")
    device = devices.pop()
    if device.type != "cuda" or not torch.cuda.is_available():
        raise _Uncacheable("the first artifact-cache version supports CUDA only")
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    return index, {
        "device_type": device.type,
        "device_index": index,
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "multiprocessor_count": properties.multi_processor_count,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
    }
