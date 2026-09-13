from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import json
import threading
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.kernel_compiler import KernelCompiler
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
import helion.language as hl
from helion.runtime.artifact_cache import _atomic_write
from helion.runtime.artifact_cache import _callable_identity
from helion.runtime.artifact_cache import _input_identity

if TYPE_CHECKING:
    from pathlib import Path
    import types

    from helion._compiler.host_function import HostFunction


def _shape_key(x: torch.Tensor, y: torch.Tensor) -> tuple[object, ...]:
    return x.shape, x.stride(), x.dtype, y.shape, y.stride(), y.dtype


def _artifact_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _artifact_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] - y[tile]
    return out


_HELPER_BIAS = 1


def _nested_helper(value: int) -> int:
    return value + _HELPER_BIAS


def _helper(value: int) -> int:
    return _nested_helper(value)


def _make_kernel(
    *,
    block_size: int = 32,
    static_shapes: bool = False,
    fn: types.FunctionType = _artifact_add,
) -> helion.runtime.Kernel[torch.Tensor]:
    return helion.kernel(
        fn,
        config=helion.Config(block_sizes=[block_size], indexing="pointer"),
        settings=helion.Settings(
            static_shapes=static_shapes,
            output_origin_lines=False,
        ),
        key=_shape_key,
    )


def _enable_cache(monkeypatch: pytest.MonkeyPatch, cache_dir: Path) -> None:
    monkeypatch.setenv("HELION_KERNEL_ARTIFACT_CACHE", "1")
    monkeypatch.setenv("HELION_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("HELION_SKIP_CACHE", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)


def test_input_identity_tracks_metadata_and_aliasing() -> None:
    base = torch.empty_strided((4, 4), (4, 1))
    other = torch.empty_strided((4, 4), (4, 1))
    transposed = base.T

    identity = _input_identity((base, base, other))
    assert identity["tensor_aliases"] == [0, 0, 1]
    assert identity["storage_aliases"] == [0, 0, 1]
    assert _input_identity((base, other)) != _input_identity((base, transposed))
    assert _input_identity((base, 1)) != _input_identity((base, 2))


def test_callable_identity_tracks_nested_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _callable_identity(_helper)
    monkeypatch.setitem(_nested_helper.__globals__, "_HELPER_BIAS", 2)
    assert _callable_identity(_helper) != original


def test_concurrent_atomic_writes_never_expose_partial(tmp_path: Path) -> None:
    destination = tmp_path / "entry.json"
    payloads = [bytes([value]) * (64 * 1024) for value in range(1, 5)]
    finished = threading.Event()
    observed: list[bytes] = []

    def read_while_writing() -> None:
        while not finished.is_set():
            with contextlib.suppress(FileNotFoundError):
                observed.append(destination.read_bytes())

    def write_repeated(data: bytes) -> None:
        for _ in range(20):
            _atomic_write(destination, data)

    with ThreadPoolExecutor(max_workers=5) as executor:
        reader = executor.submit(read_while_writing)
        writers = [executor.submit(write_repeated, payload) for payload in payloads]
        for writer in writers:
            writer.result()
        finished.set()
        reader.result()

    assert destination.read_bytes() in payloads
    assert observed
    assert all(value in payloads for value in observed)
    assert not list(tmp_path.glob(".*.tmp"))


@skipIfNotCUDA()
def test_fresh_kernel_loads_artifact_before_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_cache(monkeypatch, tmp_path)
    first = _make_kernel()
    x = torch.randn(257, device=DEVICE)
    y = torch.randn(257, device=DEVICE)
    torch.testing.assert_close(first(x, y), x + y)

    entries = list((tmp_path / "kernel_artifacts" / "v1").glob("*.json"))
    assert len(entries) == 1
    entry = json.loads(entries[0].read_text())
    assert entries[0].with_suffix(".py").is_file()
    assert entry["kernel_name"] == "_artifact_add"

    second = _make_kernel()
    x2 = torch.randn_like(x)
    y2 = torch.randn_like(y)
    with patch.object(
        KernelCompiler,
        "compile",
        side_effect=AssertionError("artifact hit reached KernelCompiler.compile"),
    ):
        torch.testing.assert_close(second(x2, y2), x2 + y2)
        torch.testing.assert_close(second(x2, y2), x2 + y2)
    assert not second._bound_kernels
    assert len(second._artifact_cache_hits) == 1

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = second(x2, y2)
    x2.copy_(torch.randn_like(x2))
    y2.copy_(torch.randn_like(y2))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, x2 + y2)


@skipIfNotCUDA()
def test_shape_config_and_disable_take_normal_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_cache(monkeypatch, tmp_path)
    x = torch.randn(128, device=DEVICE)
    y = torch.randn(128, device=DEVICE)
    _make_kernel()(x, y)

    def expect_compile(
        kernel: helion.runtime.Kernel[torch.Tensor],
        new_x: torch.Tensor,
        new_y: torch.Tensor,
    ) -> None:
        with (
            patch.object(
                KernelCompiler,
                "compile",
                side_effect=AssertionError("expected normal compilation path"),
            ),
            pytest.raises(AssertionError, match="expected normal compilation"),
        ):
            kernel(new_x, new_y)

    expect_compile(
        _make_kernel(),
        torch.randn(129, device=DEVICE),
        torch.randn(129, device=DEVICE),
    )
    expect_compile(_make_kernel(block_size=64), x, y)
    expect_compile(_make_kernel(static_shapes=True), x, y)
    expect_compile(_make_kernel(fn=_artifact_sub), x, y)
    expect_compile(_make_kernel(), x.half(), y.half())
    expect_compile(_make_kernel(), x, x)
    offset_x = torch.randn(129, device=DEVICE)[1:]
    expect_compile(_make_kernel(), offset_x, y)

    strided_x = torch.randn(256, device=DEVICE)[::2]
    strided_y = torch.randn(256, device=DEVICE)[::2]
    expect_compile(_make_kernel(), strided_x, strided_y)

    reset_kernel = _make_kernel()
    reset_kernel(x, y)
    assert reset_kernel._artifact_cache_hits
    reset_kernel.reset()
    expect_compile(reset_kernel, x, y)

    monkeypatch.setenv("HELION_KERNEL_ARTIFACT_CACHE", "0")
    expect_compile(_make_kernel(), x, y)
    monkeypatch.setenv("HELION_KERNEL_ARTIFACT_CACHE", "1")
    monkeypatch.setenv("HELION_SKIP_CACHE", "1")
    expect_compile(_make_kernel(), x, y)


@skipIfNotCUDA()
def test_corrupt_artifact_falls_back_and_repairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_cache(monkeypatch, tmp_path)
    x = torch.randn(64, device=DEVICE)
    y = torch.randn(64, device=DEVICE)
    _make_kernel()(x, y)
    entry_path = next((tmp_path / "kernel_artifacts" / "v1").glob("*.json"))
    module_path = entry_path.with_suffix(".py")
    module_path.write_text("this is not valid Python\n")

    original_compile = KernelCompiler.compile
    compile_calls = 0

    def counted_compile(
        compiler: KernelCompiler,
        fn: types.FunctionType,
        fake_args: list[object],
        constexpr_args: dict[str, object],
    ) -> HostFunction:
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(compiler, fn, fake_args, constexpr_args)

    fresh = _make_kernel()
    with patch.object(KernelCompiler, "compile", counted_compile):
        torch.testing.assert_close(fresh(x, y), x + y)
    assert compile_calls == 1
    entry = json.loads(entry_path.read_text())
    assert module_path.read_text().startswith("from __future__ import annotations")
    assert entry["source_sha256"]
