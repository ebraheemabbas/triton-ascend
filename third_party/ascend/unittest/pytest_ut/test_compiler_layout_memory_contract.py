# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
"""Source-level contracts for the layout / memory-access compiler closure.

These tests intentionally load ``backend/compiler.py`` from this checkout.
The installed Triton package can point at another worktree, so importing
``triton.backends.ascend.compiler`` directly would not validate the source
being changed here.
"""

import ast
import importlib.util
import itertools
import json
import sys
import types
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.backend("none")

_UNSET = object()


def _stub_ub_size_in_kbytes_for_arch(arch):
    """Mirror the raw UB architecture table for the compiler import shim.

    The table itself is covered by the source-loaded backend-utils test.  This
    local shim keeps this compiler-only contract independent of the installed
    Ascend package while retaining meaningful architecture-budget assertions.
    """
    if not isinstance(arch, str) or not arch:
        return 0
    if arch.startswith(("Ascend910_95", "Ascend950")):
        return 256
    if arch.startswith((
            "Ascend910A",
            "Ascend910B",
            "Ascend910D",
            "Ascend910_93",
            "Ascend310B",
    )):
        return 192
    return 0


def _stub_graph_ub_budget_bytes_for_arch(arch):
    return _stub_ub_size_in_kbytes_for_arch(arch) * 1024 * 80 // 100


class _FakeModule:

    def __init__(self, events):
        self.context = object()
        self._events = events
        self._string_count = 0

    def __str__(self):
        self._events.append(f"str:{self._string_count}")
        self._string_count += 1
        return "module {}"


class _FakePassManager:

    def __init__(self, events):
        self._events = events

    def enable_debug(self):
        self._events.append("enable_debug")

    def run(self, _module, _pipeline_name):
        self._events.append("run_row")


@pytest.fixture(scope="module")
def compiler_module():
    """Load this checkout's compiler without depending on installed Ascend utils.

    The test invokes only ``ttir_to_npubin`` and replaces all external tool
    interactions below.  A tiny import-time shim keeps the source-level
    contract test runnable when the installed Triton wheel predates an import
    added by this checkout (for example ``_enable_msdebug``).
    """
    compiler_path = Path(__file__).resolve().parents[2] / "backend" / "compiler.py"
    module_name = "triton.backends.ascend.compiler_layout_memory_contract_under_test"
    utils_name = "triton.backends.ascend.utils"
    driver_name = "triton.backends.ascend.driver"
    cache_name = "triton.runtime.cache"

    def return_false(*_args, **_kwargs):
        return False

    def remove_deprecated_npu_options(options, *, in_place=False):
        normalized = options if in_place else dict(options)
        normalized.pop("compile_on_910_95", None)
        return normalized

    utils_stub = types.ModuleType(utils_name)
    utils_path = compiler_path.with_name("utils.py")
    utils_tree = ast.parse(utils_path.read_text())
    mode_helper = next(node for node in utils_tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "_multibuffer_mode_to_tuple")
    exec(compile(ast.Module(body=[mode_helper], type_ignores=[]), str(utils_path), "exec"), utils_stub.__dict__)
    for name in (
            "_check_bishengir_api_change",
            "_check_bishengir_able_save_ir",
            "_check_bishengir_is_regbased",
            "_enable_print_ub_bits",
            "_enable_dump_memory_info",
            "_enable_msdebug",
            "_is_ascend_sanitizer_enabled",
            "_is_debug_line_info_disabled",
            "_is_auto_map_parallel_blocks_enabled",
            "force_disable_ffts",
            "is_cann_version_at_least",
    ):
        setattr(utils_stub, name, return_false)
    for name in (
            "_get_kernel_target",
            "_get_triton_adapter_opt_path",
            "_get_triton_mlir_opt_path",
            "_get_triton_opt_path",
            "_get_bishengir_opt_path",
    ):
        setattr(utils_stub, name, lambda *_args, **_kwargs: "")
    utils_stub._get_npucompiler_path = lambda *_args, **_kwargs: ("", {})
    utils_stub._get_auto_blockify_blacklist_reasons = lambda *_args, **_kwargs: []
    utils_stub._warn_auto_blockify_disabled = lambda *_args, **_kwargs: None
    utils_stub._remove_deprecated_npu_options = remove_deprecated_npu_options
    utils_stub._warn_deprecated_npu_option = lambda name: warnings.warn(
        f"Ascend compile option '{name}' is deprecated and ignored.", FutureWarning)
    utils_stub._warn_deprecated_ascend_env_vars = lambda: None
    utils_stub.downgrade_llir = lambda llir: llir
    utils_stub.get_cann_version_file_hash = lambda: ""
    utils_stub.graph_ub_budget_bytes_for_arch = _stub_graph_ub_budget_bytes_for_arch
    utils_stub.ub_size_in_kbytes_for_arch = _stub_ub_size_in_kbytes_for_arch

    class UnusedNPUUtils:
        pass

    driver_stub = types.ModuleType(driver_name)
    driver_stub.NPUUtils = UnusedNPUUtils

    cache_stub = types.ModuleType(cache_name)
    cache_stub._base32 = lambda value: str(value)
    dump_dir = Path("/fake/dump")
    cache_stub.get_dump_manager = lambda *_args, **_kwargs: SimpleNamespace(
        cache_dir=str(dump_dir),
        _make_path=lambda filename: str(dump_dir / filename),
        put=lambda *_args, **_kwargs: None,
    )

    # Initialize the installed Triton package before temporarily replacing its
    # cache module below.  Loading compiler.py starts from triton._C; doing it
    # in the reverse order would make unrelated backend imports see the tiny
    # test stub instead of the runtime cache API they require.
    importlib.import_module("triton")

    previous_utils = sys.modules.get(utils_name)
    previous_driver = sys.modules.get(driver_name)
    previous_cache = sys.modules.get(cache_name)
    sys.modules[utils_name] = utils_stub
    sys.modules[driver_name] = driver_stub
    sys.modules[cache_name] = cache_stub
    sys.modules.pop(module_name, None)
    toolchain_name = "triton.backends.ascend.toolchain"
    previous_toolchain = sys.modules.get(toolchain_name)
    toolchain_spec = importlib.util.spec_from_file_location(toolchain_name, compiler_path.with_name("toolchain.py"))
    toolchain_module = importlib.util.module_from_spec(toolchain_spec)
    sys.modules[toolchain_name] = toolchain_module
    toolchain_spec.loader.exec_module(toolchain_module)
    debug_line_rewriter_name = "triton.backends.ascend.debug_line_rewriter"
    previous_debug_line_rewriter = sys.modules.get(debug_line_rewriter_name)
    sys.modules.pop(debug_line_rewriter_name, None)
    try:
        debug_line_rewriter_path = compiler_path.with_name("debug_line_rewriter.py")
        debug_line_rewriter_spec = importlib.util.spec_from_file_location(debug_line_rewriter_name,
                                                                          debug_line_rewriter_path)
        debug_line_rewriter = importlib.util.module_from_spec(debug_line_rewriter_spec)
        assert debug_line_rewriter_spec is not None and debug_line_rewriter_spec.loader is not None
        sys.modules[debug_line_rewriter_name] = debug_line_rewriter
        debug_line_rewriter_spec.loader.exec_module(debug_line_rewriter)
        spec = importlib.util.spec_from_file_location(module_name, compiler_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_toolchain is None:
            sys.modules.pop(toolchain_name, None)
        else:
            sys.modules[toolchain_name] = previous_toolchain
        if previous_debug_line_rewriter is None:
            sys.modules.pop(debug_line_rewriter_name, None)
        else:
            sys.modules[debug_line_rewriter_name] = previous_debug_line_rewriter
        if previous_utils is None:
            sys.modules.pop(utils_name, None)
        else:
            sys.modules[utils_name] = previous_utils
        if previous_driver is None:
            sys.modules.pop(driver_name, None)
        else:
            sys.modules[driver_name] = previous_driver
        if previous_cache is None:
            sys.modules.pop(cache_name, None)
        else:
            sys.modules[cache_name] = previous_cache
    return module


def _parse_options(compiler, arch, opts=None):
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch=arch))
    return backend.parse_options({} if opts is None else opts)


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("arch", "requested_capacity", "expected_capacity"),
    (
        ("Ascend910B1", _UNSET, 192 * 1024 * 80 // 100),
        ("Ascend910B1", None, 192 * 1024 * 80 // 100),
        ("Ascend910_9581", None, 256 * 1024 * 80 // 100),
        ("Ascend950A3", None, 256 * 1024 * 80 // 100),
        ("Ascend910B1", 0, 0),
        ("Ascend910B1", 4096, 4096),
        ("Ascend910B1", 192 * 1024 * 80 // 100 + 1, 192 * 1024 * 80 // 100),
        ("Ascend910_9581", 256 * 1024 * 80 // 100 + 1, 256 * 1024 * 80 // 100),
        ("unknown-arch", None, 0),
    ),
)
def test_npu_options_normalizes_graph_ub_budget(compiler_module, arch, requested_capacity, expected_capacity):
    """Direct NPUOptions users receive the same final integer as JIT users."""
    kwargs = {"arch": arch}
    if requested_capacity is not _UNSET:
        kwargs["graph_optimize_ub_capacity_bytes"] = requested_capacity

    options = compiler_module.NPUOptions(**kwargs)

    assert options.graph_optimize_ub_capacity_bytes == expected_capacity


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("arch", "requested_capacity", "expected_capacity"),
    (
        ("Ascend910B1", _UNSET, 192 * 1024 * 80 // 100),
        ("Ascend910B1", None, 192 * 1024 * 80 // 100),
        ("Ascend910_9581", None, 256 * 1024 * 80 // 100),
        ("Ascend950A3", None, 256 * 1024 * 80 // 100),
        ("Ascend910B1", 0, 0),
        ("Ascend910B1", 4096, 4096),
        ("Ascend910B1", 192 * 1024 * 80 // 100 + 1, 192 * 1024 * 80 // 100),
    ),
)
def test_parse_options_normalizes_graph_ub_budget(compiler_module, arch, requested_capacity, expected_capacity):
    opts = {}
    if requested_capacity is not _UNSET:
        opts["graph_optimize_ub_capacity_bytes"] = requested_capacity

    options = _parse_options(compiler_module, arch, opts)

    assert options.arch == arch
    assert not hasattr(options, "_arch")
    assert options.graph_optimize_ub_capacity_bytes == expected_capacity


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_normalized_graph_ub_budget_contributes_to_npu_hash(compiler_module):
    auto = compiler_module.NPUOptions(arch="Ascend910B1")
    explicit_none = compiler_module.NPUOptions(arch="Ascend910B1", graph_optimize_ub_capacity_bytes=None)
    disabled = compiler_module.NPUOptions(arch="Ascend910B1", graph_optimize_ub_capacity_bytes=0)
    small = compiler_module.NPUOptions(arch="Ascend910B1", graph_optimize_ub_capacity_bytes=4096)
    clamped = compiler_module.NPUOptions(arch="Ascend910B1",
                                         graph_optimize_ub_capacity_bytes=192 * 1024 * 80 // 100 + 1)

    assert auto.__dict__["graph_optimize_ub_capacity_bytes"] == 192 * 1024 * 80 // 100
    assert explicit_none.graph_optimize_ub_capacity_bytes == 192 * 1024 * 80 // 100
    assert clamped.graph_optimize_ub_capacity_bytes == 192 * 1024 * 80 // 100
    assert auto.hash() == explicit_none.hash() == clamped.hash()
    assert auto.hash() != disabled.hash()
    assert auto.hash() != small.hash()


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("requested_capacity", "error_type"),
    (
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ),
)
def test_npu_options_rejects_invalid_graph_ub_budget_requests(compiler_module, requested_capacity, error_type):
    with pytest.raises(error_type):
        compiler_module.NPUOptions(
            arch="Ascend910B1",
            graph_optimize_ub_capacity_bytes=requested_capacity,
        )


def _make_opt(
    *,
    is_pure_simt,
    superblock_factor=0,
    simt_optimization_mode=0,
    shared_mem_dynamic_size=None,
    disable_fma=False,
    compile_on_910_95=False,
):
    return SimpleNamespace(
        is_pure_simt=is_pure_simt,
        num_warps=4,
        warp_size=32,
        simt_optimization_mode=simt_optimization_mode,
        shared_mem_dynamic_size=shared_mem_dynamic_size,
        disable_fma=disable_fma,
        superblock_factor=superblock_factor,
        compile_on_910_95=compile_on_910_95,
    )


def _make_program_grid_contract(*transforms):
    return {
        "version": 2,
        "extent_source": "runtime_original_grid",
        "hidden_extent_axes": [0, 1],
        "hidden_argument_order": ["originalGridX", "originalGridY"],
        "hidden_argument_types": ["i32", "i32"],
        "transforms": list(transforms),
    }


def _run_ttir_to_npubin(
    compiler,
    monkeypatch,
    *,
    is_pure_simt=True,
    auto_map_enabled=False,
    has_blacklist_op=False,
    row_coalescing_applied=False,
    superblock_factor=0,
    common_options=(),
    simt_optimization_mode=0,
    resolved_simt_stack_limit=1152,
    shared_mem_dynamic_size=None,
    disable_fma=False,
):
    events = []
    commands = []
    module = _FakeModule(events)
    pass_manager = _FakePassManager(events)

    def parse_ttir_metadata(_ttir, metadata):
        events.append("parse")
        return {
            **metadata,
            "has_auto_blockify_blacklist_op": has_blacklist_op,
            "mix_mode": "aiv",
            "row_coalescing_applied": row_coalescing_applied,
        }

    def export_coalesce_metadata(_mod, _metadata, *, require_row_contract=False):
        events.append(f"export:{require_row_contract}")

    def run_bisheng(command, **_kwargs):
        commands.append(list(command))
        output = Path(command[command.index("-o") + 1] + ".o")
        output.write_bytes(b"npubin")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        compiler,
        "ir",
        SimpleNamespace(pass_manager=lambda _context: (events.append("pass_manager") or pass_manager)),
    )
    monkeypatch.setattr(compiler, "_parse_ttir_metadata", parse_ttir_metadata)
    monkeypatch.setattr(compiler, "_export_coalesce_metadata", export_coalesce_metadata)
    monkeypatch.setattr(
        compiler,
        "get_common_bishengir_compile_options",
        lambda _metadata: list(common_options),
    )
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bisheng", {}))
    monkeypatch.setattr(
        compiler,
        "_is_auto_map_parallel_blocks_enabled",
        lambda: auto_map_enabled,
    )

    # Keep this argv matrix independent of the host torch_npu configuration
    # while checking that Pure-SIMT uses the backend-resolved stack limit.
    def get_simt_stack_limit():
        return resolved_simt_stack_limit

    monkeypatch.setattr(compiler, "get_simt_stack_limit", get_simt_stack_limit)
    monkeypatch.setattr(compiler.subprocess, "run", run_bisheng)

    result = compiler.ttir_to_npubin(
        module,
        {"bisheng_options": None},
        _make_opt(
            is_pure_simt=is_pure_simt,
            superblock_factor=superblock_factor,
            simt_optimization_mode=simt_optimization_mode,
            shared_mem_dynamic_size=shared_mem_dynamic_size,
            disable_fma=disable_fma,
        ),
    )
    assert result == b"npubin"
    assert len(commands) == 1
    return events, commands[0]


def test_simt_stack_limit_is_not_a_compile_option(compiler_module):
    assert "simt_stack_limit" not in compiler_module.NPUOptions.__dataclass_fields__
    assert "simt_stack_limit" not in _parse_options(compiler_module, "Ascend910_9581").__dict__
    with pytest.raises(TypeError, match="simt_stack_limit"):
        compiler_module.NPUOptions(arch="Ascend910_9581", simt_stack_limit=8192)


def _run_linalg_to_npubin(compiler, monkeypatch, function_name, has_blacklist_op, compiler_help=""):
    commands = []
    parsed_metadata = defaultdict(
        lambda: None, {
            "hash": "layout-memory-contract-test",
            "target": SimpleNamespace(arch="Ascend910B"),
            "program_grid_transforms": None,
            "program_grid_mapping_applied": False,
            "row_coalescing_applied": False,
            "has_auto_blockify_blacklist_op": has_blacklist_op,
            "auto_blockify_enabled": not has_blacklist_op,
            "ptsm_cap_authorized": False,
            "mix_mode": "aiv",
        })

    class FakeNPUUtils:

        def has_device_limit(self):
            return False

        def get_arch(self):
            return "Ascend910B"

    def parse_linalg_metadata(linalg, _metadata):
        return linalg, parsed_metadata

    def run_bisheng(command, **_kwargs):
        if command[1:] == ["--help"]:
            return SimpleNamespace(returncode=0, stdout=compiler_help, stderr="")
        commands.append(list(command))
        Path(command[command.index("-o") + 1] + "_reloc.o").write_bytes(b"npubin")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(compiler, "_parse_linalg_metadata", parse_linalg_metadata)
    monkeypatch.setattr(compiler, "get_common_bishengir_compile_options", lambda _metadata: [])
    monkeypatch.setattr(compiler, "get_auto_bind_sub_block_option", lambda _metadata: False)
    monkeypatch.setattr(compiler, "NPUUtils", FakeNPUUtils)
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", {}))
    monkeypatch.setattr(compiler, "_is_auto_map_parallel_blocks_enabled", lambda: True)
    monkeypatch.setattr(compiler.subprocess, "run", run_bisheng)

    result = getattr(compiler, function_name)(
        "module {}",
        {},
        SimpleNamespace(debug=False, target_arch="Ascend950PR", is_pure_simt=False),
    )
    assert result == b"npubin"
    assert len(commands) == 1
    return commands[0]


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_export_coalesce_metadata_removes_attrs_and_marks_row(compiler_module, monkeypatch):
    removed = []

    def get_int_attr(module, name):
        return module.attrs.get(name)

    def remove_attr(module, name):
        removed.append(name)
        module.attrs.pop(name, None)

    monkeypatch.setattr(
        compiler_module,
        "ascend",
        SimpleNamespace(ir=SimpleNamespace(
            get_int_attr=get_int_attr,
            remove_attr=remove_attr,
        )),
    )

    coalesced = SimpleNamespace(attrs={
        "hacc.coalesce_factor": 4,
        "hacc.coalesce_axis": 2,
        "hacc.coalesce_grid_ceil_div": 1,
    })
    metadata = {}
    compiler_module._export_coalesce_metadata(coalesced, metadata)

    assert metadata == {
        "coalesce_factor": 4,
        "coalesce_axis": 2,
        "coalesce_grid_ceil_div": True,
        "row_coalescing_applied": True,
    }
    assert coalesced.attrs == {}
    assert removed == [
        "hacc.coalesce_factor",
        "hacc.coalesce_axis",
        "hacc.coalesce_grid_ceil_div",
    ]

    uncoalesced = SimpleNamespace(attrs={})
    uncoalesced_metadata = {}
    compiler_module._export_coalesce_metadata(uncoalesced, uncoalesced_metadata)
    assert uncoalesced_metadata == {
        "coalesce_factor": 1,
        "coalesce_axis": -1,
        "coalesce_grid_ceil_div": False,
        "row_coalescing_applied": False,
    }


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_export_coalesce_metadata_rejects_partial_row_contract(compiler_module, monkeypatch):

    def get_int_attr(module, name):
        return module.attrs.get(name)

    def remove_attr(module, name):
        module.attrs.pop(name, None)

    monkeypatch.setattr(
        compiler_module,
        "ascend",
        SimpleNamespace(ir=SimpleNamespace(
            get_int_attr=get_int_attr,
            remove_attr=remove_attr,
        )),
    )

    with pytest.raises(RuntimeError, match="launch contract"):
        compiler_module._export_coalesce_metadata(
            SimpleNamespace(attrs={"hacc.coalesce_factor": 4}),
            {},
            require_row_contract=True,
        )

    with pytest.raises(RuntimeError, match="RowCoalescing"):
        compiler_module._export_coalesce_metadata(
            SimpleNamespace(attrs={
                "hacc.coalesce_factor": 4,
                "hacc.coalesce_axis": 0,
            }),
            {},
            require_row_contract=True,
        )


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_ttir_to_npubin_exports_make_ttir_row_contract_only_for_pure_simt(compiler_module, monkeypatch):
    events, _command = _run_ttir_to_npubin(
        compiler_module,
        monkeypatch,
        is_pure_simt=True,
    )
    assert events == [
        "str:0",
        "parse",
        "export:True",
        "str:1",
    ]

    with monkeypatch.context() as pure_simt_off:
        events, _command = _run_ttir_to_npubin(
            compiler_module,
            pure_simt_off,
            is_pure_simt=False,
        )
    assert events == ["str:0", "parse"]


def _run_make_ttir_with_recorded_graph_options(compiler, monkeypatch, options):
    events = []
    module = _FakeModule(events)
    pass_manager = _FakePassManager(events)
    graph_calls = []

    def record(name):
        return lambda _pm, *args, **kwargs: events.append((name, args, kwargs))

    monkeypatch.setattr(
        compiler,
        "ir",
        SimpleNamespace(pass_manager=lambda _context: pass_manager),
    )
    monkeypatch.setattr(
        compiler,
        "passes",
        SimpleNamespace(
            common=SimpleNamespace(
                add_inliner=record("inliner"),
                add_canonicalizer=record("canonicalizer"),
                add_cse=record("cse"),
                add_licm=record("licm"),
                add_symbol_dce=record("symbol_dce"),
            ),
            ttir=SimpleNamespace(
                add_rewrite_tensor_descriptor_to_pointer=record("rewrite_tensor_descriptor_to_pointer"),
                add_combine=record("combine"),
                add_reorder_broadcast=record("reorder_broadcast"),
                add_loop_unroll=record("loop_unroll"),
            ),
        ),
    )
    monkeypatch.setattr(
        compiler,
        "ascend",
        SimpleNamespace(passes=SimpleNamespace(ttir=SimpleNamespace(
            add_graph_optimize=lambda _pm, **kwargs: graph_calls.append(kwargs)))),
    )

    assert compiler.make_ttir(module, {}, options) is module
    return events, graph_calls


def test_make_ttir_passes_canonical_compile_mode_to_graph_optimize(compiler_module, monkeypatch):
    options = SimpleNamespace(
        enable_graph_optimize=True,
        target_arch="Ascend910B1",
        compile_mode="simt_only",
        debug=False,
    )

    events, graph_calls = _run_make_ttir_with_recorded_graph_options(compiler_module, monkeypatch, options)

    assert graph_calls == [{
        "ub_capacity_bytes": 192 * 1024 * 80 // 100,
        "compile_mode": "simt_only",
    }]
    assert events[-1] == "run_row"


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_npu_options_keep_graph_remark_compatibility_default(compiler_module):
    """The legacy graph-remarks name remains discoverable with a fixed default."""
    options = compiler_module.NPUOptions(arch="Ascend910B1")
    assert options.__dict__["graph_optimize_emit_remarks"] is False


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("arch", "expected_capacity"),
    (
        ("Ascend910B1", 192 * 1024 * 80 // 100),
        ("Ascend910_9581", 256 * 1024 * 80 // 100),
        ("Ascend950A3", 256 * 1024 * 80 // 100),
        ("unknown-arch", 0),
    ),
)
def test_make_ttir_forwards_normalized_graph_ub_budget(compiler_module, monkeypatch, arch, expected_capacity):
    options = compiler_module.NPUOptions(arch=arch)

    events, graph_calls = _run_make_ttir_with_recorded_graph_options(compiler_module, monkeypatch, options)

    assert graph_calls[0]["ub_capacity_bytes"] == expected_capacity
    assert events[-1] == "run_row"


def test_ttir_to_npubin_auto_blockify_argv_matrix(compiler_module, monkeypatch):
    common_options = ["--common-before-pure-simt", "--common-after-pure-simt"]
    pure_simt_prefix = [
        "--enable-hivm-compile=false",
        "--enable-triton-ir-compile",
        "--pure-simt",
        "--num-warps=4",
        "--threads-per-warp=32",
        "--simt-optimization-mode=1000017",
        "--simt-stack-limit=64",
        "--shared-mem-dynamic-size=4096",
        "--disable-fma",
    ]
    auto_blockify_flag = "--enable-auto-blockify-loop"

    for env_enabled, blacklisted, row_applied, superblock in itertools.product(
        (False, True),
        (False, True),
        (False, True),
        (0, 7),
    ):
        with monkeypatch.context() as case_monkeypatch:
            _events, command = _run_ttir_to_npubin(
                compiler_module,
                case_monkeypatch,
                auto_map_enabled=env_enabled,
                has_blacklist_op=blacklisted,
                row_coalescing_applied=row_applied,
                superblock_factor=superblock,
                common_options=common_options,
                simt_optimization_mode=1000017,
                resolved_simt_stack_limit=64,
                shared_mem_dynamic_size=4096,
                disable_fma=True,
            )

        second_injection = env_enabled and not row_applied
        case = f"E={env_enabled}, B={blacklisted}, R={row_applied}, superblock={superblock}"

        expected_options = [*common_options, *pure_simt_prefix]
        if second_injection:
            expected_options.append(auto_blockify_flag)
            if superblock > 0:
                expected_options.append(f"--super-block-factor={superblock}")

        assert command[0] == "/fake/bisheng", case
        assert Path(command[1]).name == "kernel.ttir.mlir", case
        assert command[2:-2] == expected_options, case
        assert command[-2] == "-o", case
        assert Path(command[-1]).name == "kernel", case


@pytest.mark.parametrize(
    ("auto_map_enabled", "blacklisted", "row_applied", "transforms", "expected_auto", "expected_ptsm"),
    (
        (False, False, False, None, False, False),
        (True, False, False, None, True, False),
        (True, False, True, None, True, False),
        (True, True, False, None, False, False),
        (True, False, False, ((0, 1, 16, False, False), ), True, False),
        (True, False, False, ((0, 0, 64, True, True), ), False, True),
        (True, False, False, ((0, 1, 16, False, False), (1, 0, 4, True, True)), False, True),
    ),
)
def test_finalize_program_launch_policy_uses_linalg_ptsm_gate(
    compiler_module,
    monkeypatch,
    auto_map_enabled,
    blacklisted,
    row_applied,
    transforms,
    expected_auto,
    expected_ptsm,
):
    contract = None
    if transforms is not None:
        contract = _make_program_grid_contract(*(dict(
            order=order,
            kind="ceil_div",
            axis=axis,
            factor=factor,
            persistent_coverage=persistent,
            grid_stride_abi_verified=grid_stride,
        ) for order, axis, factor, persistent, grid_stride in transforms))
    metadata = {
        "program_grid_transforms": contract,
        "program_grid_mapping_applied": contract is not None,
        "row_coalescing_applied": row_applied,
        "has_auto_blockify_blacklist_op": blacklisted,
        "mix_mode": "aiv",
    }
    monkeypatch.setattr(compiler_module, "_is_auto_map_parallel_blocks_enabled", lambda: auto_map_enabled)

    compiler_module._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=False))

    assert metadata["auto_blockify_enabled"] is expected_auto
    assert metadata["ptsm_cap_authorized"] is expected_ptsm


@pytest.mark.parametrize(
    ("auto_map_enabled", "blacklisted", "row_applied", "expected_auto"),
    (
        (False, False, False, False),
        (False, True, False, False),
        (True, False, False, True),
        (True, True, False, True),
        (True, False, True, False),
        (True, True, True, False),
    ),
)
def test_finalize_program_launch_policy_pure_simt_ignores_blacklist(
    compiler_module,
    monkeypatch,
    auto_map_enabled,
    blacklisted,
    row_applied,
    expected_auto,
):
    metadata = {
        "program_grid_transforms": None,
        "program_grid_mapping_applied": False,
        "row_coalescing_applied": row_applied,
        "has_auto_blockify_blacklist_op": blacklisted,
        "mix_mode": "aiv",
    }
    monkeypatch.setattr(compiler_module, "_is_auto_map_parallel_blocks_enabled", lambda: auto_map_enabled)

    compiler_module._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=True))

    assert metadata["auto_blockify_enabled"] is expected_auto
    assert metadata["ptsm_cap_authorized"] is False


@pytest.mark.parametrize(
    "function_name",
    (
        "linalg_to_bin_enable_npu_compile_910_95",
        "linalg_to_bin_enable_npu_compile_A2_A3",
    ),
)
@pytest.mark.parametrize("has_blacklist_op", (False, True))
def test_non_pure_simt_linalg_compilers_keep_blacklist_auto_blockify_gate(
    compiler_module,
    monkeypatch,
    function_name,
    has_blacklist_op,
):
    command = _run_linalg_to_npubin(
        compiler_module,
        monkeypatch,
        function_name,
        has_blacklist_op,
    )

    assert ("--enable-auto-blockify-loop" in command) is not has_blacklist_op
    assert "--pure-simt" not in command


def test_default_compile_mode_keeps_the_91095_layout_memory_gate_prepared(compiler_module):
    """The canonical default is portable and enables the Ascend 950 template gate."""

    a2_default = compiler_module.NPUOptions(arch="Ascend910B1")
    assert a2_default.compile_on_910_95 is False
    assert a2_default.compile_mode == "simd_simt_template"
    assert a2_default.is_pure_simt is False

    a5_default = compiler_module.NPUOptions(arch="Ascend910_9589")
    assert a5_default.compile_on_910_95 is True
    assert a5_default.compile_mode == "simd_simt_template"
    assert a5_default.is_pure_simt is False

    with pytest.warns(FutureWarning, match="compile_on_910_95"):
        ignored_legacy_value = compiler_module.NPUOptions(
            arch="Ascend910_9589",
            compile_on_910_95=False,
        )
    assert ignored_legacy_value.compile_on_910_95 is True

    canonical = compiler_module.NPUOptions(
        arch="Ascend910_9589",
        compile_mode="simd_simt_template",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = compiler_module.NPUOptions(
            arch="Ascend910_9589",
            compile_mode="unstructured_in_simt",
        )
    assert not caught
    assert alias.compile_mode == canonical.compile_mode == "simd_simt_template"
    assert alias.hash() == canonical.hash()

    with pytest.raises(ValueError, match=r"invalid compile_mode='simt_template'"):
        compiler_module.NPUOptions(arch="Ascend910_9589", compile_mode="simt_template")

    explicit_simd = compiler_module.NPUOptions(arch="Ascend910_9589", compile_mode="simd")
    assert explicit_simd.compile_mode == "simd"
    assert explicit_simd.is_pure_simt is False

    explicit_template = compiler_module.NPUOptions(
        arch="Ascend910_9589",
        compile_mode="simd_simt_template",
    )
    assert explicit_template.compile_mode == "simd_simt_template"
    assert explicit_template.is_pure_simt is False

    explicit_only = compiler_module.NPUOptions(arch="Ascend910_9589", compile_mode="simt_only")
    assert explicit_only.compile_mode == "simt_only"
    assert explicit_only.is_pure_simt is True

    # Legacy spellings remain discoverable while compile_mode controls lowering.
    assert explicit_only.__dict__["force_simt_only"] is False
    assert explicit_template.__dict__["force_simt_template"] is False


@pytest.mark.parametrize("arch", ["Ascend910B4", "Ascend910_9391", "Ascend950PR"])
@pytest.mark.parametrize("mode", [
    {"gm": 4, "l1": 2, "l0c": 1, "ub": 2},
    {"future": 0, "gm": -1},
    {},
])
def test_multibuffer_mode_preserves_values_and_metadata(compiler_module, arch, mode):
    options = _parse_options(compiler_module, arch, {"multibuffer_mode": mode})
    expected = tuple(sorted(mode.items()))
    assert options.multibuffer_mode == expected
    assert options.num_stages is None
    assert options.multibuffer is True
    metadata = dict(options.__dict__, multibuffer_mode=json.loads(json.dumps(options.multibuffer_mode)))
    restored = _parse_options(compiler_module, arch, metadata)
    assert restored.multibuffer_mode == expected
    assert restored.hash() == options.hash()
    assert restored.hash() != _parse_options(compiler_module, arch).hash()


@pytest.mark.parametrize("mode", [
    "[(gm,2)]",
    True,
    2,
    {"gm": "2"},
    {1: 2},
    {"gm": 2.0},
    {"gm": True},
    ("gm", 2),
    (("gm", ), ),
    (("gm", 2, 3), ),
    ((1, 2), ),
    (("gm", 2.0), ),
    (("gm", True), ),
])
def test_multibuffer_mode_requires_string_keys_and_integer_counts(compiler_module, mode):
    with pytest.raises(TypeError, match="multibuffer_mode must be a dict"):
        compiler_module.NPUOptions(multibuffer_mode=mode)


@pytest.mark.parametrize("name,value", [
    ("limit_auto_multi_buffer_only_for_local_buffer", False),
    ("limit_auto_multi_buffer_of_local_buffer", "no-l0c"),
    ("limit_auto_multi_buffer_buffer", "only-vector"),
])
@pytest.mark.parametrize("mode", [None, {"ub": 2}])
def test_multibuffer_legacy_options_are_preserved_without_ta_warnings(compiler_module, name, value, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        options = compiler_module.NPUOptions(multibuffer_mode=mode, **{name: value})
    assert not caught
    assert getattr(options, name) == value
    assert options.multibuffer is True


def _capture_multibuffer_command(compiler, monkeypatch, is_a5, requested, derived=None, compiler_help=""):
    options = _parse_options(compiler, "Ascend950PR" if is_a5 else "Ascend910B4", requested)
    metadata = dict(options.__dict__, hash="layout-memory-contract-test", mix_mode="mix", bitcodes=None,
                    auto_blockify_enabled=False)
    metadata.update(derived or {})
    monkeypatch.setattr(compiler, "_parse_linalg_metadata", lambda source, meta: (source, meta))
    monkeypatch.setattr(compiler, "_finalize_program_launch_policy", lambda *_args: None)
    monkeypatch.setattr(compiler, "get_common_bishengir_compile_options", lambda _meta: [])
    monkeypatch.setattr(compiler, "get_auto_bind_sub_block_option", lambda _meta: True)
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/compiler", {}))
    monkeypatch.setattr(compiler, "NPUUtils",
                        lambda: SimpleNamespace(has_device_limit=lambda: False, get_arch=lambda: options.target_arch))
    monkeypatch.delenv("TRITON_ENABLE_LIBDEVICE", raising=False)

    class CommandCaptured(BaseException):
        pass

    commands = []

    def capture(command, **_kwargs):
        if command[1:] == ["--help"]:
            return SimpleNamespace(returncode=0, stdout=compiler_help, stderr="")
        commands.append(command)
        raise CommandCaptured

    monkeypatch.setattr(compiler.subprocess, "run", capture)
    compile_fn = (compiler.linalg_to_bin_enable_npu_compile_910_95
                  if is_a5 else compiler.linalg_to_bin_enable_npu_compile_A2_A3)
    with pytest.raises(CommandCaptured):
        compile_fn("module {}", metadata, options)
    assert len(commands) == 1
    return commands[0]


@pytest.mark.parametrize("is_a5", [False, True])
@pytest.mark.parametrize("requested,enabled", [({}, True), ({"num_stages": 1}, False), ({"num_stages": 3}, True),
                                               ({"multibuffer": False}, False)])
def test_multibuffer_legacy_command_stays_on_old_interface(compiler_module, monkeypatch, is_a5, requested, enabled):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, is_a5, requested)
    assert not any(arg.startswith("--multibuffer-mode=") for arg in command)
    assert f"--enable-auto-multi-buffer={enabled}" in command
    assert ("--limit-auto-multi-buffer-of-local-buffer=no-limit" in command) == is_a5


@pytest.mark.parametrize("is_a5", [False, True])
def test_multibuffer_mode_is_forwarded_alongside_existing_defaults(compiler_module, monkeypatch, is_a5):
    mode = {"ub": 2, "gm": 4, "l0c": 1, "l1": 2}
    command = _capture_multibuffer_command(compiler_module, monkeypatch, is_a5, {"multibuffer_mode": mode})
    assert command.count("--multibuffer-mode=[(gm,4),(l0c,1),(l1,2),(ub,2)]") == 1
    assert "--enable-auto-multi-buffer=True" in command
    assert ("--limit-auto-multi-buffer-of-local-buffer=no-limit" in command) == is_a5


@pytest.mark.parametrize("is_a5", [False, True])
@pytest.mark.parametrize("switch", [False, True])
def test_multibuffer_explicit_old_options_are_still_forwarded(compiler_module, monkeypatch, is_a5, switch):
    requested = {
        "multibuffer_mode": {"gm": 4, "l1": 2, "l0c": 2, "ub": 2},
        "multibuffer": switch,
        "limit_auto_multi_buffer_only_for_local_buffer": True,
        "limit_auto_multi_buffer_of_local_buffer": "no-l0c",
        "limit_auto_multi_buffer_buffer": "only-vector",
        "set_workspace_multibuffer": 0,
    }
    command = _capture_multibuffer_command(compiler_module, monkeypatch, is_a5, requested)
    assert "--multibuffer-mode=[(gm,4),(l0c,2),(l1,2),(ub,2)]" in command
    assert f"--enable-auto-multi-buffer={switch}" in command
    assert "--limit-auto-multi-buffer-only-for-local-buffer=True" in command
    assert "--limit-auto-multi-buffer-of-local-buffer=no-l0c" in command
    assert "--set-workspace-multibuffer=0" in command
    if is_a5:
        assert "--limit-auto-multi-buffer-buffer=only-vector" in command


def test_multibuffer_keeps_dynamic_cv_workspace_constraint(compiler_module, monkeypatch):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                           {"multibuffer_mode": {"gm": 4, "l1": 2, "l0c": 1, "ub": 2}},
                                           derived={"set_workspace_multibuffer": 0})
    assert "--set-workspace-multibuffer=0" in command


@pytest.mark.parametrize("is_a5", [False, True])
def test_multibuffer_value_semantics_are_left_to_npuir(compiler_module, monkeypatch, is_a5):
    # Level names and count ranges are vendor decisions; partial modes are forwarded as supplied.
    command = _capture_multibuffer_command(compiler_module, monkeypatch, is_a5,
                                           {"multibuffer_mode": {"future": 0, "ub": -1}})
    assert "--multibuffer-mode=[(future,0),(ub,-1)]" in command


@pytest.mark.parametrize("arch", ["Ascend910B4", "Ascend910_9391", "Ascend950PR"])
@pytest.mark.parametrize("stages", [0, 1, 2, 3])
@pytest.mark.parametrize("mode", [{"gm": 2, "l1": 2, "l0c": 2, "ub": 2}, {}])
def test_multibuffer_mode_rejects_num_stages_even_when_equivalent(compiler_module, arch, stages, mode):
    with pytest.raises(ValueError, match="num_stages and multibuffer_mode cannot be specified together"):
        _parse_options(compiler_module, arch, {"multibuffer_mode": mode, "num_stages": stages})


def test_multibuffer_default_num_stages_does_not_conflict(compiler_module):
    assert compiler_module.NPUOptions().num_stages == 2
    assert compiler_module.NPUOptions(num_stages=1).num_stages == 1
    options = compiler_module.NPUOptions(multibuffer_mode={"ub": 2}, num_stages=None)
    assert options.num_stages is None


@pytest.mark.parametrize("arch", ["Ascend910B4", "Ascend950PR"])
def test_multibuffer_dictionary_order_does_not_change_compiler_hash(compiler_module, arch):
    mode = {"ub": 2, "gm": 4, "l1": 2}
    options = _parse_options(compiler_module, arch, {"multibuffer_mode": mode})
    reordered = _parse_options(compiler_module, arch, {"multibuffer_mode": dict(reversed(list(mode.items())))})
    assert reordered.multibuffer_mode == options.multibuffer_mode
    assert reordered.hash() == options.hash()
    mode["ub"] = 8
    assert dict(options.multibuffer_mode)["ub"] == 2


@pytest.mark.parametrize("function_name", [
    "linalg_to_bin_enable_npu_compile_910_95",
    "linalg_to_bin_enable_npu_compile_A2_A3",
])
@pytest.mark.parametrize("advertised", [False, True])
@pytest.mark.parametrize("cann_92", [False, True])
def test_linalg_metaop_uses_selected_frontend_capability(compiler_module, monkeypatch, function_name, advertised,
                                                     cann_92):
    monkeypatch.setattr(compiler_module, "is_cann_version_at_least", lambda *_args: cann_92)
    help_text = "  --enable-optimized-metaop=<bool> - enable meta operations\n" if advertised else "  --enable-optimize-math\n"
    command = _run_linalg_to_npubin(compiler_module, monkeypatch, function_name, False, compiler_help=help_text)

    assert [arg for arg in command if arg.startswith("--enable-optimized-metaop=")] == (
        [f"--enable-optimized-metaop={cann_92}"] if advertised else [])
    assert not any(arg.startswith("--enable-optimize-math") for arg in command)


@pytest.mark.parametrize("help_text", ["", "  --enable-optimized-metaop-mode=<mode>\n",
                                       "Description mentions --enable-optimized-metaop but declares no option.\n"])
def test_metaop_requires_an_exact_advertised_option(compiler_module, monkeypatch, help_text):
    monkeypatch.setattr(compiler_module.subprocess, "run",
                        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=help_text))
    assert compiler_module._optimized_metaop_compile_options("/selected/frontend") == []


def test_metaop_help_failure_keeps_frontend_defaults(compiler_module, monkeypatch):
    monkeypatch.setattr(compiler_module.subprocess, "run",
                        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="--enable-optimized-metaop"))
    assert compiler_module._optimized_metaop_compile_options("/selected/frontend") == []


@pytest.mark.parametrize("timeout", [False, True])
def test_metaop_help_exception_keeps_frontend_defaults(compiler_module, monkeypatch, timeout):
    def fail(*_args, **_kwargs):
        if timeout:
            raise compiler_module.subprocess.TimeoutExpired("/selected/frontend", 10)
        raise OSError("frontend unavailable")

    monkeypatch.setattr(compiler_module.subprocess, "run", fail)
    assert compiler_module._optimized_metaop_compile_options("/selected/frontend") == []


_VECTOR_CONTROL_HELP = "  --enable-vf-operand-substitution=<bool>\n  --enable-fused-multiply-add=<bool>\n"


@pytest.mark.parametrize("name,flag", [
    ("enable_vf_operand_substitution", "--enable-vf-operand-substitution"),
    ("enable_fused_multiply_add", "--enable-fused-multiply-add"),
])
@pytest.mark.parametrize("value", [False, True])
def test_native_vector_controls_forward_without_dcvp(compiler_module, monkeypatch, name, flag, value):
    requested = {"enable_dynamic_cv_pipeline": False, name: value}
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, requested,
                                         compiler_help=_VECTOR_CONTROL_HELP)
    assert command.count(f"{flag}={value}") == 1
    assert not any("dynamic-cv" in arg or "cv-split" in arg for arg in command)


@pytest.mark.parametrize("dynamic,disabled,expected", [(False, False, []), (True, False, ["--enable-vf-operand-substitution=True"]),
                                                       (True, True, []), (False, True, [])])
def test_omitted_vector_controls_preserve_legacy_flags(compiler_module, monkeypatch, dynamic, disabled, expected):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                         {"enable_dynamic_cv_pipeline": dynamic},
                                         {"disable_vf_operand_substitution": disabled})
    assert [arg for arg in command if arg.startswith("--enable-vf-operand-substitution=")] == expected
    assert not any(arg.startswith("--enable-fused-multiply-add=") for arg in command)


def test_explicit_vf_on_cannot_bypass_safety_disable(compiler_module, monkeypatch):
    with pytest.raises(ValueError, match="compiler safety disable"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {"enable_vf_operand_substitution": True},
                                    {"disable_vf_operand_substitution": True}, _VECTOR_CONTROL_HELP)


def test_explicit_vf_off_overrides_legacy_enable(compiler_module, monkeypatch):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                         {"enable_dynamic_cv_pipeline": True, "enable_vf_operand_substitution": False},
                                         compiler_help=_VECTOR_CONTROL_HELP)
    assert "--enable-vf-operand-substitution=False" in command
    assert "--enable-vf-operand-substitution=True" not in command


@pytest.mark.parametrize("name", ["enable_vf_operand_substitution", "enable_fused_multiply_add"])
@pytest.mark.parametrize("value", [0, 1, "true", [], object()])
def test_vector_controls_require_optional_booleans(compiler_module, name, value):
    with pytest.raises(TypeError, match="boolean or None"):
        _parse_options(compiler_module, "Ascend950PR", {name: value})


@pytest.mark.parametrize("name", ["enable_vf_operand_substitution", "enable_fused_multiply_add"])
@pytest.mark.parametrize("arch,mode", [("Ascend910B4", "simd"), ("Ascend950PR", "simt_only")])
def test_vector_controls_reject_unsupported_routes(compiler_module, name, arch, mode):
    with pytest.raises(ValueError, match="SIMD compilation"):
        _parse_options(compiler_module, arch, {"compile_mode": mode, name: True})


def test_vector_fma_conflicting_permission_is_rejected(compiler_module):
    with pytest.raises(ValueError, match="conflicts with disable_fma"):
        _parse_options(compiler_module, "Ascend950PR", {"enable_fused_multiply_add": True, "disable_fma": True})


@pytest.mark.parametrize("name", ["enable_vf_operand_substitution", "enable_fused_multiply_add"])
def test_vector_controls_hash_and_serialization(compiler_module, name):
    hashes = set()
    for value in (None, False, True):
        options = _parse_options(compiler_module, "Ascend950PR", {name: value, "enable_dynamic_cv_pipeline": False})
        serialized = json.loads(json.dumps(options.__dict__))
        # Match JITFunction.preload: JSON list-valued options are restored to tuples.
        restored_values = {key: tuple(item) if isinstance(item, list) else item for key, item in serialized.items()}
        restored = _parse_options(compiler_module, "Ascend950PR", restored_values)
        assert getattr(restored, name) is value
        assert restored.hash() == options.hash()
        hashes.add(options.hash())
    assert len(hashes) == 3


@pytest.mark.parametrize("name", ["enable_vf_operand_substitution", "enable_fused_multiply_add"])
def test_explicit_vector_control_requires_frontend_capability(compiler_module, monkeypatch, name):
    with pytest.raises(RuntimeError, match="does not support requested"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {"enable_dynamic_cv_pipeline": False, name: True}, compiler_help="")


def test_backend_compilation_key_tracks_toolchain_replacement(compiler_module, monkeypatch):
    from triton.runtime import cache
    identity = ["compiler-and-templates-one"]
    monkeypatch.setattr(compiler_module, "toolchain_fingerprint", lambda *_args: identity[0])
    monkeypatch.setattr(compiler_module, "_get_npucompiler_path", lambda: ("/compiler", {}))
    monkeypatch.setattr(compiler_module, "_get_bishengir_opt_path", lambda: ("/reader", {}))
    monkeypatch.setattr(cache, "triton_key", lambda: "unchanged-triton")
    backend = compiler_module.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend950PR"))
    source = SimpleNamespace(hash=lambda: "unchanged-source")
    options = SimpleNamespace(hash=lambda: "unchanged-options-and-CANN-version")
    first = cache.get_cache_key(source, backend, options, {})
    assert cache.get_cache_key(source, backend, options, {}) == first
    identity[0] = "compiler-and-templates-two"
    assert cache.get_cache_key(source, backend, options, {}) != first


_ATTENTION_ANALYSIS = {"enable_mixed_cv": True, "enable_preload": True, "enable_dynamic_cv_pipeline": False}
_ATTENTION_ANALYSIS_HELP = "  --enable-cvp-attention-analysis=<bool>\n"


@pytest.mark.parametrize("value", [False, True])
def test_attention_analysis_option_reaches_native_command(compiler_module, monkeypatch, value):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                         {**_ATTENTION_ANALYSIS, "enable_cvp_attention_analysis": value},
                                         compiler_help=_ATTENTION_ANALYSIS_HELP)
    assert command.count(f"--enable-cvp-attention-analysis={value}") == 1
    assert not any("dynamic-cv" in arg or "cv-split" in arg for arg in command)


def test_omitted_attention_analysis_keeps_original_command(compiler_module, monkeypatch):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, _ATTENTION_ANALYSIS)
    assert not any("cvp-attention" in arg for arg in command)


@pytest.mark.parametrize("value", [0, 1, "true", [], object()])
def test_attention_analysis_option_requires_optional_bool(compiler_module, value):
    with pytest.raises(TypeError, match="boolean or None"):
        _parse_options(compiler_module, "Ascend950PR", {**_ATTENTION_ANALYSIS, "enable_cvp_attention_analysis": value})


@pytest.mark.parametrize("arch,mode", [("Ascend910B4", "simd"), ("Ascend950PR", "simt_only")])
def test_attention_analysis_option_requires_a5_simd(compiler_module, arch, mode):
    with pytest.raises(ValueError, match="SIMD compilation"):
        _parse_options(compiler_module, arch, {**_ATTENTION_ANALYSIS, "compile_mode": mode, "enable_cvp_attention_analysis": True})


@pytest.mark.parametrize("setting,value", [("enable_dynamic_cv_pipeline", True), ("enable_preload", False),
                                          ("enable_preload", None), ("enable_mixed_cv", False), ("enable_mixed_cv", None)])
def test_attention_analysis_option_rejects_unproved_route(compiler_module, setting, value):
    with pytest.raises(ValueError, match="requires explicit mixed-CV and preload with DCVP disabled"):
        _parse_options(compiler_module, "Ascend950PR", {**_ATTENTION_ANALYSIS, "enable_cvp_attention_analysis": True,
                                                       setting: value})


@pytest.mark.parametrize("help_text", ["", "  --enable-cvp-attention-analysis-extra=<bool>\n"])
def test_attention_analysis_option_requires_exact_frontend_capability(compiler_module, monkeypatch, help_text):
    with pytest.raises(RuntimeError, match="does not support requested"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {**_ATTENTION_ANALYSIS, "enable_cvp_attention_analysis": True}, compiler_help=help_text)


def test_attention_analysis_option_cache_and_serialization(compiler_module):
    hashes = set()
    for value in (None, False, True):
        parsed = _parse_options(compiler_module, "Ascend950PR", {**_ATTENTION_ANALYSIS, "enable_cvp_attention_analysis": value})
        data = json.loads(json.dumps(parsed.__dict__))
        restored = _parse_options(compiler_module, "Ascend950PR", {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})
        assert restored.enable_cvp_attention_analysis is value
        assert parsed.hash() == restored.hash()
        hashes.add(parsed.hash())
    assert len(hashes) == 3



_ATTENTION_SCORE_REMATERIALIZATION = {
    **_ATTENTION_ANALYSIS,
    "enable_vf_operand_substitution": True,
    "enable_fused_multiply_add": False,
}
_ATTENTION_SCORE_REMATERIALIZATION_HELP = (
    _VECTOR_CONTROL_HELP + "  --enable-cvp-attention-score-rematerialization=<bool>\n")
_ATTENTION_ROWS2_HELP = "  --enable-cvp-attention-rows2=<bool>\n"
_BOUNDARY_OPTIONS = ("enable_cvp_attention_first_output", "enable_cvp_attention_last_output",
                     "enable_cvp_attention_first_state")
_BOUNDARY_HELP = "".join("  --" + name.replace("_", "-") + "=<bool>\n" for name in _BOUNDARY_OPTIONS)


@pytest.mark.parametrize("values", itertools.product([None, False, True], repeat=3))
def test_attention_boundary_switches_are_independent(compiler_module, monkeypatch, values):
    requested = {**_ATTENTION_SCORE_REMATERIALIZATION,
                 **dict(zip(_BOUNDARY_OPTIONS, values))}
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, requested,
                                         compiler_help=_VECTOR_CONTROL_HELP + _BOUNDARY_HELP)
    for name, value in zip(_BOUNDARY_OPTIONS, values):
        flag = "--" + name.replace("_", "-")
        assert [arg for arg in command if arg.startswith(flag + "=")] == (
            [] if value is None else [f"{flag}={value}"])
    assert not any("score-rematerialization" in arg or "max-scaling" in arg or "rows2" in arg for arg in command)


@pytest.mark.parametrize("name", _BOUNDARY_OPTIONS)
@pytest.mark.parametrize("value", [0, 1, "true", "false", [], {}])
def test_attention_boundary_controls_are_typed(compiler_module, name, value):
    with pytest.raises(TypeError, match=name + " must be a boolean or None"):
        _parse_options(compiler_module, "Ascend950PR", {name: value})
    with pytest.raises(TypeError, match=name + " must be a boolean or None"):
        compiler_module._native_attention_boundary_options({name: value}, "/unused", {})


@pytest.mark.parametrize("name", _BOUNDARY_OPTIONS)
@pytest.mark.parametrize("setting,value,diagnostic", [
    ("enable_dynamic_cv_pipeline", True, "DCVP disabled"),
    ("enable_preload", False, "explicit mixed-CV"),
    ("enable_mixed_cv", False, "explicit mixed-CV"),
    ("enable_vf_operand_substitution", None, "explicit enable_vf_operand_substitution"),
    ("enable_fused_multiply_add", None, "explicit enable_fused_multiply_add"),
])
def test_attention_boundaries_require_explicit_policy(compiler_module, name, setting, value, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_SCORE_REMATERIALIZATION, name: True, setting: value})


@pytest.mark.parametrize("name", _BOUNDARY_OPTIONS)
@pytest.mark.parametrize("setting,value", [("parallel_mode", "simt"), ("mix_mode", "aiv"),
                                           ("compile_on_910_95", False)])
def test_attention_boundaries_recheck_final_route(compiler_module, monkeypatch, name, setting, value):
    with pytest.raises(ValueError, match="final Ascend 950/910_95 SIMD mixed-CV route"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {**_ATTENTION_SCORE_REMATERIALIZATION, name: True},
                                    {setting: value}, _VECTOR_CONTROL_HELP + _BOUNDARY_HELP)


@pytest.mark.parametrize("name", _BOUNDARY_OPTIONS)
@pytest.mark.parametrize("value", [False, True])
def test_attention_boundaries_require_each_capability(compiler_module, monkeypatch, name, value):
    missing = "--" + name.replace("_", "-")
    help_text = _VECTOR_CONTROL_HELP + "".join(
        "  --" + option.replace("_", "-") + "=<bool>\n" for option in _BOUNDARY_OPTIONS if option != name)
    with pytest.raises(RuntimeError, match="does not support requested " + missing):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {**_ATTENTION_SCORE_REMATERIALIZATION, name: value},
                                    compiler_help=help_text)


def test_attention_boundary_cache_separates_each_combination(compiler_module):
    hashes = set()
    for values in itertools.product([False, True], repeat=3):
        parsed = _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_SCORE_REMATERIALIZATION, **dict(zip(_BOUNDARY_OPTIONS, values))})
        data = json.loads(json.dumps(parsed.__dict__))
        restored = _parse_options(compiler_module, "Ascend950PR", {
            key: tuple(value) if isinstance(value, list) else value for key, value in data.items()})
        assert tuple(getattr(restored, key) for key in _BOUNDARY_OPTIONS) == values
        assert restored.hash() == parsed.hash()
        hashes.add(parsed.hash())
    assert len(hashes) == 8


@pytest.mark.parametrize("vf,fma,max_scaling,rows2", itertools.product([False, True], repeat=4))
def test_attention_independent_rows_and_fma_compose(compiler_module, monkeypatch, vf, fma, max_scaling, rows2):
    requested = {**_ATTENTION_SCORE_REMATERIALIZATION,
                 "enable_vf_operand_substitution": vf, "enable_fused_multiply_add": fma,
                 "enable_cvp_attention_score_rematerialization": True,
                 "enable_cvp_attention_max_scaling": max_scaling, "enable_cvp_attention_rows2": rows2}
    command = _capture_multibuffer_command(
        compiler_module, monkeypatch, True, requested,
        compiler_help=_ATTENTION_SCORE_REMATERIALIZATION_HELP
        + "  --enable-cvp-attention-max-scaling=<bool>\n" + _ATTENTION_ROWS2_HELP)
    for flag, value in [("vf-operand-substitution", vf), ("fused-multiply-add", fma),
                        ("cvp-attention-max-scaling", max_scaling), ("cvp-attention-rows2", rows2)]:
        assert command.count(f"--enable-{flag}={value}") == 1
    assert command.count("--enable-cvp-attention-score-rematerialization=True") == 1
    assert not any("dynamic-cv" in arg or "cv-split" in arg for arg in command)
    assert not any("reassoc" in arg or "reduction-chain" in arg for arg in command)


@pytest.mark.parametrize("rows2", [None, False, True])
def test_attention_rows2_independent_of_score_rewrite(compiler_module, monkeypatch, rows2):
    requested = {**_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_rows2": rows2}
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, requested,
                                         compiler_help=_VECTOR_CONTROL_HELP + _ATTENTION_ROWS2_HELP)
    assert [arg for arg in command if arg.startswith("--enable-cvp-attention-rows2=")] == (
        [] if rows2 is None else [f"--enable-cvp-attention-rows2={rows2}"])
    assert not any("score-rematerialization" in arg or "max-scaling" in arg for arg in command)


@pytest.mark.parametrize("setting,value,diagnostic", [
    ("enable_dynamic_cv_pipeline", True, "DCVP disabled"),
    ("enable_preload", False, "explicit mixed-CV"),
    ("enable_mixed_cv", False, "explicit mixed-CV"),
    ("enable_vf_operand_substitution", None, "explicit enable_vf_operand_substitution"),
    ("enable_fused_multiply_add", None, "explicit enable_fused_multiply_add"),
])
def test_attention_rows2_requires_explicit_route_and_math_policy(compiler_module, setting, value, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_rows2": True, setting: value})


@pytest.mark.parametrize("setting,value", [("parallel_mode", "simt"), ("mix_mode", "aiv"),
                                           ("compile_on_910_95", False)])
def test_attention_rows2_rechecks_final_route(compiler_module, monkeypatch, setting, value):
    with pytest.raises(ValueError, match="final Ascend 950/910_95 SIMD mixed-CV route"):
        _capture_multibuffer_command(
            compiler_module, monkeypatch, True,
            {**_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_rows2": True},
            {setting: value}, _VECTOR_CONTROL_HELP + _ATTENTION_ROWS2_HELP)


@pytest.mark.parametrize("value", [0, 1, "true", "false", [], {}])
def test_attention_rows2_has_typed_options(compiler_module, value):
    with pytest.raises(TypeError, match="enable_cvp_attention_rows2 must be a boolean or None"):
        _parse_options(compiler_module, "Ascend950PR", {"enable_cvp_attention_rows2": value})
    with pytest.raises(TypeError, match="enable_cvp_attention_rows2 must be a boolean or None"):
        compiler_module._native_attention_rows2_options({"enable_cvp_attention_rows2": value}, "/unused", {})


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("help_text", ["", "  --enable-cvp-attention-rows2-other=<bool>\n",
                                      "Description mentions --enable-cvp-attention-rows2 only.\n"])
def test_attention_rows2_requires_exact_capability(compiler_module, monkeypatch, value, help_text):
    with pytest.raises(RuntimeError, match="does not support requested --enable-cvp-attention-rows2"):
        _capture_multibuffer_command(
            compiler_module, monkeypatch, True,
            {**_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_rows2": value},
            compiler_help=_VECTOR_CONTROL_HELP + help_text)


def test_attention_rows2_fma_cache_and_serialization(compiler_module):
    hashes = set()
    for rows2, fma in itertools.product([None, False, True], [False, True]):
        parsed = _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_rows2": rows2,
            "enable_fused_multiply_add": fma})
        values = json.loads(json.dumps(parsed.__dict__))
        restored = _parse_options(compiler_module, "Ascend950PR", {
            key: tuple(item) if isinstance(item, list) else item for key, item in values.items()})
        assert restored.enable_cvp_attention_rows2 is rows2
        assert restored.enable_fused_multiply_add is fma
        assert restored.hash() == parsed.hash()
        hashes.add(parsed.hash())
    assert len(hashes) == 6


@pytest.mark.parametrize("value", [False, True])
def test_attention_score_rematerialization_reaches_native_command(compiler_module, monkeypatch, value):
    requested = {**_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_score_rematerialization": value}
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, requested,
                                         compiler_help=_ATTENTION_SCORE_REMATERIALIZATION_HELP)
    assert command.count(f"--enable-cvp-attention-score-rematerialization={value}") == 1
    assert "--enable-vf-operand-substitution=True" in command
    assert "--enable-fused-multiply-add=False" in command
    assert "--enable-preload=True" in command
    assert "--enable-mixed-cv=True" in command
    assert not any("dynamic-cv" in arg or "cv-split" in arg for arg in command)


@pytest.mark.parametrize("requested", [{}, {"enable_cvp_attention_score_rematerialization": None}])
def test_omitted_score_rematerialization_keeps_original_command(compiler_module, monkeypatch, requested):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, {**_ATTENTION_ANALYSIS, **requested})
    assert not any("cvp-attention-score-rematerialization" in arg for arg in command)


def test_disabled_score_rematerialization_preserves_legacy_route(compiler_module, monkeypatch):
    # Explicitly disabling this pass does not require its enable-only route or arithmetic settings.
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                         {"enable_cvp_attention_score_rematerialization": False},
                                         compiler_help=_ATTENTION_SCORE_REMATERIALIZATION_HELP)
    assert "--enable-cvp-attention-score-rematerialization=False" in command
    assert "--enable-vf-operand-substitution=True" in command  # Existing A5 DCVP default.
    assert not any(arg.startswith("--enable-fused-multiply-add=") for arg in command)


@pytest.mark.parametrize("value", [0, 1, "true", "false", [], {}, object()])
def test_attention_score_rematerialization_requires_optional_bool(compiler_module, value):
    with pytest.raises(TypeError, match="boolean or None"):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_score_rematerialization": value})


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("arch,mode", [("Ascend910B4", "simd"), ("Ascend950PR", "simt_only")])
def test_attention_score_rematerialization_requires_a5_simd(compiler_module, arch, mode, value):
    with pytest.raises(ValueError, match="SIMD compilation"):
        _parse_options(compiler_module, arch, {
            **_ATTENTION_ANALYSIS, "compile_mode": mode, "enable_cvp_attention_score_rematerialization": value})


@pytest.mark.parametrize("setting,value", [
    ("enable_dynamic_cv_pipeline", True), ("enable_dynamic_cv_pipeline", None), ("enable_dynamic_cv_pipeline", 0),
    ("enable_preload", False), ("enable_preload", None), ("enable_preload", 1),
    ("enable_mixed_cv", False), ("enable_mixed_cv", None), ("enable_mixed_cv", 1),
])
def test_attention_score_rematerialization_rejects_unproved_route(compiler_module, setting, value):
    with pytest.raises(ValueError, match="requires explicit mixed-CV and preload with DCVP disabled"):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_score_rematerialization": True, setting: value})


def test_attention_score_rematerialization_cannot_inherit_default_dcvp(compiler_module):
    requested = {**_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_score_rematerialization": True}
    requested.pop("enable_dynamic_cv_pipeline")
    with pytest.raises(ValueError, match="DCVP disabled"):
        _parse_options(compiler_module, "Ascend950PR", requested)


@pytest.mark.parametrize("setting,value,required", [
    ("enable_vf_operand_substitution", None, "explicit enable_vf_operand_substitution"),
    ("enable_fused_multiply_add", None, "explicit enable_fused_multiply_add"),
])
def test_attention_score_rematerialization_requires_qualified_arithmetic(compiler_module, setting, value, required):
    with pytest.raises(ValueError, match=required):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_score_rematerialization": True, setting: value})


@pytest.mark.parametrize("setting,value", [
    ("parallel_mode", "simt"), ("parallel_mode", "mix"), ("mix_mode", "aiv"), ("mix_mode", "aic"),
    ("compile_on_910_95", False),
])
def test_attention_score_rematerialization_rechecks_effective_route(compiler_module, monkeypatch, setting, value):
    with pytest.raises(ValueError, match="requires a final Ascend 950/910_95 SIMD mixed-CV route"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {**_ATTENTION_SCORE_REMATERIALIZATION,
                                     "enable_cvp_attention_score_rematerialization": True},
                                    {setting: value}, _ATTENTION_SCORE_REMATERIALIZATION_HELP)


def test_attention_score_rematerialization_honors_vf_safety_disable(compiler_module, monkeypatch):
    with pytest.raises(ValueError, match="compiler safety disable"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {**_ATTENTION_SCORE_REMATERIALIZATION,
                                     "enable_cvp_attention_score_rematerialization": True},
                                    {"disable_vf_operand_substitution": True}, _ATTENTION_SCORE_REMATERIALIZATION_HELP)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("help_text", [
    "", "  --enable-cvp-attention-score-rematerialization-extra=<bool>\n",
    "Description mentions --enable-cvp-attention-score-rematerialization but declares no option.\n",
])
def test_attention_score_rematerialization_requires_exact_capability(compiler_module, monkeypatch, value, help_text):
    with pytest.raises(RuntimeError, match="does not support requested --enable-cvp-attention-score-rematerialization"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {**_ATTENTION_SCORE_REMATERIALIZATION,
                                     "enable_cvp_attention_score_rematerialization": value},
                                    compiler_help=_VECTOR_CONTROL_HELP + help_text)


@pytest.mark.parametrize("failure", ["nonzero", "oserror", "timeout"])
def test_attention_score_rematerialization_requires_successful_capability_query(compiler_module, monkeypatch, failure):
    def query(*_args, **_kwargs):
        if failure == "oserror":
            raise OSError("frontend unavailable")
        if failure == "timeout":
            raise compiler_module.subprocess.TimeoutExpired("/selected/frontend", 10)
        return SimpleNamespace(returncode=1, stdout=_ATTENTION_SCORE_REMATERIALIZATION_HELP)

    monkeypatch.setattr(compiler_module.subprocess, "run", query)
    with pytest.raises(RuntimeError, match="does not support requested|cannot verify requested"):
        compiler_module._native_attention_score_rematerialization_options(
            {"enable_cvp_attention_score_rematerialization": False}, "/selected/frontend", {})


def test_attention_score_rematerialization_and_analysis_are_independent(compiler_module, monkeypatch):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                         {**_ATTENTION_SCORE_REMATERIALIZATION,
                                          "enable_cvp_attention_score_rematerialization": True,
                                          "enable_cvp_attention_analysis": False},
                                         compiler_help=_ATTENTION_SCORE_REMATERIALIZATION_HELP + _ATTENTION_ANALYSIS_HELP)
    assert "--enable-cvp-attention-score-rematerialization=True" in command
    assert "--enable-cvp-attention-analysis=False" in command


def test_attention_score_rematerialization_cache_and_serialization(compiler_module):
    hashes = set()
    for value in (None, False, True):
        parsed = _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_score_rematerialization": value})
        data = json.loads(json.dumps(parsed.__dict__))
        restored = _parse_options(compiler_module, "Ascend950PR", {
            key: tuple(item) if isinstance(item, list) else item for key, item in data.items()})
        assert restored.enable_cvp_attention_score_rematerialization is value
        assert restored.hash() == parsed.hash()
        hashes.add(parsed.hash())
    assert len(hashes) == 3



_ATTENTION_MAX_SCALING = {
    **_ATTENTION_SCORE_REMATERIALIZATION,
    "enable_cvp_attention_score_rematerialization": True,
}
_ATTENTION_MAX_SCALING_HELP = _ATTENTION_SCORE_REMATERIALIZATION_HELP + "  --enable-cvp-attention-max-scaling=<bool>\n"


@pytest.mark.parametrize("value", [False, True])
def test_attention_max_scaling_reaches_native_command(compiler_module, monkeypatch, value):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                         {**_ATTENTION_MAX_SCALING, "enable_cvp_attention_max_scaling": value},
                                         compiler_help=_ATTENTION_MAX_SCALING_HELP)
    assert command.count(f"--enable-cvp-attention-max-scaling={value}") == 1
    assert command.count("--enable-cvp-attention-score-rematerialization=True") == 1
    assert "--enable-vf-operand-substitution=True" in command
    assert "--enable-fused-multiply-add=False" in command
    assert not any("dynamic-cv" in arg or "cv-split" in arg for arg in command)


@pytest.mark.parametrize("requested", [{}, {"enable_cvp_attention_max_scaling": None}])
def test_omitted_max_scaling_preserves_score_rematerialization_command(compiler_module, monkeypatch, requested):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                         {**_ATTENTION_MAX_SCALING, **requested},
                                         compiler_help=_ATTENTION_SCORE_REMATERIALIZATION_HELP)
    assert not any("cvp-attention-max-scaling" in arg for arg in command)
    assert "--enable-cvp-attention-score-rematerialization=True" in command


def test_disabled_max_scaling_preserves_legacy_route_without_dependency(compiler_module, monkeypatch):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                         {"enable_cvp_attention_max_scaling": False},
                                         compiler_help="  --enable-cvp-attention-max-scaling=<bool>\n")
    assert "--enable-cvp-attention-max-scaling=False" in command
    assert not any("cvp-attention-score-rematerialization" in arg for arg in command)
    assert "--enable-vf-operand-substitution=True" in command  # Existing default DCVP route.
    assert not any(arg.startswith("--enable-fused-multiply-add=") for arg in command)


@pytest.mark.parametrize("value", [0, 1, "true", "false", [], {}, object()])
def test_attention_max_scaling_requires_optional_bool(compiler_module, value):
    with pytest.raises(TypeError, match="enable_cvp_attention_max_scaling must be a boolean or None"):
        _parse_options(compiler_module, "Ascend950PR", {**_ATTENTION_MAX_SCALING, "enable_cvp_attention_max_scaling": value})


@pytest.mark.parametrize("dependency", [None, False])
def test_attention_max_scaling_requires_score_rematerialization(compiler_module, dependency):
    with pytest.raises(ValueError, match="requires enable_cvp_attention_score_rematerialization=True"):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_MAX_SCALING, "enable_cvp_attention_max_scaling": True,
            "enable_cvp_attention_score_rematerialization": dependency})


def test_attention_max_scaling_requires_explicit_dependency(compiler_module):
    with pytest.raises(ValueError, match="requires enable_cvp_attention_score_rematerialization=True"):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_SCORE_REMATERIALIZATION, "enable_cvp_attention_max_scaling": True})


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("arch,mode", [("Ascend910B4", "simd"), ("Ascend950PR", "simt_only")])
def test_attention_max_scaling_requires_a5_simd(compiler_module, arch, mode, value):
    with pytest.raises(ValueError, match="SIMD compilation"):
        _parse_options(compiler_module, arch, {"compile_mode": mode, "enable_cvp_attention_max_scaling": value})


@pytest.mark.parametrize("setting,value,required", [
    ("enable_dynamic_cv_pipeline", True, "DCVP disabled"),
    ("enable_dynamic_cv_pipeline", None, "DCVP disabled"),
    ("enable_preload", False, "requires explicit mixed-CV and preload"),
    ("enable_mixed_cv", False, "requires explicit mixed-CV and preload"),
    ("enable_vf_operand_substitution", None, "explicit enable_vf_operand_substitution"),
    ("enable_fused_multiply_add", None, "explicit enable_fused_multiply_add"),
])
def test_attention_max_scaling_inherits_qualified_recipe(compiler_module, setting, value, required):
    with pytest.raises(ValueError, match=required):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_MAX_SCALING, "enable_cvp_attention_max_scaling": True, setting: value})


@pytest.mark.parametrize("dependency", [None, False])
def test_attention_max_scaling_rechecks_effective_dependency(compiler_module, monkeypatch, dependency):
    with pytest.raises(ValueError, match="requires enable_cvp_attention_score_rematerialization=True"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {**_ATTENTION_MAX_SCALING, "enable_cvp_attention_max_scaling": True},
                                    {"enable_cvp_attention_score_rematerialization": dependency}, _ATTENTION_MAX_SCALING_HELP)


@pytest.mark.parametrize("setting,value", [("compile_on_910_95", False), ("parallel_mode", "simt"), ("mix_mode", "aiv")])
def test_attention_max_scaling_native_helper_rechecks_effective_route(compiler_module, setting, value):
    metadata = {**_ATTENTION_MAX_SCALING, "enable_cvp_attention_max_scaling": True,
                "compile_on_910_95": True, "parallel_mode": "simd", "mix_mode": "mix", setting: value}
    with pytest.raises(ValueError, match="requires a final Ascend 950/910_95 SIMD mixed-CV route"):
        compiler_module._native_attention_max_scaling_options(metadata, "/selected/frontend", {})


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("help_text", [
    "", "  --enable-cvp-attention-max-scaling-extra=<bool>\n",
    "Description mentions --enable-cvp-attention-max-scaling but declares no option.\n",
])
def test_attention_max_scaling_requires_exact_capability(compiler_module, monkeypatch, value, help_text):
    with pytest.raises(RuntimeError, match="does not support requested --enable-cvp-attention-max-scaling"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {**_ATTENTION_MAX_SCALING, "enable_cvp_attention_max_scaling": value},
                                    compiler_help=_ATTENTION_SCORE_REMATERIALIZATION_HELP + help_text)


@pytest.mark.parametrize("failure", ["nonzero", "oserror", "timeout"])
def test_attention_max_scaling_requires_successful_capability_query(compiler_module, monkeypatch, failure):
    def query(*_args, **_kwargs):
        if failure == "oserror":
            raise OSError("frontend unavailable")
        if failure == "timeout":
            raise compiler_module.subprocess.TimeoutExpired("/selected/frontend", 10)
        return SimpleNamespace(returncode=1, stdout=_ATTENTION_MAX_SCALING_HELP)

    monkeypatch.setattr(compiler_module.subprocess, "run", query)
    with pytest.raises(RuntimeError, match="does not support requested|cannot verify requested"):
        compiler_module._native_attention_max_scaling_options({"enable_cvp_attention_max_scaling": False},
                                                            "/selected/frontend", {})


def test_attention_max_scaling_cache_and_serialization(compiler_module):
    hashes = set()
    for value in (None, False, True):
        parsed = _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_MAX_SCALING, "enable_cvp_attention_max_scaling": value})
        data = json.loads(json.dumps(parsed.__dict__))
        restored = _parse_options(compiler_module, "Ascend950PR", {
            key: tuple(item) if isinstance(item, list) else item for key, item in data.items()})
        assert restored.enable_cvp_attention_max_scaling is value
        assert restored.enable_cvp_attention_score_rematerialization is True
        assert restored.hash() == parsed.hash()
        hashes.add(parsed.hash())
    assert len(hashes) == 3



@pytest.mark.parametrize("max_scaling", [None, False, True])
def test_attention_vf_off_reaches_native_command(compiler_module, monkeypatch, max_scaling):
    requested = {**_ATTENTION_MAX_SCALING, "enable_vf_operand_substitution": False,
                 "enable_cvp_attention_max_scaling": max_scaling}
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, requested,
                                         compiler_help=_ATTENTION_MAX_SCALING_HELP)
    assert command.count("--enable-vf-operand-substitution=False") == 1
    assert "--enable-vf-operand-substitution=True" not in command
    assert command.count("--enable-cvp-attention-score-rematerialization=True") == 1
    assert command.count("--enable-fused-multiply-add=False") == 1
    assert "--enable-mixed-cv=True" in command and "--enable-preload=True" in command
    assert [arg for arg in command if arg.startswith("--enable-cvp-attention-max-scaling=")] == (
        [] if max_scaling is None else [f"--enable-cvp-attention-max-scaling={max_scaling}"])
    assert not any("dynamic-cv" in arg or "cv-split" in arg for arg in command)


@pytest.mark.parametrize("max_scaling", [False, True])
def test_attention_native_forwarding_uses_effective_vf_off(compiler_module, monkeypatch, max_scaling):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                         {**_ATTENTION_MAX_SCALING, "enable_cvp_attention_max_scaling": max_scaling},
                                         {"enable_vf_operand_substitution": False,
                                          "disable_vf_operand_substitution": True}, _ATTENTION_MAX_SCALING_HELP)
    assert "--enable-vf-operand-substitution=False" in command
    assert "--enable-vf-operand-substitution=True" not in command


@pytest.mark.parametrize("max_scaling", [False, True])
@pytest.mark.parametrize("effective", [None, 0, 1, "false"])
def test_attention_native_route_still_requires_explicit_vf_bool(compiler_module, monkeypatch, max_scaling, effective):
    with pytest.raises(ValueError, match="requires explicit enable_vf_operand_substitution"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {**_ATTENTION_MAX_SCALING, "enable_cvp_attention_max_scaling": max_scaling},
                                    {"enable_vf_operand_substitution": effective}, _ATTENTION_MAX_SCALING_HELP)


@pytest.mark.parametrize("max_scaling", [False, True])
@pytest.mark.parametrize("setting,value,required", [
    ("enable_dynamic_cv_pipeline", True, "DCVP disabled"),
    ("enable_dynamic_cv_pipeline", None, "DCVP disabled"),
    ("enable_preload", False, "requires explicit mixed-CV and preload"),
    ("enable_mixed_cv", False, "requires explicit mixed-CV and preload"),
    ("enable_fused_multiply_add", None, "explicit enable_fused_multiply_add"),
])
def test_attention_vf_off_preserves_recipe_restrictions(compiler_module, max_scaling, setting, value, required):
    with pytest.raises(ValueError, match=required):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_MAX_SCALING, "enable_vf_operand_substitution": False,
            "enable_cvp_attention_max_scaling": max_scaling, setting: value})


@pytest.mark.parametrize("max_scaling", [False, True])
@pytest.mark.parametrize("arch,mode", [("Ascend910B4", "simd"), ("Ascend950PR", "simt_only")])
def test_attention_vf_off_keeps_supported_target_route(compiler_module, max_scaling, arch, mode):
    with pytest.raises(ValueError, match="SIMD compilation"):
        _parse_options(compiler_module, arch, {
            **_ATTENTION_MAX_SCALING, "enable_vf_operand_substitution": False,
            "enable_cvp_attention_max_scaling": max_scaling, "compile_mode": mode})


@pytest.mark.parametrize("help_text,missing", [
    (_VECTOR_CONTROL_HELP, "--enable-cvp-attention-score-rematerialization"),
    (_ATTENTION_SCORE_REMATERIALIZATION_HELP, "--enable-cvp-attention-max-scaling"),
])
def test_attention_vf_off_still_requires_actual_native_capabilities(compiler_module, monkeypatch, help_text, missing):
    with pytest.raises(RuntimeError, match="does not support requested " + missing):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {**_ATTENTION_MAX_SCALING, "enable_vf_operand_substitution": False,
                                     "enable_cvp_attention_max_scaling": True}, compiler_help=help_text)


@pytest.mark.parametrize("dependency", [None, False])
def test_attention_vf_off_keeps_max_scaling_dependency(compiler_module, dependency):
    with pytest.raises(ValueError, match="requires enable_cvp_attention_score_rematerialization=True"):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_MAX_SCALING, "enable_vf_operand_substitution": False,
            "enable_cvp_attention_score_rematerialization": dependency, "enable_cvp_attention_max_scaling": True})


def test_attention_vf_recipe_cache_and_serialization(compiler_module):
    hashes = set()
    for vf in (False, True):
        for max_scaling in (False, True):
            parsed = _parse_options(compiler_module, "Ascend950PR", {
                **_ATTENTION_MAX_SCALING, "enable_vf_operand_substitution": vf,
                "enable_cvp_attention_max_scaling": max_scaling})
            data = json.loads(json.dumps(parsed.__dict__))
            restored = _parse_options(compiler_module, "Ascend950PR", {
                key: tuple(item) if isinstance(item, list) else item for key, item in data.items()})
            assert restored.enable_vf_operand_substitution is vf
            assert restored.enable_cvp_attention_max_scaling is max_scaling
            assert restored.enable_cvp_attention_score_rematerialization is True
            assert restored.hash() == parsed.hash()
            hashes.add(parsed.hash())
    assert len(hashes) == 4


_ATTENTION_LAYOUT_OPTIONS = (
    "enable_cvp_attention_key_major", "enable_cvp_attention_dense_p",
    "cvp_attention_producer_fusion", "cvp_attention_reduction_chains",
    "allow_cvp_attention_sum_reassociation",
)
_ATTENTION_LAYOUT_HELP = _VECTOR_CONTROL_HELP + "".join(
    "  --" + name.replace("_", "-") + "=<value>\n" for name in _ATTENTION_LAYOUT_OPTIONS)
_ATTENTION_LAYOUT_RECIPE = {
    **_ATTENTION_SCORE_REMATERIALIZATION,
    "enable_cvp_attention_key_major": True,
    "allow_cvp_attention_sum_reassociation": True,
}


def test_attention_layout_default_emits_no_new_flags(compiler_module, monkeypatch):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, {})
    assert not any(argument.startswith("--" + name.replace("_", "-"))
                   for name in _ATTENTION_LAYOUT_OPTIONS for argument in command)


@pytest.mark.parametrize("dense,fusion,chains,fma", itertools.product(
    [False, True], ["none", "p-state", "max-p-state"], [1, 4], [False, True]))
def test_attention_layout_controls_forward_independently(compiler_module, monkeypatch, dense, fusion, chains, fma):
    requested = {**_ATTENTION_LAYOUT_RECIPE, "enable_cvp_attention_dense_p": dense,
                 "cvp_attention_producer_fusion": fusion, "cvp_attention_reduction_chains": chains,
                 "enable_fused_multiply_add": fma}
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, requested,
                                         compiler_help=_ATTENTION_LAYOUT_HELP)
    for name in _ATTENTION_LAYOUT_OPTIONS:
        assert "--" + name.replace("_", "-") + "=" + str(requested[name]) in command
    assert "--enable-fused-multiply-add=" + str(fma) in command


@pytest.mark.parametrize("permission", [None, False])
@pytest.mark.parametrize("fma", [False, True])
def test_attention_layout_fma_does_not_authorize_sum_reassociation(compiler_module, permission, fma):
    with pytest.raises(ValueError, match="explicit FP32 sum reassociation permission"):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_LAYOUT_RECIPE, "allow_cvp_attention_sum_reassociation": permission,
            "enable_fused_multiply_add": fma})


@pytest.mark.parametrize("setting,value", [
    ("enable_cvp_attention_dense_p", True), ("cvp_attention_producer_fusion", "p-state"),
    ("cvp_attention_producer_fusion", "max-p-state"), ("cvp_attention_reduction_chains", 4),
])
def test_attention_layout_dependents_require_layout(compiler_module, setting, value):
    with pytest.raises(ValueError, match="require explicit key-major layout"):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_LAYOUT_RECIPE, "enable_cvp_attention_key_major": False, setting: value})


@pytest.mark.parametrize("setting,value", [
    ("cvp_attention_reduction_chains", True), ("cvp_attention_reduction_chains", "4"),
    ("cvp_attention_reduction_chains", 0), ("cvp_attention_reduction_chains", 3),
    ("cvp_attention_producer_fusion", True), ("cvp_attention_producer_fusion", "all"),
])
def test_attention_layout_rejects_invalid_policies(compiler_module, setting, value):
    with pytest.raises(ValueError, match=setting + " must be"):
        _parse_options(compiler_module, "Ascend950PR", {**_ATTENTION_LAYOUT_RECIPE, setting: value})


_ATTENTION_DENSE_TWO_CHAIN_RECIPE = {
    **_ATTENTION_LAYOUT_RECIPE,
    "enable_cvp_attention_score_rematerialization": True,
    "enable_cvp_attention_dense_p": True,
    "cvp_attention_producer_fusion": "max-p-state",
    "cvp_attention_reduction_chains": 2,
}


@pytest.mark.parametrize("fma", [False, True])
def test_attention_dense_two_chain_option_reaches_compiler(compiler_module, monkeypatch, fma):
    requested = {**_ATTENTION_DENSE_TWO_CHAIN_RECIPE, "enable_fused_multiply_add": fma}
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, requested,
                                         compiler_help=_ATTENTION_LAYOUT_HELP + _ATTENTION_SCORE_REMATERIALIZATION_HELP)
    assert "--cvp-attention-reduction-chains=2" in command
    assert "--enable-cvp-attention-dense-p=True" in command
    assert "--cvp-attention-producer-fusion=max-p-state" in command
    assert "--enable-fused-multiply-add=" + str(fma) in command


@pytest.mark.parametrize("setting,value", [
    ("enable_cvp_attention_dense_p", False), ("enable_cvp_attention_dense_p", None),
    ("cvp_attention_producer_fusion", "none"), ("cvp_attention_producer_fusion", "p-state"),
    ("enable_cvp_attention_score_rematerialization", False),
    ("allow_cvp_attention_sum_reassociation", False),
])
def test_attention_dense_two_chain_requires_complete_recipe(compiler_module, setting, value):
    with pytest.raises(ValueError, match="requires|reassociation"):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_DENSE_TWO_CHAIN_RECIPE, setting: value})


def test_attention_dense_two_chain_unroll_is_rejected(compiler_module):
    with pytest.raises(ValueError, match="Unroll"):
        _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_DENSE_TWO_CHAIN_RECIPE, "enable_cvp_query_pipeline": True,
            "cvp_query_pipeline_mode": "unroll"})


def test_attention_dense_two_chain_cache_and_json_roundtrip(compiler_module):
    hashes = set()
    for chains in (1, 2):
        for fma in (False, True):
            parsed = _parse_options(compiler_module, "Ascend950PR", {
                **_ATTENTION_DENSE_TWO_CHAIN_RECIPE, "cvp_attention_reduction_chains": chains,
                "enable_fused_multiply_add": fma})
            data = json.loads(json.dumps(parsed.__dict__))
            restored = _parse_options(compiler_module, "Ascend950PR", {
                key: tuple(value) if isinstance(value, list) else value for key, value in data.items()})
            assert restored.cvp_attention_reduction_chains == chains
            assert restored.hash() == parsed.hash()
            hashes.add(parsed.hash())
    assert len(hashes) == 4


@pytest.mark.parametrize("setting", ["enable_cvp_attention_key_major", "enable_cvp_attention_dense_p",
                                     "allow_cvp_attention_sum_reassociation"])
@pytest.mark.parametrize("value", [0, 1, "True", "False"])
def test_attention_layout_booleans_are_strict(compiler_module, setting, value):
    with pytest.raises(TypeError, match=setting + " must be a boolean or None"):
        _parse_options(compiler_module, "Ascend950PR", {**_ATTENTION_LAYOUT_RECIPE, setting: value})


def test_attention_layout_rejects_rows2_combination(compiler_module):
    with pytest.raises(ValueError, match="cannot use the independent rows2 schedule"):
        _parse_options(compiler_module, "Ascend950PR", {**_ATTENTION_LAYOUT_RECIPE, "enable_cvp_attention_rows2": True})


@pytest.mark.parametrize("setting,value", [("mix_mode", "cube"), ("parallel_mode", "simt"),
                                          ("compile_on_910_95", False), ("enable_dynamic_cv_pipeline", True)])
def test_attention_layout_rechecks_effective_route(compiler_module, monkeypatch, setting, value):
    with pytest.raises(ValueError, match="CVPipeline"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True, _ATTENTION_LAYOUT_RECIPE,
                                    {setting: value}, _ATTENTION_LAYOUT_HELP)


@pytest.mark.parametrize("help_text", ["", "  --enable-cvp-attention-key-major-extra=<bool>\n"])
def test_attention_layout_requires_selected_compiler_capability(compiler_module, monkeypatch, help_text):
    with pytest.raises(RuntimeError, match="does not support requested --enable-cvp-attention-key-major"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True, _ATTENTION_LAYOUT_RECIPE,
                                    compiler_help=_VECTOR_CONTROL_HELP + help_text)


def test_attention_layout_checks_each_requested_capability(compiler_module, monkeypatch):
    help_text = _ATTENTION_LAYOUT_HELP.replace("  --cvp-attention-producer-fusion=<value>\n", "")
    with pytest.raises(RuntimeError, match="does not support requested --cvp-attention-producer-fusion"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    {**_ATTENTION_LAYOUT_RECIPE, "cvp_attention_producer_fusion": "p-state"},
                                    compiler_help=help_text)


def test_attention_layout_absent_controls_do_not_probe_compiler(compiler_module, monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("Absent layout controls must not inspect a compiler")
    monkeypatch.setattr(compiler_module.subprocess, "run", unexpected)
    assert compiler_module._native_attention_layout_options({}, "/missing/compiler", {}) == []


def test_attention_layout_policy_cache_and_serialization(compiler_module):
    hashes = set()
    for dense, fusion, chains, fma in itertools.product([False, True], ["none", "p-state", "max-p-state"], [1, 4], [False, True]):
        parsed = _parse_options(compiler_module, "Ascend950PR", {
            **_ATTENTION_LAYOUT_RECIPE, "enable_cvp_attention_dense_p": dense,
            "cvp_attention_producer_fusion": fusion, "cvp_attention_reduction_chains": chains,
            "enable_fused_multiply_add": fma})
        data = json.loads(json.dumps(parsed.__dict__))
        restored = _parse_options(compiler_module, "Ascend950PR", {
            key: tuple(item) if isinstance(item, list) else item for key, item in data.items()})
        assert restored.hash() == parsed.hash()
        assert all(getattr(restored, name) == getattr(parsed, name) for name in _ATTENTION_LAYOUT_OPTIONS)
        hashes.add(parsed.hash())
    assert len(hashes) == 24


def _state_copy_forward_recipe(value):
    return {
        **_ATTENTION_LAYOUT_RECIPE,
        "enable_cvp_query_pipeline": True,
        "enable_cvp_query_state_copy_forward": value,
        "enable_preload": True,
        "enable_dynamic_cv_pipeline": False,
        "disable_auto_inject_block_sync": False,
    }


def test_state_copy_forward_default_keeps_existing_interface(compiler_module, monkeypatch):
    parsed = _parse_options(compiler_module, "Ascend950PR", {})
    assert parsed.enable_cvp_query_state_copy_forward is None
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, {})
    assert not any(arg.startswith("--enable-cvp-query-state-copy-forward") for arg in command)


@pytest.mark.parametrize("value", [False, True])
def test_state_copy_forward_has_independent_control_and_safe_buffers(compiler_module, monkeypatch, value):
    help_text = (_ATTENTION_LAYOUT_HELP
                 + "  --enable-cvp-query-pipeline=<bool>\n"
                 + "  --enable-cvp-query-state-copy-forward=<bool>\n")
    parsed = _parse_options(compiler_module, "Ascend950PR", _state_copy_forward_recipe(value))
    assert parsed.disable_tightly_coupled_buffer_reuse is True
    command = _capture_multibuffer_command(
        compiler_module, monkeypatch, True, _state_copy_forward_recipe(value), compiler_help=help_text)
    assert command.count("--enable-cvp-query-state-copy-forward=" + str(value)) == 1
    assert "--enable-cvp-query-pipeline=True" in command
    assert "--disable-tightly-coupled-buffer-reuse" in command


@pytest.mark.parametrize("value", [0, 1, "true", []])
def test_state_copy_forward_rejects_non_boolean_options(compiler_module, value):
    with pytest.raises(TypeError, match="enable_cvp_query_state_copy_forward.*boolean or None"):
        _parse_options(compiler_module, "Ascend950PR", _state_copy_forward_recipe(value))


def test_state_copy_forward_requires_query_pipeline(compiler_module):
    requested = {**_ATTENTION_LAYOUT_RECIPE, "enable_cvp_query_state_copy_forward": True,
                 "enable_cvp_query_pipeline": False}
    with pytest.raises(ValueError, match="copy forwarding requires an explicit query pipeline"):
        _parse_options(compiler_module, "Ascend950PR", requested)


def test_state_copy_forward_checks_selected_compiler_support(compiler_module, monkeypatch):
    help_text = _ATTENTION_LAYOUT_HELP + "  --enable-cvp-query-pipeline=<bool>\n"
    with pytest.raises(RuntimeError, match="does not support requested --enable-cvp-query-state-copy-forward"):
        _capture_multibuffer_command(
            compiler_module, monkeypatch, True, _state_copy_forward_recipe(True), compiler_help=help_text)


def test_state_copy_forward_is_serialized_and_distinguishes_cache_entries(compiler_module):
    hashes = set()
    for value in (None, False, True):
        parsed = _parse_options(compiler_module, "Ascend950PR", _state_copy_forward_recipe(value))
        serialized = json.loads(json.dumps(parsed.__dict__))
        restored = _parse_options(compiler_module, "Ascend950PR", {
            name: tuple(item) if isinstance(item, list) else item for name, item in serialized.items()})
        assert restored.enable_cvp_query_state_copy_forward is value
        assert restored.hash() == parsed.hash()
        hashes.add(parsed.hash())
    assert len(hashes) == 3


def _final_output_forward_recipe(value):
    return {
        **_ATTENTION_LAYOUT_RECIPE,
        "enable_cvp_query_pipeline": True,
        "enable_cvp_query_final_output_forward": value,
        "enable_preload": True,
        "enable_dynamic_cv_pipeline": False,
        "disable_auto_inject_block_sync": False,
    }


def test_final_output_forward_default_preserves_command(compiler_module, monkeypatch):
    parsed = _parse_options(compiler_module, "Ascend950PR", {})
    assert parsed.enable_cvp_query_final_output_forward is None
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, {})
    assert not any(arg.startswith("--enable-cvp-query-final-output-forward") for arg in command)


@pytest.mark.parametrize("value", [False, True])
def test_final_output_forward_independent_option_and_owned_buffers(compiler_module, monkeypatch, value):
    requested = _final_output_forward_recipe(value)
    parsed = _parse_options(compiler_module, "Ascend950PR", requested)
    assert parsed.disable_tightly_coupled_buffer_reuse is True
    assert parsed.enable_cvp_query_state_copy_forward is None
    help_text = (_ATTENTION_LAYOUT_HELP + "  --enable-cvp-query-pipeline=<bool>\n"
                 + "  --enable-cvp-query-final-output-forward=<bool>\n")
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, requested,
                                          compiler_help=help_text)
    assert command.count("--enable-cvp-query-final-output-forward=" + str(value)) == 1
    assert "--disable-tightly-coupled-buffer-reuse" in command
    assert not any(arg.startswith("--enable-cvp-query-state-copy-forward") for arg in command)


@pytest.mark.parametrize("value", [0, 1, "true", []])
def test_final_output_forward_rejects_non_boolean(compiler_module, value):
    with pytest.raises(TypeError, match="enable_cvp_query_final_output_forward.*boolean or None"):
        _parse_options(compiler_module, "Ascend950PR", _final_output_forward_recipe(value))


def test_final_output_forward_requires_explicit_query_pipeline(compiler_module):
    requested = {**_final_output_forward_recipe(True), "enable_cvp_query_pipeline": False}
    with pytest.raises(ValueError, match="final output forwarding requires an explicit query pipeline"):
        _parse_options(compiler_module, "Ascend950PR", requested)


def test_final_output_forward_requires_selected_compiler_capability(compiler_module, monkeypatch):
    help_text = _ATTENTION_LAYOUT_HELP + "  --enable-cvp-query-pipeline=<bool>\n"
    with pytest.raises(RuntimeError, match="does not support requested --enable-cvp-query-final-output-forward"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                    _final_output_forward_recipe(True), compiler_help=help_text)


def test_final_output_forward_distinguishes_serialized_cache_entries(compiler_module):
    hashes = set()
    for value in (None, False, True):
        parsed = _parse_options(compiler_module, "Ascend950PR", _final_output_forward_recipe(value))
        serialized = json.loads(json.dumps(parsed.__dict__))
        restored = _parse_options(compiler_module, "Ascend950PR", {
            name: tuple(item) if isinstance(item, list) else item for name, item in serialized.items()})
        assert restored.enable_cvp_query_final_output_forward is value
        assert restored.hash() == parsed.hash()
        hashes.add(parsed.hash())
    assert len(hashes) == 3


def test_input_and_final_output_forward_controls_are_cumulative(compiler_module, monkeypatch):
    requested = {**_final_output_forward_recipe(True), "enable_cvp_query_state_copy_forward": True}
    parsed = _parse_options(compiler_module, "Ascend950PR", requested)
    assert parsed.enable_cvp_query_state_copy_forward is True
    assert parsed.enable_cvp_query_final_output_forward is True
    help_text = (_ATTENTION_LAYOUT_HELP + "  --enable-cvp-query-pipeline=<bool>\n"
                 + "  --enable-cvp-query-state-copy-forward=<bool>\n"
                 + "  --enable-cvp-query-final-output-forward=<bool>\n")
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True, requested,
                                          compiler_help=help_text)
    assert "--enable-cvp-query-state-copy-forward=True" in command
    assert "--enable-cvp-query-final-output-forward=True" in command


def _output_copy_forward_recipe(value=True):
    return {**_final_output_forward_recipe(True),
            "enable_cvp_query_state_copy_forward": True,
            "enable_vf_operand_substitution": True,
            "enable_cvp_query_output_copy_forward": value}


@pytest.mark.parametrize("value", [None, False, True])
def test_output_copy_forward_option_reaches_compiler(compiler_module, monkeypatch, value):
    help_text = (_ATTENTION_LAYOUT_HELP + "  --enable-cvp-query-pipeline=<bool>\n"
                 + "  --enable-cvp-query-state-copy-forward=<bool>\n"
                 + "  --enable-cvp-query-final-output-forward=<bool>\n"
                 + "  --enable-cvp-query-output-copy-forward=<bool>\n")
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
        _output_copy_forward_recipe(value), compiler_help=help_text)
    forwarded = [arg for arg in command if arg.startswith("--enable-cvp-query-output-copy-forward=")]
    assert forwarded == ([] if value is None else ["--enable-cvp-query-output-copy-forward=" + str(value)])


@pytest.mark.parametrize("name", ["enable_cvp_query_pipeline", "enable_cvp_query_state_copy_forward",
                                 "enable_cvp_query_final_output_forward", "enable_vf_operand_substitution"])
@pytest.mark.parametrize("value", [None, False])
def test_output_copy_forward_requires_complete_recipe(compiler_module, name, value):
    with pytest.raises(ValueError, match="requires"):
        _parse_options(compiler_module, "Ascend950PR", {**_output_copy_forward_recipe(), name: value})


@pytest.mark.parametrize("value", [0, 1, "true", [], {}])
def test_output_copy_forward_rejects_non_booleans(compiler_module, value):
    with pytest.raises(TypeError, match="boolean"):
        _parse_options(compiler_module, "Ascend950PR", _output_copy_forward_recipe(value))


def test_output_copy_forward_rejects_unroll(compiler_module):
    with pytest.raises(ValueError, match="Unroll"):
        _parse_options(compiler_module, "Ascend950PR", {
            **_output_copy_forward_recipe(), "cvp_query_pipeline_mode": "unroll"})


def test_output_copy_forward_cache_and_metadata_roundtrip(compiler_module):
    hashes = set()
    for value in (None, False, True):
        parsed = _parse_options(compiler_module, "Ascend950PR", _output_copy_forward_recipe(value))
        serialized = json.loads(json.dumps(parsed.__dict__))
        restored = _parse_options(compiler_module, "Ascend950PR", {
            name: tuple(item) if isinstance(item, list) else item for name, item in serialized.items()})
        assert restored.enable_cvp_query_output_copy_forward is value
        assert restored.hash() == parsed.hash()
        hashes.add(parsed.hash())
    assert len(hashes) == 3


def _scalar_input_forward_recipe(value=True):
    return {**_output_copy_forward_recipe(), "enable_cvp_query_scalar_input_forward": value}


@pytest.mark.parametrize("value", [None, False, True])
def test_scalar_input_forward_reaches_compiler(compiler_module, monkeypatch, value):
    help_text = (_ATTENTION_LAYOUT_HELP + "  --enable-cvp-query-pipeline=<bool>\n"
                 + "  --enable-cvp-query-state-copy-forward=<bool>\n"
                 + "  --enable-cvp-query-final-output-forward=<bool>\n"
                 + "  --enable-cvp-query-output-copy-forward=<bool>\n"
                 + "  --enable-cvp-query-scalar-input-forward=<bool>\n")
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
        _scalar_input_forward_recipe(value), compiler_help=help_text)
    forwarded = [arg for arg in command if arg.startswith("--enable-cvp-query-scalar-input-forward=")]
    assert forwarded == ([] if value is None else ["--enable-cvp-query-scalar-input-forward=" + str(value)])


@pytest.mark.parametrize("value", [0, 1, "true", [], {}])
def test_scalar_input_forward_rejects_non_booleans(compiler_module, value):
    with pytest.raises(TypeError, match="boolean"):
        _parse_options(compiler_module, "Ascend950PR", _scalar_input_forward_recipe(value))


def test_scalar_input_forward_rejects_unroll(compiler_module):
    with pytest.raises(ValueError, match="Unroll"):
        _parse_options(compiler_module, "Ascend950PR", {
            **_scalar_input_forward_recipe(), "enable_cvp_query_output_copy_forward": False,
            "cvp_query_pipeline_mode": "unroll"})


def test_scalar_input_forward_missing_native_capability_rejects(compiler_module, monkeypatch):
    help_text = (_ATTENTION_LAYOUT_HELP + "  --enable-cvp-query-pipeline=<bool>\n"
                 + "  --enable-cvp-query-state-copy-forward=<bool>\n"
                 + "  --enable-cvp-query-final-output-forward=<bool>\n"
                 + "  --enable-cvp-query-output-copy-forward=<bool>\n")
    with pytest.raises(RuntimeError, match="does not support"):
        _capture_multibuffer_command(compiler_module, monkeypatch, True,
            _scalar_input_forward_recipe(), compiler_help=help_text)


def test_scalar_input_forward_cache_and_metadata(compiler_module):
    hashes = set()
    for value in (None, False, True):
        parsed = _parse_options(compiler_module, "Ascend950PR", _scalar_input_forward_recipe(value))
        values = json.loads(json.dumps(parsed.__dict__))
        restored = _parse_options(compiler_module, "Ascend950PR", {
            name: tuple(item) if isinstance(item, list) else item for name, item in values.items()})
        assert restored.enable_cvp_query_scalar_input_forward is value
        assert restored.hash() == parsed.hash()
        hashes.add(parsed.hash())
    assert len(hashes) == 3
