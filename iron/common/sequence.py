# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
import numpy as np
import ml_dtypes
import torch
from . import compilation as comp
from .base import AIEOperatorBase, MLIROperator
import aie.utils as aie_utils
from aie.utils.hostruntime.tensor_class import CPUOnlyTensor
from aie.utils.npukernel import NPUKernel

logger = logging.getLogger(__name__)


# ##########################################################################
# Dispatch policies
# ##########################################################################


class SequenceDispatch:
    """Policy object that decides how an :class:`OperatorSequence` is compiled
    and how its runtime callable is built.

    One concrete policy corresponds to one dispatch mode. The three hooks are:

    * ``resolve(device)`` -- the only device-aware step; expands ``"auto"`` to
      a concrete policy and validates device requirements. Called once, at
      ``set_up_artifacts()`` time (the device is not known at construction).
    * ``set_up_artifacts(seq)`` -- registers the compile artifacts for this
      mode on the owning sequence.
    * ``make_callable(seq)`` -- returns the runtime callable for this mode.
    """

    name = None

    def resolve(self, device):
        """Return the concrete policy for ``device`` (default: unchanged)."""
        return self

    def set_up_artifacts(self, seq):
        """Register the compile artifacts needed by this mode on ``seq``."""
        raise NotImplementedError

    def make_callable(self, seq):
        """Return the runtime callable for this mode."""
        raise NotImplementedError


class AutoDispatch(SequenceDispatch):
    """Selects the platform default.

    Under the HRX (amdxdna) runtime the only supported hardware dispatch is the
    chained-xclbin path (one xclbin+insts per operator, batched into a single
    HRX ``run_chain``), so ``auto`` always resolves to :class:`SeparateDispatch`
    regardless of device generation.
    """

    name = "auto"

    def resolve(self, device):
        return SeparateDispatch()


class FusedDispatch(SequenceDispatch):
    """Single-ELF dispatch (NPU2 only), XRT-only -- unsupported under HRX.

    The fused single-ELF path relied on XRT's ``hw_context`` / ``run.set_arg``
    mechanism to bind three consolidated input/output/scratch buffers to one
    full ELF. HRX exposes no equivalent, so this mode is not available on the
    HRX runtime. Use ``dispatch='separate'`` (the default via ``auto``), which
    batches per-operator dispatches into a single HRX ``run_chain``.
    """

    name = "fused"

    def resolve(self, device):
        raise NotImplementedError(
            "dispatch='fused' (single full-ELF) is XRT-only and is not supported "
            "on the HRX runtime. Use dispatch='separate' (or the default "
            "dispatch='auto'), which batches per-operator dispatches into one "
            "HRX run_chain."
        )

    def set_up_artifacts(self, seq):
        mlir_artifact = self.build_fused_mlir(seq)
        kernel_objects = self._collect_kernel_artifacts(seq)
        full_elf_artifact = comp.FullElfArtifact(
            f"{seq.name}.elf",
            mlir_input=mlir_artifact,
            dependencies=[mlir_artifact] + kernel_objects,
        )
        seq.add_artifacts([full_elf_artifact])

    def build_fused_mlir(self, seq):
        """Build the fused MLIR source that inlines every operator into a single
        module.

        ``seq``'s buffer-layout attributes (``subbuffer_layout``,
        ``buffer_sizes``, ``slice_info``) must already be set.
        """
        operator_mlir_map = {}
        comp_runlist = []
        op_names = {}  # id(op) -> op_name

        for idx, op in enumerate(seq.unique_operators()):
            mlir_artifact = op.get_mlir_artifact()
            if len(op.get_kernel_artifacts()) > 0:
                mlir_artifact.generator.kwargs["func_prefix"] = f"op{idx}_"
            op_name = f"op{idx}_{op.__class__.__name__}"
            op_names[id(op)] = op_name
            operator_mlir_map[op_name] = mlir_artifact

        for op, *bufs in seq.runlist:
            comp_runlist.append((op_names[id(op)], *bufs))

        return comp.SequenceMLIRArtifact(
            seq.name + "_fused.mlir",
            operator_mlir_map=operator_mlir_map,
            runlist=comp_runlist,
            subbuffer_layout=seq.subbuffer_layout,
            buffer_sizes=seq.buffer_sizes,
            slice_info=seq.slice_info,
        )

    def _collect_kernel_artifacts(self, seq):
        """Kernel artifacts from all child operators, prefixed per operator index."""
        kernel_artifacts = []
        for idx, op in enumerate(seq.unique_operators()):
            objs = op.get_kernel_artifacts()
            for obj in objs:
                obj.filename = f"op{idx}_{obj.filename}"
                obj.prefix_symbols = f"op{idx}_"
            kernel_artifacts.extend(objs)
        return kernel_artifacts

    def make_callable(self, seq):
        return SequenceFullELFCallable(seq)


class SeparateDispatch(SequenceDispatch):
    """Chained-dispatch: one independent xclbin+insts per unique operator.

    Under HRX each operator is packaged as its own amdxdna executable (exactly
    like a single-operator dispatch); the runtime callable loads them all and
    submits the runlist as one batched HRX ``run_chain`` (an ``ERT_CMD_CHAIN``),
    so a producer -> consumer buffer handoff between consecutive operators is
    honored within a single submit. Unlike the old XRT path, the per-operator
    xclbins are *not* linked via ``--xclbin-input``; batching happens at dispatch
    time in the HRX runtime, not at packaging time. Owns the compiled
    per-operator xclbin/insts maps consumed by the runtime callable.
    """

    name = "separate"

    def __init__(self):
        self.op_xclbin_map = {}  # id(op) -> xclbin artifact
        self.op_insts_map = {}  # id(op) -> insts artifact
        self.op_kernel_name_map = {}  # id(op) -> kernel_name

    def set_up_artifacts(self, seq):
        artifacts = []
        for idx, op in enumerate(seq.unique_operators()):
            # Unique per-operator prefix so two operators in one sequence never
            # collide on artifact filenames. Each op is a self-contained
            # xclbin+insts (its own amdxdna executable at runtime).
            op_label = f"seq_op{idx}"
            xclbin, insts = op.get_artifacts(prefix=f"{op_label}_")
            artifacts.extend([xclbin, insts])
            self.op_xclbin_map[id(op)] = xclbin
            self.op_insts_map[id(op)] = insts
            # get_artifacts leaves the default kernel/export name ("MLIR_AIE"),
            # which is what HRX looks up per executable.
            self.op_kernel_name_map[id(op)] = xclbin.kernel_name
        seq.add_artifacts(artifacts)

    def make_callable(self, seq):
        return SequenceXclbinCallable(seq, self)


class CompareDispatch(SeparateDispatch):
    """Same compile path as ``separate``, but the callable additionally re-runs
    each operator's CPU ``reference()`` on the NPU-produced inputs and flags
    per-step deviation.

    Args:
        rel_tol / abs_tol: Per-step tolerances; a step counts as a mismatch
            only when it exceeds both.
        raise_on_mismatch: When True (default), raise ``RuntimeError`` on the
            first mismatching step instead of only logging it.
    """

    name = "compare"

    def __init__(self, rel_tol=0.05, abs_tol=1e-2, raise_on_mismatch=True):
        super().__init__()
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.raise_on_mismatch = raise_on_mismatch

    def make_callable(self, seq):
        return SequenceCompareCallable(seq, self)


class ReferenceDispatch(SequenceDispatch):
    """Pure-CPU evaluation via each operator's ``reference()``; compiles nothing."""

    name = "reference"

    def set_up_artifacts(self, seq):
        pass

    def make_callable(self, seq):
        return SequenceReferenceCallable(seq)


_DISPATCH_ALIASES = {
    "auto": AutoDispatch,
    "fused": FusedDispatch,
    "separate": SeparateDispatch,
    "compare": CompareDispatch,
    "reference": ReferenceDispatch,
}


# ##########################################################################
# Compileable: operator sequence
# ##########################################################################


class OperatorSequence(AIEOperatorBase):
    """Operator that concatenates a runlist of operators into a
    single dispatch.

    Args:
        dispatch: Dispatch strategy, given either as a mode name or as a
            :class:`SequenceDispatch` instance. Recognised names:
            ``"auto"`` (default) resolves to ``"separate"`` on the HRX runtime.
            ``"separate"`` compiles each sub-operator to its own xclbin+insts
            and submits the runlist as a single batched HRX ``run_chain``.
            ``"fused"`` (single full-ELF) is XRT-only and raises on HRX.
            ``"reference"`` runs only the per-operator CPU reference
            implementations (no NPU compilation/dispatch).  ``"compare"``
            runs the ``"separate"`` path step-by-step and, after each NPU step,
            also runs the operator's CPU reference on the NPU-produced
            inputs and logs the deviation for testing/debugging.  Pass a
            :class:`CompareDispatch` instance to tune the compare tolerances.
    """

    def __init__(
        self,
        name,
        runlist,
        input_args,
        output_args,
        buffer_sizes=None,
        dispatch="auto",
        *args,
        **kwargs,
    ):
        dispatch = self._coerce_dispatch(dispatch)
        if not all(
            isinstance(op, MLIROperator) and all(isinstance(buf, str) for buf in bufs)
            for op, *bufs in runlist
        ):
            raise TypeError(
                "runlist entries must be (MLIROperator, *str) tuples; "
                "each operator must be an MLIROperator and each buffer name must be a str"
            )
        super().__init__(*args, **kwargs)
        self.runlist = runlist
        self.name = name
        self.input_args = input_args
        self.output_args = output_args
        self.explicit_buffer_sizes = (
            buffer_sizes or {}
        )  # Optional dict: buffer_name -> size_in_bytes
        self._dispatch = dispatch

    @staticmethod
    def _coerce_dispatch(dispatch):
        """Normalise the ``dispatch`` argument to a :class:`SequenceDispatch`."""
        if isinstance(dispatch, SequenceDispatch):
            return dispatch
        elif isinstance(dispatch, str) and dispatch in _DISPATCH_ALIASES:
            return _DISPATCH_ALIASES[dispatch]()
        raise TypeError("selected dispatch mode not supported")

    def unique_operators(self):
        """Operators in runlist order, de-duplicated by identity."""
        seen = {}
        for op, *_ in self.runlist:
            seen.setdefault(id(op), op)
        return list(seen.values())

    def calculate_buffer_layout(self):
        args = {}  # base_buffer_name -> args_spec
        sliced_buffers = (
            {}
        )  # full_buffer_name (with slice) -> (base_name, start, end, args_spec)

        for op, *bufs in self.runlist:
            args_specs = op.get_arg_spec()
            if len(args_specs) != len(bufs):
                raise ValueError(
                    f"Number of buffers ({len(bufs)}) must match operator argument "
                    f"specification ({len(args_specs)}) for operator {op!r}"
                )
            for i, buf_name in enumerate(bufs):
                args_spec = args_specs[i]

                # Parse slice notation: "buffer_name[start:end]"
                if "[" in buf_name and buf_name.endswith("]"):
                    base_name = buf_name[: buf_name.index("[")]
                    slice_part = buf_name[buf_name.index("[") + 1 : -1]
                    start, end = map(int, slice_part.split(":"))
                    sliced_buffers[buf_name] = (base_name, start, end, args_spec)
                    # Track that base buffer exists (size will be set later)
                    if (
                        base_name not in args
                        and base_name not in self.explicit_buffer_sizes
                    ):
                        raise ValueError(
                            f"Sliced buffer '{buf_name}' requires explicit size for base buffer '{base_name}' in buffer_sizes parameter"
                        )
                else:
                    if buf_name not in args:
                        args[buf_name] = args_spec
                    else:
                        if np.prod(args[buf_name].shape) != np.prod(args_spec.shape):
                            raise ValueError(
                                f"Buffer '{buf_name}' has conflicting sizes between operators: "
                                f"{args[buf_name].shape} vs {args_spec.shape}"
                            )

        # Verify all input/output args are present (either as regular or sliced buffers)
        all_buffer_names = set(args.keys()) | set(sliced_buffers.keys())
        for arg in self.input_args:
            if arg not in all_buffer_names and arg not in self.explicit_buffer_sizes:
                raise ValueError(f"Input argument {arg} not found in runlist buffers")
        for arg in self.output_args:
            if arg not in all_buffer_names and arg not in self.explicit_buffer_sizes:
                raise ValueError(f"Output argument {arg} not found in runlist buffers")

        subbuffer_layout = {}
        slice_info = {}  # full_buffer_name -> (base_name, start, end)

        def add_buffers(buffer_type, args_list):
            offset = 0
            for arg in args_list:
                if arg in self.explicit_buffer_sizes:
                    # Explicit size specified - this is a parent buffer for slices
                    length = self.explicit_buffer_sizes[arg]
                    subbuffer_layout[arg] = (buffer_type, offset, length)
                    offset += length
                elif arg in args:
                    arg_spec = args[arg]
                    length = int(
                        np.prod(arg_spec.shape) * np.dtype(arg_spec.dtype).itemsize
                    )
                    subbuffer_layout[arg] = (buffer_type, offset, length)
                    offset += length
                # Note: sliced buffers are handled separately, not in args_list
            return offset  # == total length

        # Add sliced buffer entries to layout (they reference parent buffers)
        for buf_name, (base_name, start, end, args_spec) in sliced_buffers.items():
            slice_info[buf_name] = (base_name, start, end)

        input_buffer_size = add_buffers("input", self.input_args)
        output_buffer_size = add_buffers("output", self.output_args)
        scratch_args = [
            arg
            for arg in args
            if arg not in self.input_args and arg not in self.output_args
        ]
        # Also include explicit buffers that are only used for slicing
        for explicit_buf in self.explicit_buffer_sizes:
            if (
                explicit_buf not in self.input_args
                and explicit_buf not in self.output_args
                and explicit_buf not in scratch_args
            ):
                scratch_args.append(explicit_buf)
        scratch_buffer_size = add_buffers("scratch", scratch_args)

        buffer_sizes = (input_buffer_size, output_buffer_size, scratch_buffer_size)
        return subbuffer_layout, buffer_sizes, slice_info

    def set_up_artifacts(self):
        """Resolve the dispatch policy and build its compile artifacts."""
        self.subbuffer_layout, self.buffer_sizes, self.slice_info = (
            self.calculate_buffer_layout()
        )
        self._dispatch = self._dispatch.resolve(aie_utils.get_current_device())
        self._dispatch.set_up_artifacts(self)

    def get_arg_spec(self):
        raise NotImplementedError(
            "OperatorSequence does not expose a unified arg spec; "
            "use get_layout_for_buffer() to inspect individual buffer layouts"
        )

    def get_callable(self):
        """Return the runtime callable for the resolved dispatch policy."""
        return self._dispatch.make_callable(self)

    def get_layout_for_buffer(self, buffer_name):
        """Return the (buffer_type, offset, length) layout for a named buffer.

        Sliced buffers are resolved recursively to their parent's absolute
        offset.

        Args:
            buffer_name: Name of the buffer, optionally with slice notation.

        Returns:
            Tuple of (buf_type, offset_bytes, length_bytes).
        """
        if buffer_name in self.slice_info:
            buf_name, start, end = self.slice_info[buffer_name]
            buf_type, parent_start, parent_end = self.get_layout_for_buffer(buf_name)
            return buf_type, parent_start + start, parent_start + end

        buf_type, offset, length = self.subbuffer_layout[buffer_name]
        return buf_type, offset, length


# ##########################################################################
# Module helpers
# ##########################################################################


BF16 = np.dtype(ml_dtypes.bfloat16)


def _n_elements(nbytes):
    return max(nbytes, BF16.itemsize) // BF16.itemsize


# ##########################################################################
# Runtime callables
# ##########################################################################


class SequenceCallable:
    """Base for the runtime callables of an ``OperatorSequence``.

    Subclasses provide a buffer model (``_allocate_buffers`` / ``get_buffer``)
    and a step-execution primitive (``_run``). Shared here: step/arg zipping,
    input and output syncing, and timing. Calling the object runs the whole
    sequence once.
    """

    def __init__(self, op):
        self.op = op
        self.last_elapsed = 0.0
        self._buffer_cache = {}
        self._allocate_buffers()

    def _allocate_buffers(self):
        raise NotImplementedError

    def get_buffer(self, buffer_name):
        raise NotImplementedError

    def _iter_steps(self):
        """Yield ``(op, in_names, in_specs, out_name, out_spec)`` per runlist step."""
        for step_op, *buf_names in self.op.runlist:
            specs = step_op.get_arg_spec()
            if len(specs) != len(buf_names):
                raise ValueError(
                    f"Operator {step_op!r} arg-spec count {len(specs)} does not "
                    f"match runlist buffer count {len(buf_names)}"
                )
            *in_names, out_name = buf_names
            *in_specs, out_spec = specs
            yield step_op, in_names, in_specs, out_name, out_spec

    def _sync_inputs(self):
        pass

    def _sync_outputs(self):
        pass

    def _run(self):
        raise NotImplementedError

    def __call__(self):
        self._sync_inputs()
        t0 = time.perf_counter()
        self._run()
        self.last_elapsed = time.perf_counter() - t0
        self._sync_outputs()


class SequenceFullELFCallable(SequenceCallable):
    """Single-ELF dispatch (NPU2), XRT-only -- unsupported under HRX.

    This callable bound three consolidated input/output/scratch buffers to one
    full ELF via XRT's ``hw_context`` / ``run.set_arg``. HRX exposes no
    equivalent binding-by-offset-into-a-shared-arena mechanism, so it is not
    available on the HRX runtime. It is only reachable via
    :class:`FusedDispatch`, whose ``resolve`` already raises; this constructor
    fails loudly as a defensive backstop.
    """

    def __init__(self, op, *args, **kwargs):
        raise NotImplementedError(
            "The fused single full-ELF dispatch is XRT-only and is not supported "
            "on the HRX runtime. Use dispatch='separate' (or the default "
            "dispatch='auto')."
        )


class _PerBufferCallable(SequenceCallable):
    """Callable whose buffers are allocated one per name, with slice views into
    their parent. Inputs sync to the device before the run, all non-input
    buffers back to the host afterwards.
    """

    def _make_buffer(self, n_elements):
        raise NotImplementedError

    def _make_subbuffer(self, parent, offset_bytes, size_bytes):
        raise NotImplementedError

    def _allocate_buffers(self):
        self._buffers = {}
        for name, (_, _, length) in self.op.subbuffer_layout.items():
            self._buffers[name] = self._make_buffer(_n_elements(length))

    def _resolve_buffer(self, buf_name):
        if buf_name in self._buffers:
            return self._buffers[buf_name]
        if buf_name in self.op.slice_info:
            base_name, start_bytes, end_bytes = self.op.slice_info[buf_name]
            sub = self._make_subbuffer(
                self._buffers[base_name], start_bytes, end_bytes - start_bytes
            )
            self._buffers[buf_name] = sub
            return sub
        raise ValueError(f"Unknown buffer '{buf_name}' in fused runlist")

    def get_buffer(self, buffer_name):
        if buffer_name not in self._buffer_cache:
            self._buffer_cache[buffer_name] = self._resolve_buffer(buffer_name)
        return self._buffer_cache[buffer_name]

    def _sync_inputs(self):
        for name in self.op.input_args:
            self._buffers[name].to("npu")

    def _sync_outputs(self):
        for name in self.op.subbuffer_layout:
            if name not in self.op.input_args:
                self._buffers[name].to("cpu")


class SequenceXclbinCallable(_PerBufferCallable):
    """Executes each runlist step as its own xclbin dispatch. Buffers shared by
    name give zero-copy handoff between consecutive operators.

    The compiled per-operator xclbin/insts maps live on the ``SeparateDispatch``
    policy passed in as ``dispatch``.
    """

    def __init__(self, op, dispatch):
        self._dispatch = dispatch
        super().__init__(op)

    def _make_buffer(self, n_elements):
        return aie_utils.tensor((n_elements,), dtype=ml_dtypes.bfloat16)

    def _make_subbuffer(self, parent, offset_bytes, size_bytes):
        raise NotImplementedError(
            "Sliced sequence buffers (buffer_name[start:end]) are not yet "
            "supported on the HRX runtime: HRX dispatch bindings have no byte "
            "offset and HRXTensor exposes no sub-view. Restructure the sequence "
            "to use whole named buffers, or extend the HRX runtime to bind "
            "hrx_buffer_ref_t offsets before using slice notation."
        )

    def _allocate_buffers(self):
        super()._allocate_buffers()
        dispatch = self._dispatch
        runtime = aie_utils.DefaultNPURuntime
        # Load each unique operator as its own amdxdna executable. The runlist
        # is then submitted as a single batched HRX run_chain.
        self._op_handle_map = {}  # id(op) -> KernelHandle
        for op_id, xclbin in dispatch.op_xclbin_map.items():
            npu_kernel = NPUKernel(
                xclbin_path=xclbin.filename,
                kernel_name=dispatch.op_kernel_name_map[op_id],
                insts_path=dispatch.op_insts_map[op_id].filename,
            )
            self._op_handle_map[op_id] = runtime.load(npu_kernel)
        self._execution_plan = [
            (
                self._op_handle_map[id(step_op)],
                [self._resolve_buffer(name) for name in buf_names],
            )
            for step_op, *buf_names in self.op.runlist
        ]

    def _run(self):
        # Submit the whole runlist as one batched dispatch. HRX inserts an
        # execution + memory barrier between chained dispatches, so a later
        # operator observes an earlier one's device writes (producer -> consumer
        # handoff through shared named buffers is correct).
        aie_utils.DefaultNPURuntime.run_chain(list(self._execution_plan))

    def _run_step(self, step_idx, handle, args, step):
        # Per-step single dispatch, used by the compare path (which must read
        # each operator's output back before the next runs). The batched _run
        # above is preferred for normal execution.
        aie_utils.DefaultNPURuntime.run(handle, args)


def _reshape_for_spec(flat_tensor, spec):
    """Slice a flat host buffer to ``spec``'s element count and reshape (a view)."""
    n = int(np.prod(spec.shape)) if spec.shape else 1
    return flat_tensor[:n].reshape(spec.shape)


class SequenceReferenceCallable(_PerBufferCallable):
    """Pure-CPU evaluation via each operator's ``reference()``; no NPU dispatch.
    Device syncs are no-ops on the CPU buffers.
    """

    def _make_buffer(self, n_elements):
        return CPUOnlyTensor((n_elements,), dtype=BF16)

    def _make_subbuffer(self, parent, offset_bytes, size_bytes):
        start = offset_bytes // BF16.itemsize
        end = (offset_bytes + size_bytes) // BF16.itemsize
        # Alias the parent's memory (numpy slice is zero-copy) so a write to
        # this slice is visible when a later step reads the parent by name.
        view = CPUOnlyTensor((end - start,), dtype=BF16)
        view._data = parent.data[start:end]
        view._shape = view._data.shape
        return view

    def _run(self):
        for step_op, in_names, in_specs, out_name, out_spec in self._iter_steps():
            inputs = [
                _reshape_for_spec(self._resolve_buffer(n).torch_view(), s).clone()
                for n, s in zip(in_names, in_specs)
            ]
            out = step_op.reference(*inputs)
            out_flat = self._resolve_buffer(out_name).torch_view()
            n_out = int(np.prod(out_spec.shape)) if out_spec.shape else 1
            out_flat[:n_out].copy_(out.reshape(-1).to(torch.bfloat16))


class SequenceCompareCallable(SequenceXclbinCallable):
    """Runs the xclbin pipeline and, after each step, re-runs the operator's
    reference on the same NPU-produced inputs, logging per-step deviation. The
    NPU output propagates on both sides, so each comparison isolates a single
    operator (no error accumulation).
    """

    def __init__(self, op, dispatch):
        super().__init__(op, dispatch)
        self.rel_tol = dispatch.rel_tol
        self.abs_tol = dispatch.abs_tol
        self.raise_on_mismatch = dispatch.raise_on_mismatch
        self.last_step_stats = []

    def _read_to_cpu(self, name, spec):
        buf = self._resolve_buffer(name)
        buf.to("cpu")
        n = int(np.prod(spec.shape)) if spec.shape else 1
        return buf.torch_view()[:n].clone().reshape(spec.shape)

    def _run(self):
        # Compare mode cannot batch: it must read each operator's NPU output
        # back and re-run its CPU reference before the next operator runs. So
        # dispatch step-by-step (one run() per step) instead of a single
        # run_chain, walking the execution plan alongside the resolved steps.
        self.last_step_stats = []
        for step_idx, ((handle, args), step) in enumerate(
            zip(self._execution_plan, self._iter_steps())
        ):
            self._run_step(step_idx, handle, args, step)

    def _run_step(self, step_idx, handle, args, step):
        step_op, in_names, in_specs, out_name, out_spec = step

        cpu_inputs = [
            self._read_to_cpu(name, spec) for name, spec in zip(in_names, in_specs)
        ]

        aie_utils.DefaultNPURuntime.run(handle, args)

        npu_out = self._read_to_cpu(out_name, out_spec).to(torch.float32)
        ref_out = step_op.reference(*cpu_inputs)

        stats = {
            "step": step_idx,
            "op": type(step_op).__name__,
            "op_name": getattr(step_op, "name", type(step_op).__name__),
            "inputs": list(in_names),
            "output": out_name,
        }

        ref_flat = ref_out.reshape(out_spec.shape).to(torch.float32)
        diff = (npu_out - ref_flat).abs()
        ref_mag = ref_flat.abs()
        max_abs = float(diff.max())
        ref_max = float(ref_mag.max())
        rel = float((diff / (ref_mag + 1e-6)).max())
        mean_abs = float(diff.mean())
        stats.update(
            skipped=False,
            max_abs=max_abs,
            mean_abs=mean_abs,
            max_rel=rel,
            ref_max=ref_max,
        )
        fail = (max_abs > self.abs_tol) and (rel > self.rel_tol)
        stats["mismatch"] = fail
        level = logging.ERROR if fail else logging.INFO
        logger.log(
            level,
            "[compare step %d] %s -> %s: max_abs=%.4g mean_abs=%.4g max_rel=%.4g ref_max=%.4g%s",
            step_idx,
            stats["op"],
            out_name,
            max_abs,
            mean_abs,
            rel,
            ref_max,
            "  MISMATCH" if fail else "",
        )
        if fail and self.raise_on_mismatch:
            raise RuntimeError(
                f"[compare step {step_idx}] {stats['op']} (name={stats['op_name']}) "
                f"-> {out_name}: NPU output deviates from reference "
                f"(max_abs={max_abs:.4g}, max_rel={rel:.4g}, "
                f"ref_max={ref_max:.4g}; inputs={list(in_names)}; "
                f"tolerances abs_tol={self.abs_tol}, rel_tol={self.rel_tol})"
            )
        self.last_step_stats.append(stats)
