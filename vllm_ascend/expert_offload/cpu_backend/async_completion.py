"""CPU completion signals readiness for direct mapped-host result consumption."""
from __future__ import annotations
from vllm_ascend.expert_offload.cpu_prefetch_diagnostics import Phase as CpuPrefetchPhase
import atexit
import ctypes
from collections import deque
from dataclasses import dataclass
import logging
import os
import queue
import threading
from time import monotonic_ns
from typing import Callable
import torch

_LOG = logging.getLogger(__name__)
_WORKERS = {}
_LOCK = threading.Lock()
_ACL = None
_DEFAULT_COMPLETION_SPIN_US = 3000


class SubmissionQueue:
    """Python-owned payloads with a GIL-free native warm notification wait."""

    def __init__(self, notifier, spin_us):
        self._items = queue.SimpleQueue()
        self._notifier = notifier
        self._spin_us = spin_us

    def put(self, request):
        # Publish the Python owner before the native permit, including None.
        self._items.put(request)
        self._notifier.notify()

    def get(self):
        self._notifier.wait(self._spin_us)
        return self._items.get()


@dataclass(frozen=True)
class SubmissionRequest:
    """Own input lifetimes until submit, drain and readiness publication finish."""

    backend: object
    hidden: object
    stream_handle: int
    ready: object
    on_submitted: Callable[[int, int], None] | None = None
    trace_callback_exit: bool = False


@dataclass(frozen=True)
class DrainRequest:
    """A submitted CPU task waiting for the independent drain stage."""

    request: SubmissionRequest
    task_id: int
    trace: bool
    submitted_ns: int


class CompletionWorker:
    def __init__(self, device):
        import torch_npu
        from kt_kernel import kt_kernel_ext
        self.device = device
        self.stream = torch_npu.npu.SyncLaunchStream(device=device)
        # Retain the short hot window in native code, without Python polling,
        # fixed100us sleeps, or GIL ownership during the wait.
        notifier_type = getattr(kt_kernel_ext, "CompletionNotifier", None)
        if notifier_type is None:
            _LOG.warning("kt_kernel_ext lacks CompletionNotifier; using a compatible blocking queue. "
                         "Native warm notification and GIL improvements require the updated runtime.")
            self.jobs = queue.SimpleQueue()
        else:
            spin_us = max(0, int(float(os.environ.get("KT_MOE_COMPLETION_SPIN_US",
                                                   str(_DEFAULT_COMPLETION_SPIN_US)))))
            self.jobs = SubmissionQueue(notifier_type(), spin_us)
        self._close_lock = threading.Lock()
        self._close_requested = False
        self._closed = False
        # Graph mode has one pinned input/output buffer per backend.  Keep
        # requests for the same backend serialized, while allowing different
        # layers to submit work while an earlier layer is draining.
        self._state_cv = threading.Condition()
        self._busy_backends = set()
        self._pending_by_backend = {}
        self._drain_jobs = queue.SimpleQueue()
        self._drain_thread = threading.Thread(
            target=self._run_drain, name="kt_cpu_drain", daemon=True)
        self._drain_thread.start()
        self.thread = threading.Thread(target=self.run, name="kt_cpu_h2d", daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def close(self):
        """Owner first quiesces all poll/ACL producers; drain the FIFO once."""
        with self._close_lock:
            if self._closed:
                return
            if not self._close_requested:
                self.jobs.put(None)
                self._close_requested = True
            self.thread.join(timeout=5)
            drain_thread = getattr(self, "_drain_thread", None)
            if drain_thread is not None:
                # The live submit loop normally publishes this sentinel after
                # all pending backends have drained.  The fallback also makes
                # tests/older callers that replace ``run`` shut down cleanly.
                self._drain_jobs.put(None)
                drain_thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError("CPU completion still has pending work during close")
            if drain_thread is not None and drain_thread.is_alive():
                raise RuntimeError("CPU drain worker is still active during close")
            self._closed = True

    def submit(self, backend, hidden, stream_handle, ready, on_submitted=None,
               trace_callback_exit=False):
        """Publish the request before running native submit on the consumer.

        The consumer always launches successfully before it calls sync. Merely
        publishing a request therefore cannot observe pending==0 and release
        stale output, even when the CPU expert list is empty.
        """
        self.jobs.put(SubmissionRequest(backend, hidden, int(stream_handle), ready,
                                        on_submitted, trace_callback_exit))

    def _process_submission(self, request, ext, trace):
        backend, hidden, ready = request.backend, request.hidden, request.ready
        # Both native halves run on this worker. Input-poll/ACL threads need
        # only notify the queue; their later scheduling cannot delay job put.
        task_id = backend._launch_pinned_forward((hidden, request.stream_handle))
        submitted_ns = monotonic_ns() if request.on_submitted is not None else 0
        if trace:
            ext.lifecycle_set_context(task_id, backend.config.layer_idx)
            if request.trace_callback_exit:
                # Historical lifecycle phase: end of the CPU submit half,
                # now executed by this worker rather than the ACL producer.
                ext.lifecycle_mark("callback_exit")
            ext.lifecycle_mark("completion_wait_enter")
        cpu_diag = getattr(backend, "_cpu_prefetch_trace", None)
        if cpu_diag is not None:
            cpu_diag.mark_operation(CpuPrefetchPhase.KT_DRAIN_BEGIN, backend.config.layer_idx, task_id)
        backend._wrapper.drain_pinned_forward()
        if cpu_diag is not None:
            cpu_diag.mark_operation(CpuPrefetchPhase.KT_DRAIN_END, backend.config.layer_idx, task_id)
        if trace:
            ext.lifecycle_mark("completion_wait_end")
        # CPU writes are complete. The NPU combine kernel reads the mapped
        # host result directly after this readiness event. Exceptions before
        # here propagate to run's fail-stop handler without recording ready.
        if cpu_diag is not None:
            cpu_diag.mark_operation(CpuPrefetchPhase.KT_READY_RECORD_BEGIN, backend.config.layer_idx, task_id)
        ready.record(self.stream)
        if cpu_diag is not None:
            cpu_diag.mark_operation(CpuPrefetchPhase.KT_READY_RECORD_END, backend.config.layer_idx, task_id)
        if trace:
            ext.lifecycle_mark("output_ready_recorded")
        # Optional poll telemetry keeps the real submit timestamp, but its
        # Python bookkeeping is outside the readiness critical path.
        if request.on_submitted is not None:
            request.on_submitted(task_id, submitted_ns)

    def _submit_only(self, request, ext, trace):
        """Launch CPU work and hand its drain to the independent worker."""
        backend, hidden = request.backend, request.hidden
        task_id = backend._launch_pinned_forward((hidden, request.stream_handle))
        submitted_ns = monotonic_ns() if request.on_submitted is not None else 0
        if trace:
            ext.lifecycle_set_context(task_id, backend.config.layer_idx)
            if request.trace_callback_exit:
                ext.lifecycle_mark("callback_exit")
        self._drain_jobs.put(DrainRequest(request, task_id, trace, submitted_ns))

    def _release_backend(self, request):
        key = id(request.backend)
        next_request = None
        with self._state_cv:
            pending = self._pending_by_backend.get(key)
            if pending:
                next_request = pending.popleft()
                if not pending:
                    del self._pending_by_backend[key]
            else:
                self._busy_backends.discard(key)
            self._state_cv.notify_all()
        if next_request is not None:
            # Re-enter the submit stage only after the previous drain has
            # released this backend's pinned buffers.
            self.jobs.put(next_request)

    def _drain_one(self, item, ext):
        request = item.request
        backend, ready = request.backend, request.ready
        trace = item.trace
        task_id = item.task_id
        if trace:
            ext.lifecycle_set_context(task_id, backend.config.layer_idx)
            ext.lifecycle_mark("completion_wait_enter")
        cpu_diag = getattr(backend, "_cpu_prefetch_trace", None)
        try:
            if cpu_diag is not None:
                cpu_diag.mark_operation(CpuPrefetchPhase.KT_DRAIN_BEGIN,
                                        backend.config.layer_idx, task_id)
            backend._wrapper.drain_pinned_forward()
            if cpu_diag is not None:
                cpu_diag.mark_operation(CpuPrefetchPhase.KT_DRAIN_END,
                                        backend.config.layer_idx, task_id)
            if trace:
                ext.lifecycle_mark("completion_wait_end")
            if cpu_diag is not None:
                cpu_diag.mark_operation(CpuPrefetchPhase.KT_READY_RECORD_BEGIN,
                                        backend.config.layer_idx, task_id)
            ready.record(self.stream)
            if cpu_diag is not None:
                cpu_diag.mark_operation(CpuPrefetchPhase.KT_READY_RECORD_END,
                                        backend.config.layer_idx, task_id)
            if trace:
                ext.lifecycle_mark("output_ready_recorded")
            # Preserve the existing telemetry contract: the row is published
            # after the output-ready event, but the next submit no longer waits
            # for this drain because it runs on the separate worker.
            if request.on_submitted is not None:
                request.on_submitted(task_id, item.submitted_ns)
        finally:
            self._release_backend(request)

    def _run_live_submit_jobs(self, ext, trace):
        """Submit stage: never drain a CPU task on this thread."""
        stop_seen = False
        while True:
            request = self.jobs.get()
            if request is None:
                stop_seen = True
                while True:
                    with self._state_cv:
                        if not self._busy_backends and not self._pending_by_backend:
                            break
                        pending_exists = bool(self._pending_by_backend)
                        if not pending_exists:
                            self._state_cv.wait(timeout=0.001)
                            continue
                    # A drain completion requeues pending work through jobs.
                    request = self.jobs.get()
                    if request is not None:
                        self._accept_live_request(request, ext, trace)
                self._drain_jobs.put(None)
                return
            self._accept_live_request(request, ext, trace)

    def _accept_live_request(self, request, ext, trace):
        key = id(request.backend)
        with self._state_cv:
            if key in self._busy_backends:
                self._pending_by_backend.setdefault(key, deque()).append(request)
                return
            self._busy_backends.add(key)
        try:
            self._submit_only(request, ext, trace)
        except BaseException:
            with self._state_cv:
                self._busy_backends.discard(key)
                self._state_cv.notify_all()
            raise

    def _run_live_drain_jobs(self, ext, trace):
        while True:
            item = self._drain_jobs.get()
            if item is None:
                return
            self._drain_one(item, ext)

    def _run_jobs(self, ext, trace):
        while True:
            request = self.jobs.get()
            if request is None:
                return
            self._process_submission(request, ext, trace)
            # Do not keep the last input/output alive during the next idle get.
            del request

    @torch.inference_mode()
    def run(self):
        torch.npu.set_device(self.device)
        cpu = os.environ.get("KT_MOE_COMPLETION_CPU")
        priority = int(os.environ.get("KT_MOE_COMPLETION_RT_PRIORITY", "0"))
        if cpu is not None:
            try:
                os.sched_setaffinity(0, {int(cpu)})
            except OSError as exc:
                _LOG.warning("completion affinity to CPU %s failed: %s", cpu, exc)
        if priority:
            # Containers without RT bandwidth deny SCHED_FIFO; a dead
            # completion thread hangs the engine, so never let this raise.
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
            except OSError as exc:
                _LOG.warning("SCHED_FIFO priority %d denied: %s; staying SCHED_OTHER", priority, exc)
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), -10)
        except OSError:
            pass
        from kt_kernel import kt_kernel_ext as ext
        trace = bool(os.environ.get("KT_MOE_LIFECYCLE_TRACE"))
        try:
            self._run_live_submit_jobs(ext, trace)
        except BaseException:
            _LOG.exception("CPU MoE submission/completion failed; terminating this engine instead of releasing stale output")
            os._exit(1)

    @torch.inference_mode()
    def _run_drain(self):
        torch.npu.set_device(self.device)
        from kt_kernel import kt_kernel_ext as ext
        trace = bool(os.environ.get("KT_MOE_LIFECYCLE_TRACE"))
        try:
            self._run_live_drain_jobs(ext, trace)
        except BaseException:
            _LOG.exception("CPU MoE drain failed; terminating this engine instead of releasing stale output")
            os._exit(1)

def worker_for(device):
    index = device.index if device.index is not None else torch.npu.current_device()
    with _LOCK:
        if index not in _WORKERS:
            _WORKERS[index] = CompletionWorker(index)
        return _WORKERS[index]

def record_submission(backend, hidden, stream, ready):
    global _ACL
    worker = worker_for(hidden.device)
    if _ACL is None:
        _ACL = ctypes.CDLL("libascendcl.so")
        _ACL.aclrtLaunchCallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        _ACL.aclrtLaunchCallback.restype = ctypes.c_int
    args = (hidden, int(stream.npu_stream))
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def submit(_unused):
        try:
            worker.submit(backend, hidden, args[1], ready, trace_callback_exit=True)
        except BaseException:
            _LOG.exception("CPU MoE submission failed")
            os._exit(1)
    backend._async_callback_refs.append(submit)
    from kt_kernel import kt_kernel_ext
    kt_kernel_ext.launch_no_block_callback(
        ctypes.cast(submit, ctypes.c_void_p).value, 0, args[1])
