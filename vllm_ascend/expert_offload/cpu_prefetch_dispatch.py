"""Candidate CPU planner/DMA dispatch from captured D2H sequence flags.

No ACL host callback is used by this path. Prediction and copies remain as
before; a private native poller wakes one CPU worker. A graph-captured reset
precedes each ready publication, so an earlier completion cannot satisfy a
later replay. The original planner synchronizes its DMA stream before release.
"""
from __future__ import annotations

import atexit
import ctypes
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import platform
import threading
import time
from typing import Callable

import torch
import torch_npu


@dataclass(eq=False)
class PrefetchChannel:
    callback: Callable[[], None]
    layer: int
    static_id: int
    device_flag: torch.Tensor
    host_flag: torch.Tensor
    done: object
    submission_stream: object
    completed_generation: int = 0


class CpuPrefetchDispatcher:
    """Instance-owned flag bank/native scanner/channel metadata/worker.

    Initialize before graph capture. Main capture thread is the sole channel
    registration producer; worker only consumes already-published channels.
    The scanner is intentionally hot for this correctness candidate and uses
    one extra CPU worker without changing affinity, priority or CPU pool size.
    """

    def __init__(self, *, device: int, capacity: int, library_path: str,
                 on_ready: Callable[[int, int, int, int], None] | None = None,
                 fatal_handler: Callable[[int], None] = os._exit,
                 observation_capacity: int = 65536):
        if platform.machine().lower() not in ("x86_64", "amd64"):
            raise ValueError("CPU ready polling requires coherent x86_64 host memory")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Initialize the CPU prefetch dispatcher before capture")
        self.device = int(device)
        self.capacity = int(capacity)
        self._on_ready = on_ready
        self._fatal_handler = fatal_handler
        self._logger = logging.getLogger(__name__)
        self._error = None
        self._closed = False
        self._closing = False
        self._lifecycle = threading.RLock()
        self._channels = []
        self.observations = []
        self.observation_capacity = int(observation_capacity)
        self.observations_dropped = 0
        self._library = ctypes.CDLL(library_path, use_errno=True)
        self._create = self._library.cpu_prefetch_poll_create
        self._create.argtypes = [ctypes.c_uint32]
        self._create.restype = ctypes.c_void_p
        self._register = self._library.cpu_prefetch_poll_register
        self._register.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._register.restype = ctypes.c_int
        self._next = self._library.cpu_prefetch_poll_next
        self._next.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]
        self._next.restype = ctypes.c_int
        self._stop = self._library.cpu_prefetch_poll_stop
        self._stop.argtypes = [ctypes.c_void_p]
        self._stop.restype = None
        self._destroy = self._library.cpu_prefetch_poll_destroy
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = ctypes.c_int
        self._native_error = self._library.cpu_prefetch_poll_error
        self._native_error.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                      ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
        self._native_error.restype = None
        self._handle = self._create(capacity)
        if not self._handle:
            raise OSError(ctypes.get_errno(), "Cannot create private CPU prefetch poll context")
        try:
            # Each padded row occupies its own64B cache line. Every capture/site
            # gets a distinct row; no zeros/reset allocation occurs in capture.
            self._device_counters = torch.zeros((capacity, 16), dtype=torch.int32,
                                                device=torch.device("npu", self.device))
            self._host_counters = torch.zeros((capacity, 16), dtype=torch.int32,
                                              device="cpu", pin_memory=True)
            self._device_flags = [self._device_counters[i, :1] for i in range(capacity)]
            self._host_flags = [self._host_counters[i, :1] for i in range(capacity)]
            self._completion_stream = torch_npu.npu.SyncLaunchStream(device=self.device)
            # DMA commands must bypass the global taskqueue too. A plain
            # Python worker lacks ACL host-callback's native sync-launch guard.
            self.dma_stream = torch_npu.npu.SyncLaunchStream(device=self.device)
            # Require the existing value-based ExternalEvent implementation
            # to materialize its handle at construction. Do NOT pre-record an
            # artificial completion: native event state may retain that record
            # and turn a later wait into an already-satisfied dependency.
            self._events = [torch_npu.npu.ExternalEvent() for _ in range(capacity)]
            for event in self._events:
                if not event.npu_event:
                    raise RuntimeError("CPU prefetch requires pre-materialized value-based ExternalEvent support")
            # Initialization barrier only, before any graph/callback can exist.
            torch.npu.current_stream(self.device).synchronize()
            self._thread = threading.Thread(target=self._run, name="cpu_prefetch_dma", daemon=True)
            self._thread.start()
        except BaseException:
            self._stop(self._handle)
            self._destroy(self._handle)
            self._handle = None
            raise
        atexit.register(self.close)

    @property
    def channels(self):
        return tuple(self._channels)

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("CPU prefetch worker failed") from self._error
        if self._closed:
            raise RuntimeError("CPU prefetch dispatcher is closed")

    def record_submission(self, *, stream, callback: Callable[[], None], layer: int,
                          static_id: int = 0) -> PrefetchChannel:
        # Registration publishes to C while CDLL releases the GIL. Hold a
        # lifecycle lock through recording all corresponding graph nodes;
        # close cannot destroy the handle between metadata and publication.
        # Runtime worker never needs this lock.
        with self._lifecycle:
            if self._closing:
                raise RuntimeError("CPU prefetch dispatcher is closing")
            return self._record_submission_locked(stream=stream, callback=callback,
                                                   layer=layer, static_id=static_id)

    def _record_submission_locked(self, *, stream, callback: Callable[[], None], layer: int,
                                 static_id: int = 0) -> PrefetchChannel:
        """Record reset→ready generation D2H→external wait on the given stream.

        Caller must already have staged prediction/map D2H and waited its ready
        event. Caller then records original mapping H2D and normal done/join.
        This call executes once during each graph construction, not on replay.
        """
        self.raise_if_failed()
        if not torch.npu.is_current_stream_capturing():
            raise RuntimeError("record_submission is capture-only; eager keeps the original CPU path")
        index = len(self._channels)
        if index >= self.capacity:
            raise RuntimeError("CPU prefetch capture/site capacity exhausted")
        done = self._events[index]
        channel = PrefetchChannel(callback, int(layer), int(static_id), self._device_flags[index],
                                  self._host_flags[index], done, stream)
        # A published C count is the scanner's ownership boundary. Append
        # Python metadata first so a returned native index is always valid.
        self._channels.append(channel)
        native_index = self._register(self._handle, channel.host_flag.data_ptr())
        if native_index != index:
            self._error = RuntimeError(f"CPU prefetch native registration failed: {native_index} != {index}")
            self._stop(self._handle)
            raise self._error
        # Reset is a node in THIS stream and runs on EVERY replay. It must
        # precede the D2H ready publication that permits the worker to record.
        with torch.npu.stream(stream):
            done.reset(stream)
            channel.device_flag.add_(1)
            channel.host_flag.copy_(channel.device_flag, non_blocking=True)
            stream.wait_event(done)
        return channel

    @torch.inference_mode()
    def _run(self):
        try:
            torch.npu.set_device(self.device)
            stamp = ctypes.c_uint64()
            generation = ctypes.c_uint32()
            while True:
                index = self._next(self._handle, ctypes.byref(stamp), ctypes.byref(generation))
                if index == -1:
                    return
                if index < 0:
                    failed = ctypes.c_int32(); expected = ctypes.c_uint32(); observed = ctypes.c_uint32()
                    self._native_error(self._handle, ctypes.byref(failed), ctypes.byref(expected), ctypes.byref(observed))
                    raise RuntimeError(f"CPU prefetch sequence failure {index}: channel={failed.value} "
                                       f"expected={expected.value} observed={observed.value}")
                channel = self._channels[index]
                dispatch = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                if self._on_ready is not None:
                    self._on_ready(index, generation.value, stamp.value, dispatch)
                # A prefetch callback may enqueue DMA and return an event for
                # the completion stream to wait on.  This keeps the worker
                # out of a host-side stream.synchronize() while preserving
                # the done-event ordering seen by the captured prefetch
                # stream.  Legacy callbacks still return None.
                completion_event = channel.callback()
                planner_done = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                if completion_event is not None:
                    self._completion_stream.wait_event(completion_event)
                channel.done.record(self._completion_stream)
                recorded = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                channel.completed_generation = generation.value
                if len(self.observations) < self.observation_capacity:
                    self.observations.append(dict(channel=index,layer=channel.layer,static_id=channel.static_id,
                        generation=generation.value,ready_observed_raw_ns=stamp.value,dispatch_raw_ns=dispatch,
                        planner_done_raw_ns=planner_done,done_recorded_raw_ns=recorded,tid=threading.get_native_id()))
                else:
                    self.observations_dropped += 1
        except BaseException as exc:
            self._error = exc
            if self._handle:
                self._stop(self._handle)
            try:
                self._logger.exception("CPU prefetch worker failed; refusing to release stale weights")
            except BaseException:
                pass
            self._fatal_handler(1)

    def close(self):
        with self._lifecycle:
            if self._closed:
                return
            if self._closing:
                raise RuntimeError("CPU prefetch close already in progress")
            self._closing = True
        try:
            self._close_retired()
        except BaseException:
            with self._lifecycle:
                self._closing = False
            raise
        with self._lifecycle:
            self._closed = True
            self._closing = False

    def _close_retired(self):
        # Retire every published ready/wait while the worker is STILL alive.
        # Stopping first could leave a graph waiting forever for a dropped job.
        # Owner must already have stopped new replay submissions.
        if self._error is not None and self._channels:
            raise RuntimeError("Failed worker has live graph channels; retain events/context") from self._error
        if self._error is None:
            retired = set()
            for channel in self._channels:
                key = int(channel.submission_stream.npu_stream)
                if key not in retired:
                    channel.submission_stream.synchronize()
                    retired.add(key)
        if self._handle:
            self._stop(self._handle)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            # Never free flags/event owners while C or a DMA planner uses them.
            raise RuntimeError("CPU prefetch worker is still active; native context retained")
        self.dma_stream.synchronize()
        self._completion_stream.synchronize()
        if self._handle:
            result = self._destroy(self._handle)
            if result:
                raise RuntimeError("CPU prefetch native context still has an active poll")
            self._handle = None


def create_manager_dispatcher(manager, *, library_path: str):
    """Narrow supported case; other configurations keep the ACL host path."""
    config = manager.offload_config
    cpu = config.cpu_moe
    if (not config.expert_prefetch_enabled or config.enable_multi_card or manager.ep_size != 1
            or getattr(config, "expert_predictor", "fate") != "fate"
            or not cpu.enabled or cpu.backend != "kt_kernel" or cpu.weight_format != "gguf"
            or cpu.transfer_one_expert or manager.native_mxfp4_enabled
            or platform.machine().lower() not in ("x86_64", "amd64")
            or not Path(library_path).is_file()):
        return None
    from vllm_ascend.expert_offload.h2d_transfer import TorchCopyH2DTransport
    if not isinstance(manager._get_h2d_transport(), TorchCopyH2DTransport):
        return None
    device = manager.moe_layers[0].log2phy.device.index
    worker = CpuPrefetchDispatcher(device=device, capacity=64 * len(manager.moe_layers),
                                  library_path=library_path)
    def mark_native_ready(index, generation, ready_raw_ns, dispatch_raw_ns):
        # This POD timestamp is after acquiring the GIL; arg0 retains the
        # earlier native flag-observation timestamp. No per-job I/O/thread.
        try:
            trace = getattr(manager, "_prefetch_trace", None)
            if trace is not None:
                trace.mark_operation(70, worker._channels[index].layer, ready_raw_ns)
        except BaseException:
            pass
    worker._on_ready = mark_native_ready
    return worker
