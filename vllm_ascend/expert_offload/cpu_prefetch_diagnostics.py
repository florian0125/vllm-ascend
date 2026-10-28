"""Bounded CPU/DMA prefetch markers; no forward context access or GIL yields.

Native pre-GIL boundaries and scheduling are supplied by external uprobes.
The mmap header starts disabled; controller changes uint32 at offset24 to1.
Native timestamps use CLOCK_MONOTONIC_RAW, matching ftrace's mono_raw clock.
"""
from __future__ import annotations

import ctypes
from enum import IntEnum, IntFlag
import os
from pathlib import Path
import threading


class Phase(IntEnum):
    PYTHON_CALLBACK_ENTER = 10
    INFERENCE_ENTER = 11
    PLANNER_ENTER = 12
    LOAD_STREAM_ENTER_BEGIN = 13
    LOAD_STREAM_ENTER_END = 14
    IDS_TO_LIST_BEGIN = 15
    IDS_TO_LIST_END = 16
    MAP_TO_LIST_BEGIN = 17
    MAP_TO_LIST_END = 18
    SCORE_BEGIN = 19
    SCORE_END = 20
    VICTIM_BEGIN = 21
    VICTIM_END = 22
    PLAN_END = 23
    DESCRIPTOR_BEGIN = 24
    DESCRIPTOR_END = 25
    COPY_BATCH_BEGIN = 26
    COPY_BATCH_END = 27
    SCALE_REFRESH_BEGIN = 28
    SCALE_REFRESH_END = 29
    BOOKKEEPING_BEGIN = 30
    BOOKKEEPING_END = 31
    H2D_SYNC_BEGIN = 32
    H2D_SYNC_END = 33
    LOAD_STREAM_EXIT = 34
    PLANNER_RETURN = 35
    INFERENCE_EXIT = 36
    PYTHON_CALLBACK_RETURN = 37
    CALLBACK_ERROR = 38
    COPY_FIELD_BEGIN = 40
    COPY_FIELD_END = 41
    KT_SUBMIT_BEGIN = 60
    KT_SUBMIT_END = 61
    KT_DRAIN_BEGIN = 62
    KT_DRAIN_END = 63
    KT_READY_RECORD_BEGIN = 64
    KT_READY_RECORD_END = 65
    PREFETCH_NATIVE_READY = 70
    STATIC_DISPATCH_BEGIN = 100
    STATIC_STAGING_BEGIN = 101
    STATIC_STAGING_END = 102
    STATIC_CALLBACK_ENQUEUE = 103
    STATIC_MAP_PUBLISH_ENQUEUE = 104
    STATIC_DONE_RECORD_ENQUEUE = 105


class Flags(IntFlag):
    EAGER_STATIC = 1
    CAPTURE_STATIC = 2
    PREFETCH_CALLBACK = 4
    BACKEND_OPERATION = 8


class CpuPrefetchTrace:
    """Manager-owned trace instance; all live mutable state is per-instance."""

    def __init__(self, library_path: str, output_root: str, capacity: int = 262144):
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"cpu_prefetch_{os.getpid()}.bin"
        # PyDLL retains the GIL during tiny pure-C markers. CDLL would release
        # and reacquire it at each marker, changing the behavior being measured.
        self._library = ctypes.PyDLL(library_path, use_errno=True)
        self._open = self._library.diag_open
        self._open.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        self._open.restype = ctypes.c_void_p
        self._mark = self._library.diag_mark
        self._mark.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32,
                              ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64,
                              ctypes.c_int64, ctypes.c_uint32]
        self._mark.restype = ctypes.c_int
        self._begin = self._library.diag_begin
        self._begin.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint32]
        self._begin.restype = ctypes.c_uint64
        self._close = self._library.diag_close
        self._close.argtypes = [ctypes.c_void_p]
        self._close.restype = None
        self._handle = self._open(os.fsencode(self.path), capacity)
        if not self._handle:
            raise OSError(ctypes.get_errno(), f"Cannot create CPU prefetch diagnostics: {self.path}")
        self._local = threading.local()
        self._static_sequence = 0

    def next_static_id(self) -> int:
        self._static_sequence += 1
        return self._static_sequence

    def begin_callback(self, layer: int, static_id: int):
        """Already holds GIL. This timestamp is NOT the native callback entry."""
        previous = None
        try:
            previous = getattr(self._local, "context", None)
            sequence = self._begin(self._handle, layer, static_id, int(Flags.PREFETCH_CALLBACK))
            self._local.context = (int(sequence), int(layer), int(static_id)) if sequence else None
        except BaseException:
            try:
                self._local.context = None
            except BaseException:
                pass
        return previous

    def mark(self, phase: int, arg0: int = 0, arg1: int = 0) -> None:
        try:
            context = getattr(self._local, "context", None)
            if context is not None and self._handle:
                sequence, layer, static_id = context
                self._mark(self._handle, int(phase), layer, sequence, static_id,
                           int(arg0), int(arg1), int(Flags.PREFETCH_CALLBACK))
        except BaseException:
            # Diagnostics must never escape the installed ACL callback bridge.
            pass

    def end_callback(self, previous) -> None:
        try:
            self.mark(Phase.PYTHON_CALLBACK_RETURN)
        except BaseException:
            pass
        finally:
            try:
                self._local.context = previous
            except BaseException:
                pass

    def mark_static(self, phase: int, layer: int, static_id: int, capturing: bool,
                    arg0: int = 0, arg1: int = 0) -> None:
        """Forward Python execution only; never pretend this runs on replay."""
        try:
            flags = Flags.CAPTURE_STATIC if capturing else Flags.EAGER_STATIC
            if self._handle:
                self._mark(self._handle, int(phase), layer, 0, static_id, arg0, arg1, int(flags))
        except BaseException:
            pass

    def mark_operation(self, phase: int, layer: int, task_id: int = 0) -> None:
        """KT thread operation, independent of callback TLS/graph schedule IDs.

        sequence/static_id=0; arg0=existing task ID, arg1=per-thread marker
        ordinal. Pair operation begin/end on (tid,layer,phase), not callbackseq.
        """
        try:
            tick = getattr(self._local, "operation_tick", 0) + 1
            self._local.operation_tick = tick
            if self._handle:
                self._mark(self._handle, int(phase), int(layer), 0, 0,
                           int(task_id), tick, int(Flags.BACKEND_OPERATION))
        except BaseException:
            pass

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            self._close(handle)
