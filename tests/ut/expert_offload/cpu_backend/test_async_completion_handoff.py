"""Submission/ready ordering without loading a model or allocating NPU memory."""
import gc
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import types
import weakref

import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    base = root / "vllm_ascend/expert_offload/cpu_backend"
    torch = types.ModuleType("torch")
    torch.inference_mode = lambda: lambda fn: fn
    monkeypatch.setitem(sys.modules, "torch", torch)
    diagnostic = types.ModuleType("vllm_ascend.expert_offload.cpu_prefetch_diagnostics")
    diagnostic.Phase = types.SimpleNamespace(KT_DRAIN_BEGIN=62, KT_DRAIN_END=63,
        KT_READY_RECORD_BEGIN=64, KT_READY_RECORD_END=65)
    monkeypatch.setitem(sys.modules, diagnostic.__name__, diagnostic)
    package = types.ModuleType("_completion_handoff_test")
    package.__path__ = [str(base)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    result = {}
    for name in ("async_completion", "poll_submission"):
        full = package.__name__ + "." + name
        spec = importlib.util.spec_from_file_location(full, base / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, module)
        spec.loader.exec_module(module)
        result[name] = module
    return types.SimpleNamespace(**result)


class Hidden:
    pass


class Backend:
    def __init__(self, events, task_id=17, layer=4, launch_error=None, drain_error=None,
                 launch_entered=None, launch_release=None, drain_entered=None, drain_release=None,
                 cpu_expert_count=1):
        self.events, self.task_id = events, task_id
        self.config = types.SimpleNamespace(layer_idx=layer)
        self._wrapper = self
        self.submitted = False
        self.cpu_expert_count = cpu_expert_count
        self.native_pending = 0
        self.launch_error, self.drain_error = launch_error, drain_error
        self.launch_entered, self.launch_release = launch_entered, launch_release
        self.drain_entered, self.drain_release = drain_entered, drain_release
        self._async_callback_refs = []

    def _launch_pinned_forward(self, args):
        self.events.append((self.task_id, "launch", threading.get_ident()))
        assert isinstance(args[0], Hidden) and args[1] == 123
        if self.launch_entered: self.launch_entered.set()
        if self.launch_release: assert self.launch_release.wait(2)
        if self.launch_error: raise self.launch_error
        self.submitted = True
        self.native_pending = int(self.cpu_expert_count > 0)
        return self.task_id

    def drain_pinned_forward(self):
        assert self.submitted, "sync must never run before a successful launch"
        self.events.append((self.task_id, "drain", threading.get_ident()))
        if self.drain_entered: self.drain_entered.set()
        if self.drain_release: assert self.drain_release.wait(2)
        if self.drain_error: raise self.drain_error
        self.native_pending = 0


class Ready:
    def __init__(self, events, task_id=17):
        self.events, self.task_id = events, task_id
        self.done = threading.Event()

    def record(self, stream):
        self.events.append((self.task_id, "ready", threading.get_ident()))
        self.done.set()


def worker(module):
    result = module.CompletionWorker.__new__(module.CompletionWorker)
    class Notifier:
        def __init__(self): self.permits = threading.Semaphore(0)
        def notify(self): self.permits.release()
        def wait(self, spin_us): self.permits.acquire()
    result.jobs = module.SubmissionQueue(Notifier(), 0)
    result.stream = object()
    result._close_lock = threading.Lock()
    result._close_requested = result._closed = False
    return result


def test_request_precedes_launch_and_never_readies_before_submit_and_drain(modules):
    module = modules.async_completion
    w = worker(module);events = []
    entered, release, drain_entered, drain_release = [threading.Event() for _ in range(4)]
    backend = Backend(events, launch_entered=entered, launch_release=release,
        drain_entered=drain_entered, drain_release=drain_release)
    ready = Ready(events)
    w.submit(backend, Hidden(), 123, ready)
    assert not events  # Input producer did no CPU launch and no sync.
    w.jobs.put(None)
    errors = []

    def run():
        try: w._run_jobs(None, False)
        except BaseException as exc: errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(1) and not ready.done.is_set()
    assert [e[1] for e in events] == ["launch"]
    release.set()
    assert drain_entered.wait(1) and not ready.done.is_set()
    drain_release.set();thread.join(2)
    assert not errors and not thread.is_alive()
    assert [e[1] for e in events] == ["launch", "drain", "ready"]
    assert all(e[2] == thread.ident for e in events)


@pytest.mark.parametrize("cpu_expert_count", [0, 1])
def test_empty_or_fast_native_task_still_launches_then_drains(modules, cpu_expert_count):
    w = worker(modules.async_completion);events = []
    backend = Backend(events, task_id=0, cpu_expert_count=cpu_expert_count)
    w.submit(backend, Hidden(), 123, Ready(events, 0))
    w.jobs.put(None);w._run_jobs(None, False)
    assert [e[1] for e in events] == ["launch", "drain", "ready"]
    assert backend.native_pending == 0


@pytest.mark.parametrize("failure", ["launch", "drain"])
def test_failure_propagates_without_publishing_readiness(modules, failure):
    module = modules.async_completion;w = worker(module);events = []
    error = RuntimeError(failure + " failed")
    kwargs = {failure + "_error": error}
    ready = Ready(events)
    request = module.SubmissionRequest(Backend(events, **kwargs), Hidden(), 123, ready)
    with pytest.raises(RuntimeError, match=failure + " failed"):
        w._process_submission(request, None, False)
    assert not ready.done.is_set() and all(e[1] != "ready" for e in events)


def test_multiple_requests_are_fifo_and_input_lifetimes_are_retained(modules):
    w = worker(modules.async_completion);events = [];refs = []
    for task_id in range(3):
        hidden = Hidden();refs.append(weakref.ref(hidden))
        w.submit(Backend(events, task_id), hidden, 123, Ready(events, task_id))
    del hidden;gc.collect()
    assert all(ref() is not None for ref in refs)
    w.jobs.put(None);w._run_jobs(None, False);gc.collect()
    assert [(e[0],e[1]) for e in events] == [(i,phase) for i in range(3) for phase in ("launch","drain","ready")]
    assert all(ref() is None for ref in refs)


def test_blocked_get_is_notified_and_shutdown_wakes_it(modules):
    w = worker(modules.async_completion);events = []
    thread = threading.Thread(target=w._run_jobs, args=(None, False))
    thread.start()
    ready = Ready(events)
    w.submit(Backend(events), Hidden(), 123, ready)
    assert ready.done.wait(1)
    w.thread = thread
    w.close()
    assert not thread.is_alive()
    w.close()
    assert w.jobs._items.empty()  # Repeated close did not add a second sentinel.


def test_payload_is_published_before_each_permit_including_shutdown(modules):
    class Notifier:
        def __init__(self): self.pending = 0
        def notify(self):
            assert not q._items.empty()
            self.pending += 1
        def wait(self, spin_us):
            assert spin_us == 3000 and self.pending > 0
            self.pending -= 1
    notifier = Notifier()
    q = modules.async_completion.SubmissionQueue(notifier, 3000)
    payload = object()
    q.put(payload);q.put(None)
    assert notifier.pending == 2
    assert q.get() is payload and q.get() is None and notifier.pending == 0


def test_poll_path_only_queues_and_telemetry_runs_after_ready(modules):
    w = worker(modules.async_completion);events = []
    poll = modules.poll_submission.PollWorker.__new__(modules.poll_submission.PollWorker)
    backend, hidden, ready = Backend(events), Hidden(), Ready(events)
    poll.channels = [(backend, hidden, 123, ready, object())]
    poll.path, poll.rows, poll.completion = "enabled", [], w
    poll._queue_ready_channel(0, 3, 10, 20)
    assert not events and not poll.rows
    request = w.jobs.get()
    original = request.on_submitted

    def on_submitted(task_id, stamp):
        assert ready.done.is_set()
        original(task_id, stamp)

    request = modules.async_completion.SubmissionRequest(backend, hidden, 123, ready, on_submitted)
    w._process_submission(request, None, False)
    assert poll.rows[0]["task_id"] == 17 and poll.rows[0]["seq"] == 3
    assert poll.rows[0]["submit_end_ns"] >= poll.rows[0]["queued_ns"]
    assert poll.rows[0]["ready_observed_ns"] == 10


def test_acl_callback_only_queues_and_trace_context_is_the_launched_task(modules, monkeypatch):
    module = modules.async_completion;w = worker(module);events = [];marks = []
    backend, hidden, ready = Backend(events), Hidden(), Ready(events)
    monkeypatch.setattr(module, "worker_for", lambda device: w)
    module._ACL = object()
    ext = types.SimpleNamespace(launch_no_block_callback=lambda *a: None,
        lifecycle_set_context=lambda *a: marks.append(("context",a)),
        lifecycle_mark=lambda p: marks.append((p,None)))
    monkeypatch.setitem(sys.modules, "kt_kernel", types.SimpleNamespace(kt_kernel_ext=ext))
    hidden.device = object()
    module.record_submission(backend, hidden, types.SimpleNamespace(npu_stream=123), ready)
    backend._async_callback_refs[0](None)
    assert not events and not ready.done.is_set()
    w._process_submission(w.jobs.get(), ext, True)
    assert marks[0] == ("context", (17,4))
    assert [m[0] for m in marks[1:]] == ["callback_exit","completion_wait_enter","completion_wait_end","output_ready_recorded"]
    assert [e[1] for e in events] == ["launch","drain","ready"]


def test_close_with_pending_request_flushes_final_poll_telemetry(modules, tmp_path):
    module = modules.async_completion;w = worker(module);events = []
    entered, release = threading.Event(), threading.Event()
    backend, hidden, ready = Backend(events, launch_entered=entered, launch_release=release), Hidden(), Ready(events)
    poll = modules.poll_submission.PollWorker.__new__(modules.poll_submission.PollWorker)
    poll.channels = [(backend, hidden, 123, ready, object())]
    poll.path, poll.rows, poll.completion = str(tmp_path/'poll.jsonl'), [], w
    poll.alive = poll._writer_alive = True
    poll._closed = False;poll._close_lock = threading.Lock()
    poll.thread = threading.Thread(target=lambda: None)
    poll.thread.start();poll.thread.join()
    poll.writer = threading.Thread(target=poll.write_rows)
    poll.writer.start()
    w.thread = threading.Thread(target=w._run_jobs, args=(None,False))
    w.thread.start();poll._queue_ready_channel(0, 3, 10, 20)
    assert entered.wait(1)
    errors = []
    def close():
        try: poll.close()
        except BaseException as exc: errors.append(exc)
    closer = threading.Thread(target=close)
    closer.start()
    deadline = time.monotonic()+1
    while not w._close_requested and time.monotonic()<deadline: time.sleep(.001)
    assert w._close_requested
    # The previous alive flag would let the writer exit at this .5s boundary
    # while the completion callback still had not appended its tail row.
    time.sleep(.55)
    assert closer.is_alive() and poll.writer.is_alive() and not ready.done.is_set()
    release.set();closer.join(2)
    assert not closer.is_alive() and not errors
    assert not poll.writer.is_alive() and not w.thread.is_alive()
    rows = [json.loads(s) for s in Path(poll.path).read_text().splitlines()]
    assert len(rows)==1 and rows[0]['task_id']==17 and rows[0]['seq']==3
    poll.close();w.close()
    assert w.jobs._items.empty()


def test_old_native_library_fallback_constructs_and_completes(modules, monkeypatch, caplog):
    import queue
    module = modules.async_completion
    old_ext = types.SimpleNamespace()  # Older runtime has no CompletionNotifier.
    monkeypatch.setitem(sys.modules,'kt_kernel',types.SimpleNamespace(kt_kernel_ext=old_ext))
    monkeypatch.setitem(sys.modules,'torch_npu',types.SimpleNamespace(npu=types.SimpleNamespace(SyncLaunchStream=lambda **kw:object())))
    monkeypatch.setattr(module.atexit,'register',lambda *a:None)
    monkeypatch.setattr(module.CompletionWorker,'run',lambda self:self._run_jobs(old_ext,False))
    w = module.CompletionWorker(0);events = []
    try:
        assert isinstance(w.jobs, queue.SimpleQueue)
        for task_id in (0,17):
            ready = Ready(events, task_id)
            w.submit(Backend(events,task_id=task_id),Hidden(),123,ready)
            assert ready.done.wait(1)
        assert [(e[0],e[1]) for e in events]==[(i,p) for i in (0,17) for p in ('launch','drain','ready')]
        assert sum('lacks CompletionNotifier' in r.message for r in caplog.records)==1
    finally:
        w.close();w.close()
    assert not w.thread.is_alive() and w.jobs.empty()
