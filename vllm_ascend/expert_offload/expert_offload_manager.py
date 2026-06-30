"""Expert Offload Manager — manages CPU-side expert weights and NPU paging."""

import queue
import threading
import time
import os
import datetime
from contextlib import contextmanager, nullcontext  # time_stage is a contextmanager

import atexit
import statistics
from concurrent.futures import ThreadPoolExecutor

import torch
import torch_npu
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

from vllm_ascend.expert_offload.expert_predictor import make_predictor, AIPredictCtx


_SUBSCRIBED_COMPUTE_STREAMS = set()
def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS


class ExpertOffloadManager:
    """Singleton manager for expert weight offloading.

    Stores all expert weights on CPU and pages the needed experts to NPU
    during forward based on routing topk_ids.
    """

    _instance: "ExpertOffloadManager | None" = None

    # Parallel weight-load pool. The strided transpose-copy in load_w13/
    # load_w2 is single-threaded (~0.2 GB/s into pinned memory); fanning the
    # ~99k shard copies out over this many workers hits ~2-4 GB/s.
    _LOAD_POOL_WORKERS = 32
    # Bound on in-flight futures before a partial drain (releases owned clones
    # early so transient memory stays small). >> workers, so no starvation.
    _LOAD_POOL_DRAIN_EVERY = 2048

    @classmethod
    def get_instance(cls) -> "ExpertOffloadManager":
        assert cls._instance is not None, "ExpertOffloadManager not initialized"
        return cls._instance

    def __init__(self, vllm_config: VllmConfig):
        from vllm_ascend.ascend_config import get_ascend_config

        self.offload_config = get_ascend_config().expert_offload_config
        self.num_device_experts = self.offload_config.num_device_experts
        self.topk = vllm_config.model_config.hf_config.num_experts_per_tok
        self.offload_threshold = self.num_device_experts // self.topk

        self._num_hash_layers = getattr(
            vllm_config.model_config.hf_config, "num_hash_layers", 0)

        self._run_meta: dict = {}
        try:
            mc = vllm_config.model_config
            sc = vllm_config.scheduler_config
            pc = vllm_config.parallel_config
            model = str(getattr(mc, "model", "?")).rstrip("/").rsplit("/", 1)[-1]
            self._run_meta = {
                "model": model,
                "dtype": str(getattr(mc, "dtype", "?")),
                "quant": str(getattr(mc, "quantization", None) or "none"),
                "tp": getattr(pc, "tensor_parallel_size", "?"),
                "dp": getattr(pc, "data_parallel_size", "?"),
                "max_num_seqs": getattr(sc, "max_num_seqs", "?"),
                "max_model_len": getattr(mc, "max_model_len", "?"),
                "enforce_eager": getattr(mc, "enforce_eager", "?"),
            }
        except Exception:
            self._run_meta = {}

        # CPU weight buffers (post-transpose format, matching device after
        # process_weights_after_loading):
        #   w13 per expert: [hidden_size, w13_up_dim]
        #   w2 per expert:  [intermediate_size_per_partition, hidden_size]
        self.w13_weights_cpu: list[list[torch.Tensor]] = []
        self.w2_weights_cpu: list[list[torch.Tensor]] = []

        # Registered AscendFusedMoE layers, indexed by moe_instance_id order
        self.moe_layers: list = []

        # CPU buffers for quantized model scale/offset parameters.
        # Keyed by attr_name (e.g. "w13_weight_scale", "w2_weight_offset").
        # Each value is a list of layers, each layer is a list of expert tensors.
        self.scale_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}
        self.offset_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}

        # Temporary per-expert storage for w13 scale/offset shard assembly.
        # Key: (layer_moe_idx, expert_id, attr_name), value: first shard.
        # Scale/offset arrive as w1 + w3 shards; we stash one until the
        # other arrives, then assemble and copy into scale_cpu_buffers.
        self._scale_shard_temp: dict[tuple[int, int, str], torch.Tensor] = {}

        self.num_device_layers = self.offload_config.num_device_layers
        self.num_total_experts = None  # set in init_layer_cpu_buffers
        self.cache_policy: LRCExpertCachePolicy | None = None
        self.cache_requests: list[int] = []
        self.cache_hits: list[int] = []
        self.cache_misses: list[int] = []
        self.cache_calls: list[int] = []
        self.last_hit_experts: list[list[int]] = []
        self.last_miss_experts: list[list[int]] = []
        # Master debug switch for expert-offload diagnostics — UPDATE-W cache
        # trace, per-prefill-load logs, prefetch/update slot shortfalls.
        # Flipping it on surfaces them at info level (no need for global
        # VLLM_LOGGING_LEVEL=DEBUG).
        self._debug = self.offload_config.moe_offload_debug

        # Diagnostic: wall time of the parallel weight-load phase (safetensors
        # → pinned CPU buffers). Logged in _finalize_offload.
        self._weight_load_secs: float = 0.0
        self._weight_load_calls: int = 0

        # Deferred weight-load pool. load_w13/load_w2/_load_scale_shard clone
        # loaded_weight synchronously (while the safetensors mmap is still
        # mapped) and submit the strided transpose-copy to this pool. The
        # deferred copy reads the owned clone, so it stays correct after the
        # safetensors mmap is unmapped (which happens before _finalize_offload).
        # drain_load_pool() is called from _finalize_offload before the buffers
        # are read by process_weights_after_loading().
        self._load_pool: ThreadPoolExecutor | None = None
        self._load_futures: list = []
        self._load_phase_start: float = 0.0
        self._saved_num_threads: int | None = None

        # End-of-test cache hit-rate summary (decode path only)
        self._seq_token_layer_hits: dict[int, float] = {}     # layer_idx -> hit rate, current step
        self._seq_token_stats: list[tuple[float, float, float, float]] = []  # per step of current seq
        self._seq_token_batch: list[int] = []                 # per step of current seq: batch size
        self._seq_layer_rates: dict[int, list[float]] = {}    # layer_idx -> per-step rates, current seq
        self._pending_token_batch: int = 0                    # batch size of the step in flight
        self._seq_stats_num_seqs = self.offload_config.seq_stats_num_seqs
        self._summary_seq_stats: list[tuple[float, float, float, float]] = []  # per measured window
        self._summary_step_sum: list[float] = [0.0, 0.0, 0.0, 0.0]
        self._summary_step_cnt: int = 0
        self._summary_gen_tokens: int = 0
        self._summary_request_cnt: int = 0
        self._summary_layer_rate_sum: dict[int, float] = {}
        self._summary_layer_rate_cnt: dict[int, int] = {}
        self._seq_stats_done: bool = False

        # Generalized per-(layer, step) timing (decode path, profiling only).
        # One dict per metric. Pre-hook metrics (attn, router) are recorded
        # before the offload hook, so they land in a pending slot and are
        # harvested into the step dict at the hit (after any close). Post-hook
        # metrics (upload, compute, shared) stash directly.
        self._profile_timing = self.offload_config.cache_profile_timing
        self._t_timing_metrics = ("attn", "router", "hc", "lrc", "miss_load",
                                  "compute", "shared", "pf_compute", "pf_h2d")
        self._t_rate_metrics = ("pred_acc", "pf_in_lrc", "pf_useful")
        self._t_metrics = self._t_timing_metrics + self._t_rate_metrics
        self._t_prehook = ("attn", "router", "hc", "lrc")
        self._t_label = {
            "attn": "Attention", "router": "Router (gate + select_experts)",
            # CHANGE (#6): renamed; this is the on-demand miss path, which is
            # evict-pick + scale/offset + weight H2D, not a pure transfer.
            "miss_load": "Cache-miss on-demand load (evict+H2D)",
            "compute": "Routed experts (fused_experts)",
            "shared": "Shared expert MLP",
            # NEW (#2): structural hyperconnection ops (hc_pre/hc_post) + the two
            # RMSNorms + residual clones around attention and the MoE.
            "hc": "Hyperconnection + layernorm + residual",
            # NEW (#3): LRC cache-policy observe() bookkeeping (the O(num_experts)
            # EMA update per token).
            "lrc": "LRC policy observe()",
            # NEW
            "pf_compute": "Prefetch predict (CPU)",
            "pf_h2d": "Prefetch upload (H2D)",
            "pred_acc": "Prediction accuracy (|P&G|/|P|)",
            "pf_in_lrc": "Predicted already resident (|hit|/|P|)",
            "pf_useful": "Prefetched-expert usefulness (|miss&G|/|miss|)",
        }
        self._t_pending = {m: {} for m in self._t_prehook}        # layer_idx -> ms (await harvest)
        self._t_step = {m: {} for m in self._t_metrics}           # current step: layer_idx -> ms
        self._t_seq_stats = {m: [] for m in self._t_metrics}      # current seq: list of (avg,med,min,max)
        self._t_seq_total = {m: [] for m in self._t_metrics}      # current seq: list of per-step totals
        self._t_seq_layer = {m: {} for m in self._t_metrics}      # current seq: layer_idx -> [per-step ms]
        self._t_sum_step = {m: [0.0, 0.0, 0.0, 0.0] for m in self._t_metrics}  # over measured steps
        self._t_sum_total = {m: 0.0 for m in self._t_metrics}
        self._t_sum_win = {m: [] for m in self._t_metrics}        # per window: (avg,med,min,max)
        self._t_sum_layer_sum = {m: {} for m in self._t_metrics}
        self._t_sum_layer_cnt = {m: {} for m in self._t_metrics}
        self._t_sum_stepcnt = 0

        self._dump_csv = os.environ.get("CSV", "0") not in ("0", "", "false", "False")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.environ.get(
            "CSV_PATH",
            "./bench_csvs"
        )
        self._csv_path = os.path.join(self._csv_path, "prefetch_summary_%s_%s.csv" % (getattr(self.offload_config, "expert_predictor"), timestamp))
        self._csv_written = False

        atexit.register(self._dump_final_stats_at_exit)

        ExpertOffloadManager._instance = self

        self.load_stream = torch_npu.npu.Stream()

        # Prefill pool: ndl layers × all experts on NPU, shared round-robin
        self._prefill_w13: list[torch.Tensor] = []
        self._prefill_w2: list[torch.Tensor] = []
        self._prefill_w13_scale: list[torch.Tensor] = []        # W8A8
        self._prefill_w13_scale_fp32: list[torch.Tensor] = []   # W8A8
        self._prefill_w13_offset: list[torch.Tensor] = []       # W8A8
        self._prefill_w2_scale: list[torch.Tensor] = []         # W8A8
        self._prefill_w2_offset: list[torch.Tensor] = []        # W8A8
        self._prefill_log2phy: torch.Tensor = None              # identity [0..127]
        self._prefill_initialized: bool = False
        self._skip_prefill: bool = False  # set during profile runs

        # Next-layer expert prefetch infrastructure
        self._prefetch_stream = torch_npu.npu.Stream()
        self._gate_weights_cpu: list[torch.Tensor | None] = []
        # NPU copy of gate weights for graph-capturable on-device prediction.
        # Kept in fp32 to match the CPU prediction path.
        self._gate_weights_npu: list[torch.Tensor | None] = []

        # Threaded prefetch: daemon thread processes prefetch requests
        # so the main forward-pass thread is never blocked by H2D copies.
        self._prefetch_queue: queue.Queue = queue.Queue()
        self._prefetch_thread_ready: threading.Event = threading.Event()
        self._prefetch_state_lock = threading.Lock()
        self._prefetch_layer_done: dict[int, threading.Event] = {}
        self._prefetch_layer_npu_event: dict[int, torch_npu.npu.Event] = {}
        self._prefetch_thread: threading.Thread | None = None
        self._npu_device: torch.device | None = None

        self._prefetch_profile_pending: dict[int, dict] = {}

        # Pinned CPU staging buffers for graph-mode prefetch prediction.
        # Allocated lazily in _finalize_offload (hidden_dim / num_total_experts
        # are only known after MoE layers are registered).  Mirror
        # update_weights' non_blocking D2H + stream-order-ready pattern: the
        # graph host callback reads these (already-ready) buffers instead of
        # doing a blocking .cpu() on a live graph tensor.
        self._prefetch_hs_h: torch.Tensor | None = None
        self._prefetch_log2phy_h: torch.Tensor | None = None
        self._prefetch_log2phy_np = None

        # construct the pluggable next-layer expert predictor selected by
        # config (default "fate" == the original cross-layer-gate method).
        self._predictor = make_predictor(
            self.offload_config.expert_predictor, mgr=self
        )
        # AI driver state. Active for any learned predictor
        self._ai_active = getattr(self._predictor, "uses_model_forward_driver", False)
        self._ai_pa_buf = None             # [max_tokens, H] fp32 — pre_attn[ℓ]
        self._ai_ri_buf = [None, None]     # 2x [max_tokens, H] fp32 — router ping-pong
        self._ai_pending: dict[int, dict] = {}

    # ------------------------------------------------------------------ #
    #  Lifecycle: called during model init and after weight loading       #
    # ------------------------------------------------------------------ #

    def init_layer_cpu_buffers(self, layer, layer_moe_idx: int):
        """Allocate CPU weight + scale/offset buffers for one MoE layer.

        Called from AscendFusedMoE.__init__ after device tensors are set up,
        so CPU buffers exist before the safetensors weight loader runs.
        """
        ntotal = layer.global_num_experts
        if self.num_total_experts is None:
            self.num_total_experts = ntotal
        assert ntotal == self.num_total_experts, \
            f"MoE layers must have same expert count: {ntotal} vs {self.num_total_experts}"

        params_dtype = layer.w13_weight.dtype
        # Use logical dimensions (layer.hidden_size / intermediate_size)
        # rather than device tensor shapes.  Device tensor layout may be
        # transposed before process_weights_after_loading (e.g. W8A8
        # stores [intermediate, hidden] pre-transpose), confusing the
        # per-expert shape derivation.
        w13_shape = (layer.hidden_size, 2 * layer.intermediate_size_per_partition)
        w2_shape = (layer.intermediate_size_per_partition, layer.hidden_size)

        w13_list = [
            torch.empty(w13_shape, dtype=params_dtype, device="cpu", pin_memory=True)
            for _ in range(ntotal)
        ]
        w2_list = [
            torch.empty(w2_shape, dtype=params_dtype, device="cpu", pin_memory=True)
            for _ in range(ntotal)
        ]
        self.w13_weights_cpu.append(w13_list)
        self.w2_weights_cpu.append(w2_list)

        # Per-expert storage size in bytes. The expert shape is uniform across
        # layers (asserted above), so this is set unconditionally on the first
        # layer and reused. Used for raw-storage slicing during NZ paging.
        self.w13_expert_size_bytes = w13_list[0].nelement() * w13_list[0].element_size()
        self.w2_expert_size_bytes = w2_list[0].nelement() * w2_list[0].element_size()

        # Scale / offset CPU buffers (W8A8)
        self._init_layer_scale_buffers(layer, layer_moe_idx, ntotal)

        self.moe_layers.append(layer)

    def _init_layer_scale_buffers(self, layer, layer_moe_idx: int,
                                   ntotal: int):
        """Allocate CPU scale/offset buffers for a single MoE layer."""
        attr_specs = [
            ("scale_cpu_buffers", "w13_weight_scale"),
            ("scale_cpu_buffers", "w2_weight_scale"),
            ("offset_cpu_buffers", "w13_weight_offset"),
            ("offset_cpu_buffers", "w2_weight_offset"),
        ]
        for buffer_dict_name, attr_name in attr_specs:
            if not hasattr(layer, attr_name):
                continue
            dev_tensor = getattr(layer, attr_name)
            # Match the device slot shape the buffer is paged into. The W8A8
            # path flattens each expert's scale/offset to 1D (.view(E, -1)) in
            # its process_weights_after_loading, which runs AFTER this alloc
            # (pre-flatten, shape [.., 1]). Allocate 1D so the buffer already
            # matches the post-flatten device slot — no reshape at copy time.
            per_expert_shape = (dev_tensor[0].numel(),)
            dtype = dev_tensor.dtype
            buffer_dict: dict = getattr(self, buffer_dict_name)
            if attr_name not in buffer_dict:
                buffer_dict[attr_name] = []
            buffers = buffer_dict[attr_name]
            while len(buffers) <= layer_moe_idx:
                buffers.append([])
            for _ in range(ntotal):
                buffers[layer_moe_idx].append(
                    torch.empty(per_expert_shape, dtype=dtype,
                                device="cpu", pin_memory=True))

    def _finalize_offload(self, model):
        """Post-weight-loading finalization.

        Must be called AFTER get_model() has finished loading all weights.
        Performs NZ format conversion, cache policy init, forward buffer
        init, fp32 scale refresh, prefill pool creation, and gate weight
        registration.
        """
        if not self.moe_layers:
            return
        # Barrier: ensure all deferred load_w13/load_w2/_load_scale_shard
        # copies have landed before process_weights_after_loading reads them.
        self.drain_load_pool()
        t0 = time.perf_counter()
        logger.info(
            "[OFFLOAD] weight load (safetensors→CPU buffer): %.1fs over %d calls",
            self._weight_load_secs, self._weight_load_calls)
        t1 = time.perf_counter()
        self.process_weights_after_loading()
        t2 = time.perf_counter()

        num_moe_layers = len(self.moe_layers)
        if self.offload_config.cache_policy_enabled:
            self.cache_requests = [0 for _ in range(num_moe_layers)]
            self.cache_hits = [0 for _ in range(num_moe_layers)]
            self.cache_misses = [0 for _ in range(num_moe_layers)]
            self.cache_calls = [0 for _ in range(num_moe_layers)]
            self.last_hit_experts = [[] for _ in range(num_moe_layers)]
            self.last_miss_experts = [[] for _ in range(num_moe_layers)]
            self.cache_policy = LRCExpertCachePolicy(
                num_layers=num_moe_layers,
                num_experts=self.num_total_experts,
                cache_size=self.num_device_experts,
                topk=self.topk,
                recent_window=self.offload_config.cache_recent_window,
                ema_beta=self.offload_config.cache_ema_beta,
                recent_weight=self.offload_config.cache_recent_weight,
                ema_weight=self.offload_config.cache_ema_weight,
                router_weight=self.offload_config.cache_router_weight,
                age_weight=self.offload_config.cache_age_weight,
            )
        t3 = time.perf_counter()

        ntotal = self.num_total_experts
        self.topk_ids_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.int32, device="cpu", pin_memory=True)
        self.topk_weights_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.float32, device="cpu", pin_memory=True)
        self.log2phy_h = torch.zeros(ntotal, dtype=torch.int32,
                                     device='cpu', pin_memory=True)
        self.log2phy_np = self.log2phy_h.numpy()
        t4 = time.perf_counter()

        self.refresh_fp32_scales()
        t5 = time.perf_counter()
        self.create_prefill_pool()
        t6 = time.perf_counter()
        if self.offload_config.expert_prefetch_enabled:
            self._predictor.register_from_model(model)
            # Pinned staging buffers for graph-mode host-callback prefetch.
            # The callback runs on the compute stream's host thread during
            # graph replay, where blocking .cpu() on a live graph tensor would
            # deadlock (see md_anlysis/2026-0615-1416-...).  These buffers let
            # trigger_next_layer_prefetch stage the data with non_blocking D2H
            # *before* launching the callback, exactly like update_weights.
            hidden_dim = self.moe_layers[0].hidden_size
            self._prefetch_hs_h = torch.zeros(
                [self.offload_threshold, hidden_dim],
                dtype=torch.float32, device='cpu', pin_memory=True)
            self._prefetch_log2phy_h = torch.zeros(
                self.num_total_experts, dtype=torch.int32,
                device='cpu', pin_memory=True)
            self._prefetch_log2phy_np = self._prefetch_log2phy_h.numpy()

            # Allocate AI driver buffers per the active predictor's input_spec.
            # hidden_dim was computed just above (line 382).
            if self._ai_active:
                spec = self._predictor.input_spec
                mt = self.offload_threshold
                if "pre_attn" in spec:
                    self._ai_pa_buf = torch.zeros(
                        [mt, hidden_dim], dtype=torch.float32, device="npu")
                if "router_input_prev" in spec:
                    self._ai_ri_buf = [
                        torch.zeros([mt, hidden_dim], dtype=torch.float32, device="npu"),
                        torch.zeros([mt, hidden_dim], dtype=torch.float32, device="npu"),
                    ]
        t7 = time.perf_counter()
        logger.info(
            "[OFFLOAD] finalize breakdown: process_weights=%.1fs "
            "cache_policy=%.1fs buffers=%.1fs init_device=%.1fs "
            "prefill_pool=%.1fs gate=%.1fs | total=%.1fs",
            t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t7 - t0)

    def process_weights_after_loading(self):
        """Convert resident CPU expert buffers to fractal NZ format (W8A8).

        For W8A8 the device weight lives in NZ format, so we mirror that on
        the CPU side and page experts to the device with a raw copy_ on the
        underlying storage (avoids an implicit format cast on every H2D).

        After this runs each w13/w2 CPU tensor still reports its original
        [hidden, ...] shape, but its storage holds NZ-format bytes — a "liar
        tensor". Touch it only via untyped_storage() slicing, never through
        the tensor view. No-op for non-int8 models.
        """
        first_w13 = self.w13_weights_cpu[0][0]
        if first_w13.dtype != torch.int8:
            return
        num_moe_layers = len(self.w13_weights_cpu)
        num_experts = len(self.w13_weights_cpu[0])
        for layer_id in range(num_moe_layers):
            w13 = torch.stack(self.w13_weights_cpu[layer_id]).to('npu')
            w13_nz = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)
            w13_nz_storage = w13_nz.untyped_storage()
            w2 = torch.stack(self.w2_weights_cpu[layer_id]).to('npu')
            w2_nz = torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ)
            w2_nz_storage = w2_nz.untyped_storage()
            for expert_id in range(num_experts):
                self.w13_weights_cpu[layer_id][expert_id].untyped_storage().copy_(
                    w13_nz_storage[expert_id * self.w13_expert_size_bytes : (expert_id + 1) * self.w13_expert_size_bytes]
                )
                self.w2_weights_cpu[layer_id][expert_id].untyped_storage().copy_(
                    w2_nz_storage[expert_id * self.w2_expert_size_bytes : (expert_id + 1) * self.w2_expert_size_bytes]
                )


    # ------------------------------------------------------------------ #
    #  Deferred weight-load pool                                          #
    # ------------------------------------------------------------------ #
    #
    # Weight loading is callback-driven: the safetensors loader calls
    # load_w13/load_w2/_load_scale_shard once per shard (~99k calls), serially
    # in the main thread. The per-call strided transpose-copy into pinned
    # memory is ~0.2 GB/s single-threaded, which dominated startup (~9 min).
    #
    # Strategy: each loader callback (a) owns the shard via a synchronous
    # .clone() while the safetensors mmap is still mapped, then (b) submits
    # the strided transpose-copy to a worker pool and returns immediately.
    # The main thread keeps pulling shards while the pool churns through
    # copies concurrently. drain_load_pool() barriers before _finalize_offload
    # reads the buffers. Because the deferred copy reads the owned clone (not
    # the mmap view), it stays correct after the safetensors mmap is unmapped
    # (which happens before _finalize_offload runs).

    def _get_load_pool(self) -> ThreadPoolExecutor:
        if self._load_pool is None:
            # Pin torch intra-op threads to 1: otherwise each copy_ spawns
            # nproc libgomp threads and 32 workers x 640 cores exhausts the
            # thread limit (EAGAIN). Parallelism comes from the pool itself.
            self._saved_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            self._load_pool = ThreadPoolExecutor(
                max_workers=self._LOAD_POOL_WORKERS,
                thread_name_prefix="offload-load")
            self._load_phase_start = time.perf_counter()
            logger.info(
                "[OFFLOAD] starting parallel weight load (workers=%d)",
                self._LOAD_POOL_WORKERS)
        return self._load_pool

    def _track_load_future(self, fut) -> None:
        self._load_futures.append(fut)
        if len(self._load_futures) >= self._LOAD_POOL_DRAIN_EVERY:
            self._drain_futures()

    def _drain_futures(self) -> None:
        if not self._load_futures:
            return
        # f.result() re-raises any worker exception (e.g. shape mismatch).
        for f in self._load_futures:
            f.result()
        self._load_futures.clear()

    def drain_load_pool(self) -> None:
        """Wait for all deferred weight copies to finish.

        Safe to call after the safetensors mmap is unmapped: deferred copies
        read owned clones, not mmap views.
        """
        self._drain_futures()
        if self._load_pool is not None:
            self._load_pool.shutdown(wait=True)
            self._load_pool = None
            if self._saved_num_threads is not None:
                torch.set_num_threads(self._saved_num_threads)
                self._saved_num_threads = None
            self._weight_load_secs = time.perf_counter() - self._load_phase_start

    # -- worker copy kernels (static: no self, no shared mutable state) -- #

    @staticmethod
    def _copy_w13_shard(cpu: torch.Tensor, owned: torch.Tensor,
                        shard_id: str, intermed: int) -> None:
        if shard_id == "w1":
            cpu[:, :intermed].copy_(owned.t())
        elif shard_id == "w3":
            cpu[:, intermed: intermed + owned.shape[0]].copy_(owned.t())

    @staticmethod
    def _copy_w2(dst: torch.Tensor, owned: torch.Tensor) -> None:
        dst.copy_(owned.t())

    @staticmethod
    def _copy_scale_assembled(target: torch.Tensor,
                              w1: torch.Tensor, w3: torch.Tensor) -> None:
        assembled = torch.cat([w1, w3], dim=0).reshape(target.shape)
        target.copy_(assembled)

    @staticmethod
    def _copy_scale_direct(target: torch.Tensor, owned: torch.Tensor) -> None:
        target.copy_(owned.reshape(target.shape))

    # ------------------------------------------------------------------ #
    #  Weight-load entry points (called by the safetensors loader)        #
    # ------------------------------------------------------------------ #

    def register_gate_weights(self, model):
        """Store fp32 CPU and NPU copies of gate.weight for each MoE layer.

        Called from _register_offload_layers() after all MoE layers are
        registered.  The CPU gate weights are used by the legacy
        predict_next_layer_experts(); the NPU copies are used by
        predict_next_layer_experts_npu() so prediction can run on-device
        and be captured in a CUDA/NPU graph.
        """
        from vllm_ascend.models.deepseek_v4 import DeepseekV4MoE
        moe_wrappers = [m for m in model.modules()
                        if isinstance(m, DeepseekV4MoE)]
        for wrapper in moe_wrappers:
            gate_param = wrapper.gate.weight.data
            gate_cpu = gate_param.cpu().float().clone()
            self._gate_weights_cpu.append(gate_cpu)
            # Place on the same NPU device as the parameter for on-device
            # graph-capturable prediction.
            self._gate_weights_npu.append(gate_cpu.to(gate_param.device))
        logger.info("[PREFETCH] registered gate weights for %d MoE layers",
                    len(self._gate_weights_cpu))

    def load_w13(self, layer_moe_idx: int, expert_id: int,
                 loaded_weight: torch.Tensor, shard_id: str):
        """Store w1/w3 shard to CPU buffer (transposed) via the load pool."""
        self._weight_load_calls += 1
        cpu = self.w13_weights_cpu[layer_moe_idx][expert_id]
        intermed = cpu.shape[1] // 2
        # Own the bytes now (the mmap may unmap before the worker runs).
        owned = loaded_weight.cpu().clone()
        fut = self._get_load_pool().submit(
            self._copy_w13_shard, cpu, owned, shard_id, intermed)
        self._track_load_future(fut)

    def load_w2(self, layer_moe_idx: int, expert_id: int,
                loaded_weight: torch.Tensor):
        """Store w2 weight to CPU buffer (transposed) via the load pool."""
        self._weight_load_calls += 1
        dst = self.w2_weights_cpu[layer_moe_idx][expert_id]
        owned = loaded_weight.cpu().clone()
        fut = self._get_load_pool().submit(self._copy_w2, dst, owned)
        self._track_load_future(fut)

    # ------------------------------------------------------------------ #
    #  Scale / offset helpers (quantized models only)                     #
    # ------------------------------------------------------------------ #

    def _load_scale_shard(self, layer_moe_idx: int, expert_id: int,
                          attr_name: str, shard_id: str,
                          loaded_weight: torch.Tensor):
        """Load a scale/offset shard into its CPU buffer via the load pool.

        w13 scale/offset arrives as two shards (w1, w3) that must be
        concatenated along dim 0. We stash the first-arriving owned clone in
        _scale_shard_temp and assemble when the second shard arrives.
        w2 scale/offset is a single shard — clone and defer directly.
        """
        self._weight_load_calls += 1
        assert shard_id in ("w1", "w2", "w3"), f"unexpected shard_id: {shard_id}"
        target_dict = (self.scale_cpu_buffers if "scale" in attr_name
                       else self.offset_cpu_buffers)
        target = target_dict[attr_name][layer_moe_idx][expert_id]
        if attr_name.startswith("w13_"):
            key = (layer_moe_idx, expert_id, attr_name)
            pending_shard = self._scale_shard_temp.pop(key, None)
            if pending_shard is not None:
                # Second shard — own it, then defer cat + copy.
                cur_shard = loaded_weight.cpu().clone()
                if shard_id == "w1":
                    w1, w3 = cur_shard, pending_shard
                else:
                    w1, w3 = pending_shard, cur_shard
                fut = self._get_load_pool().submit(
                    self._copy_scale_assembled, target, w1, w3)
                self._track_load_future(fut)
            else:
                # First shard — stash an owned clone.
                self._scale_shard_temp[key] = loaded_weight.cpu().clone()
        else:
            # w2 scale/offset — single shard.
            owned = loaded_weight.cpu().clone()
            fut = self._get_load_pool().submit(
                self._copy_scale_direct, target, owned)
            self._track_load_future(fut)

    def refresh_fp32_scales(self):
        """Recompute the derived fp32 per-expert scale after weight loading.

        Device experts are already in place (loaded by the weight loader and
        process_weights_after_loading); this only refreshes
        w13_weight_scale_fp32 from the freshly-loaded w13_weight_scale.
        """
        for i, layer in enumerate(self.moe_layers):
            ndev = min(self.num_device_experts, layer.w13_weight.shape[0])
            if hasattr(layer, 'w13_weight_scale_fp32'):
                for j in range(ndev):
                    layer.w13_weight_scale_fp32[j].copy_(
                        layer.w13_weight_scale.data[j].to(torch.float32))

    def create_prefill_pool(self):
        """Allocate prefill pool tensors on NPU with full expert count.

        Called from _finalize_offload() after decode buffers are set up.
        Creates ndl device tensors each holding all experts (e.g. 128).
        These are used when num_tokens > offload_threshold (large-batch
        prefill), loaded via full-overwrite in _prefill_load_layer.
        """
        if self._prefill_initialized:
            return
        if not self.moe_layers:
            return
        ndl = self.num_device_layers
        pool_layer = self.moe_layers[0]
        dev = pool_layer.w13_weight.device
        dt = pool_layer.w13_weight.dtype
        ntotal = self.num_total_experts

        for _ in range(ndl):
            # w13: [ntotal, hidden_size, w13_up_dim] — match decode layer shape
            w13_shape = (ntotal,) + tuple(pool_layer.w13_weight.shape[1:])
            self._prefill_w13.append(
                torch.empty(w13_shape, dtype=dt, device=dev))

            # w2: [ntotal, hidden_size, intermediate_size_per_partition]
            w2_shape = (ntotal,) + tuple(pool_layer.w2_weight.shape[1:])
            self._prefill_w2.append(
                torch.empty(w2_shape, dtype=dt, device=dev))

            # W8A8 scale/offset (optional)
            if hasattr(pool_layer, 'w13_weight_scale'):
                s13_shape = (ntotal,) + tuple(pool_layer.w13_weight_scale.shape[1:])
                self._prefill_w13_scale.append(
                    torch.empty(s13_shape, dtype=pool_layer.w13_weight_scale.dtype, device=dev))
            if hasattr(pool_layer, 'w13_weight_scale_fp32'):
                fp32_13_shape = (ntotal,) + tuple(pool_layer.w13_weight_scale_fp32.shape[1:])
                self._prefill_w13_scale_fp32.append(
                    torch.empty(fp32_13_shape, dtype=torch.float32, device=dev))
            if hasattr(pool_layer, 'w13_weight_offset'):
                o13_shape = (ntotal,) + tuple(pool_layer.w13_weight_offset.shape[1:])
                self._prefill_w13_offset.append(
                    torch.empty(o13_shape, dtype=pool_layer.w13_weight_offset.dtype, device=dev))
            if hasattr(pool_layer, 'w2_weight_scale'):
                s2_shape = (ntotal,) + tuple(pool_layer.w2_weight_scale.shape[1:])
                self._prefill_w2_scale.append(
                    torch.empty(s2_shape, dtype=pool_layer.w2_weight_scale.dtype, device=dev))
            if hasattr(pool_layer, 'w2_weight_offset'):
                o2_shape = (ntotal,) + tuple(pool_layer.w2_weight_offset.shape[1:])
                self._prefill_w2_offset.append(
                    torch.empty(o2_shape, dtype=pool_layer.w2_weight_offset.dtype, device=dev))

        # Cast prefill pool weight tensors to NZ format (W8A8 kernel requires it).
        # Must happen BEFORE loading data — same order as decode path:
        # create → NZ-cast → copy_(cpu → npu)
        if dt == torch.int8:
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
            for i in range(ndl):
                self._prefill_w13[i] = torch_npu.npu_format_cast(
                    self._prefill_w13[i], ACL_FORMAT_FRACTAL_NZ)
                self._prefill_w2[i] = torch_npu.npu_format_cast(
                    self._prefill_w2[i], ACL_FORMAT_FRACTAL_NZ)

        # Prefill log2phy: identity — all experts mapped to their slots
        self._prefill_log2phy = torch.arange(ntotal, dtype=torch.int32, device=dev)

        # Pre-initialize all pool slots with layer 0 weights so that
        # profile_run / _dummy_run (which may use prefill path) has
        # valid data.  Subsequent _prefill_load_layer calls will
        # overwrite with the correct per-layer weights.
        self._init_prefill_pool_data(dev, ntotal, ndl)
        self._prefill_initialized = True
        logger.info("[PREFILL_POOL] allocated %d layers × %d experts, "
                    "w13[0].shape=%s w2[0].shape=%s",
                    ndl, ntotal,
                    tuple(self._prefill_w13[0].shape),
                    tuple(self._prefill_w2[0].shape))

    def _init_prefill_pool_data(self, dev, ntotal: int, ndl: int):
        """Load layer 0 weights into all prefill pool slots.

        Prefill pool tensors are already NZ-cast at this point (done in
        create_prefill_pool). Use simple per-expert copy_() — same pattern
        as the decode path's _update_weights.
        """
        has_scales = bool(self._prefill_w13_scale)
        has_offsets = bool(self._prefill_w13_offset)

        for slot in range(ndl):
            for eid in range(min(ntotal, len(self.w13_weights_cpu[0]))):
                self._prefill_w13[slot].untyped_storage()[eid * self.w13_expert_size_bytes : (eid + 1) * self.w13_expert_size_bytes].copy_(
                    self.w13_weights_cpu[0][eid].untyped_storage()
                )
                self._prefill_w2[slot].untyped_storage()[eid * self.w2_expert_size_bytes : (eid + 1) * self.w2_expert_size_bytes].copy_(
                    self.w2_weights_cpu[0][eid].untyped_storage()
                )

            # Initialize scale/offset buffers with layer 0 data (W8A8)
            if has_scales:
                for scale_name, prefill_list, cpu_buffers in [
                    ("w13_weight_scale", self._prefill_w13_scale, self.scale_cpu_buffers),
                    ("w2_weight_scale", self._prefill_w2_scale, self.scale_cpu_buffers),
                ]:
                    if (scale_name in cpu_buffers and
                            0 < len(cpu_buffers[scale_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[scale_name][0]))):
                            prefill_list[slot][eid].copy_(
                                cpu_buffers[scale_name][0][eid])
            if has_offsets:
                for offset_name, prefill_list, cpu_buffers in [
                    ("w13_weight_offset", self._prefill_w13_offset, self.offset_cpu_buffers),
                    ("w2_weight_offset", self._prefill_w2_offset, self.offset_cpu_buffers),
                ]:
                    if (offset_name in cpu_buffers and
                            0 < len(cpu_buffers[offset_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[offset_name][0]))):
                            prefill_list[slot][eid].copy_(
                                cpu_buffers[offset_name][0][eid])
            # Initialize fp32 scale (convert from scale)
            if has_scales and slot < len(self._prefill_w13_scale_fp32):
                for eid in range(min(ntotal, self._prefill_w13_scale[slot].shape[0])):
                    self._prefill_w13_scale_fp32[slot][eid].copy_(
                        self._prefill_w13_scale[slot][eid].to(torch.float32))

    def _prefill_load_layer(self, layer_idx: int, log2phy: torch.Tensor):
        """Load ALL experts for model layer layer_idx into the prefill pool.

        For W8A8: loads into normal-format scratch, then casts to NZ.
        For unquantized: loads directly into pool tensors via copy_().
        Full-overwrite into pool_slot = layer_idx % ndl.  No slot_owner
        tracking needed — log2phy is set to identity for prefill.
        """
        ndl = self.num_device_layers
        pool_slot = layer_idx % ndl
        dev = self._prefill_w13[pool_slot].device
        ntotal = self.num_total_experts
        is_w8a8 = self._prefill_w13[pool_slot].dtype == torch.int8

        if self._debug:
            logger.info("[PREFILL_LOAD] layer=%d pool_slot=%d ntotal=%d is_w8a8=%s",
                        layer_idx, pool_slot, ntotal, is_w8a8)

        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

        with torch_npu.npu.stream(self.load_stream):
            for eid in range(ntotal):
                self._prefill_w13[pool_slot].untyped_storage()[eid * self.w13_expert_size_bytes : (eid + 1) * self.w13_expert_size_bytes].copy_(
                    self.w13_weights_cpu[layer_idx][eid].untyped_storage()
                )
                self._prefill_w2[pool_slot].untyped_storage()[eid * self.w2_expert_size_bytes : (eid + 1) * self.w2_expert_size_bytes].copy_(
                    self.w2_weights_cpu[layer_idx][eid].untyped_storage()
                )

            # W8A8 scale/offset — load into prefill buffers
            for scale_name, prefill_list, cpu_buffers in [
                ("w13_weight_scale", self._prefill_w13_scale, self.scale_cpu_buffers),
                ("w2_weight_scale", self._prefill_w2_scale, self.scale_cpu_buffers),
            ]:
                if pool_slot < len(prefill_list):
                    if (scale_name in cpu_buffers and
                            layer_idx < len(cpu_buffers[scale_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[scale_name][layer_idx]))):
                            prefill_list[pool_slot][eid].copy_(
                                cpu_buffers[scale_name][layer_idx][eid])
            for offset_name, prefill_list, cpu_buffers in [
                ("w13_weight_offset", self._prefill_w13_offset, self.offset_cpu_buffers),
                ("w2_weight_offset", self._prefill_w2_offset, self.offset_cpu_buffers),
            ]:
                if pool_slot < len(prefill_list):
                    if (offset_name in cpu_buffers and
                            layer_idx < len(cpu_buffers[offset_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[offset_name][layer_idx]))):
                            prefill_list[pool_slot][eid].copy_(
                                cpu_buffers[offset_name][layer_idx][eid])

            # Refresh fp32 scale for prefill pool
            if (pool_slot < len(self._prefill_w13_scale_fp32) and
                    pool_slot < len(self._prefill_w13_scale)):
                # Copy scale data from freshly loaded scale to fp32
                for eid in range(min(ntotal, self._prefill_w13_scale[pool_slot].shape[0])):
                    self._prefill_w13_scale_fp32[pool_slot][eid].copy_(
                        self._prefill_w13_scale[pool_slot][eid].to(torch.float32))

            self.load_stream.synchronize()

        # NOTE: Do NOT modify the layer's own log2phy here — decode path
        # relies on it staying with 32-expert mapping.  Prefill path in
        # apply() explicitly uses self._prefill_log2phy instead.

    # ------------------------------------------------------------------ #
    #  Forward path: page in experts based on topk_ids                    #
    # ------------------------------------------------------------------ #

    def update_weights(self, layer, topk_ids: torch.Tensor,
                        log2phy: torch.Tensor,
                        topk_weights: torch.Tensor | None = None,
                        hidden_states: torch.Tensor | None = None) -> int:
        """Incrementally page in needed experts, overwriting unused slots.

        Routes to prefill pool (full-overwrite) when num_tokens exceeds
        offload_threshold, otherwise uses per-expert paging (decode path).

        Args:
            layer: AscendFusedMoE instance.
            topk_ids: [num_tokens, top_k] routed expert indices.
            log2phy: [global_num_experts] CPU tensor, modified in-place.
            topk_weights: Optional routing weights for cache policy.
            hidden_states: Optional [num_tokens, hidden_dim] tensor used
                           for next-layer expert prefetch prediction.

        Returns: number of CPU→NPU copies performed (decode path),
                 0 for prefill path (full-overwrite via pool).
        """
        num_tokens = topk_ids.size(0)
        if num_tokens > self.offload_threshold:
            # Prefill: layerwise reuse + full-overwrite of all experts
            if (self._prefill_initialized
                    and not self._skip_prefill):
                self.flush_sequence_cache_stats()
                try:
                    layer_idx = self.moe_layers.index(layer)
                except ValueError:
                    return 0
                self._prefill_load_layer(layer_idx, log2phy)
                return 0
            else:
                # Profile run or pool not ready — bail out gracefully
                return 0

        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return 0

        # Wait for any pending threaded prefetch for this layer to complete.
        # In graph mode there is no background thread; only the NPU event
        # stored by the host callback needs to be waited on.
        layer_done = self._prefetch_layer_done.get(layer_idx)
        if layer_done is not None:
            layer_done.wait()           # Block until prefetch thread finishes
            layer_done.clear()
            del self._prefetch_layer_done[layer_idx]

        # Wait for prefetch NPU copies to complete before using the weights.
        # Use stream wait (graphable) instead of host synchronize.
        with self._prefetch_state_lock:
            npu_event = self._prefetch_layer_npu_event.pop(layer_idx, None)
        if npu_event is not None:
            torch_npu.npu.current_stream().wait_event(npu_event)

        topk_ids_h = self.topk_ids_h[:num_tokens]
        topk_weights_h = None
        if (self.cache_policy is not None and topk_weights is not None 
                and self.offload_config.cache_router_weight != 0):
            topk_weights_h = self.topk_weights_h[:num_tokens]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32), non_blocking=_EXTRA_CTX.capturing)
        log2phy_h = self.log2phy_h
        log2phy_np = self.log2phy_np
        topk_ids_h.copy_(topk_ids, non_blocking=_EXTRA_CTX.capturing)
        log2phy_h.copy_(log2phy, non_blocking=_EXTRA_CTX.capturing)

        current_compute_stream = torch_npu.npu.current_stream()
        subscribed_compute_streams = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed_compute_streams:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed_compute_streams.add(current_compute_stream)

        args = (
            topk_ids_h,
            log2phy_np,
            layer,
            layer_idx,
            topk_weights_h,
        )
        if _EXTRA_CTX.capturing:
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self._update_weights,
                args,
            )
        else:
            self._update_weights(args)

        log2phy.copy_(log2phy_h, non_blocking=_EXTRA_CTX.capturing)

    def _update_weights(self, args):
        (
            topk_ids_h,
            log2phy_np,
            layer,
            layer_idx,
            topk_weights_h,
        ) = args
        with torch_npu.npu.stream(self.load_stream):
            if self.cache_policy is not None:
                router_scores = topk_weights_h.tolist() if topk_weights_h is not None else None
                # NEW (#3): time observe() as the "lrc" stage. observe() is pure-CPU
                # (the per-token EMA loop over num_experts + recent-window bookkeeping),
                # so perf_counter around it is the pure cost — no device sync needed.
                # Recorded into _t_pending (prehook): harvested at the hit just below,
                # via _record_cache_stats -> _record_seq_layer_hit.
                _time_lrc = (self._profile_timing and not self._seq_stats_done)
                _tl0 = time.perf_counter() if _time_lrc else 0.0
                needed = self.cache_policy.observe(
                    layer_idx,
                    topk_ids_h.tolist(),
                    router_scores=router_scores,
                )
                if _time_lrc:
                    self._t_pending["lrc"][layer_idx] = (time.perf_counter() - _tl0) * 1000.0
            else:
                needed = set(topk_ids_h.unique().tolist())

            # Build reverse map: slot → expert_id currently occupying it
            slot_owner: dict[int, int] = {}
            for eid, slot in enumerate(log2phy_np):
                if slot >= 0:
                    slot_owner[slot] = eid

            on_device = set(slot_owner.values())
            already_there = needed & on_device           # no-op
            need_to_load = needed - already_there          # CPU→NPU copy
            if self.cache_policy is not None:
                self._record_cache_stats(layer_idx, already_there, need_to_load,
                                         needed, on_device, topk_ids_h.shape[0])
                # fold the prefetch profiling sample for this layer into
                # the same step (no-op unless cache_profile_timing is on and a sample exists
                self._record_prefetch_stats(layer_idx, needed)
            reusable_slots = [s for s, e in slot_owner.items()
                            if e not in needed]          # slots to recycle

            if self._debug:
                logger.info("[UPDATE-W] l=%d expert_hit=%s expert_miss=%s hit_rate=%.2f",
                            layer_idx, sorted(already_there),
                            sorted(need_to_load), len(already_there) / 6)
                if need_to_load and len(need_to_load) > len(reusable_slots):
                    logger.info("[UPDATE-W] l=%d SHORTFALL: need %d load but only %d slots, "
                                "to_load=%s",
                                layer_idx, len(need_to_load), len(reusable_slots),
                                sorted(need_to_load)[:20])

            # start the load timer just before the copy loop; read it after
            # the existing load_stream.synchronize() (no extra sync introduced).
            # CHANGE (#6): renamed local _time_upload -> _time_miss_load and the
            # recorded metric "upload" -> "miss_load" (see __init__/labels). This
            # measures the on-demand miss path = evict-pick + scale/offset + weight
            # H2D, not a pure transfer.
            _time_miss_load = self._profile_timing and self.cache_policy is not None
            _tc0 = time.perf_counter() if _time_miss_load else 0.0

            n_copies = 0
            for eid in need_to_load:
                if self.cache_policy is not None:
                    victim = self.cache_policy.choose_victim(
                        layer_idx,
                        slot_owner,
                        protected=needed,
                    )
                    slot = int(log2phy_np[victim]) if victim is not None else -1
                elif reusable_slots:
                    slot = reusable_slots.pop()
                    victim = slot_owner[slot]
                else:
                    slot = -1
                    victim = None

                if slot < 0:
                    # CHANGE (#11): unconditional WARNING — see Group 2. This is the
                    # silent mis-route path (unmapped expert -> clamp(min=0) -> expert 0).
                    logger.warning(
                        "[UPDATE-W] l=%d NO SLOTS: %d experts could not be loaded; "
                        "their tokens will be MIS-ROUTED to expert 0. missed=%s",
                        layer_idx, len(need_to_load) - n_copies,
                        sorted(list(need_to_load))[n_copies:][:20])
                    break  # no free slots — should not happen in normal usage
                # Copy weights from CPU to NPU
                layer.w13_weight.data.untyped_storage()[slot * self.w13_expert_size_bytes : (slot + 1) * self.w13_expert_size_bytes].copy_(
                    self.w13_weights_cpu[layer_idx][eid].untyped_storage()
                )
                layer.w2_weight.data.untyped_storage()[slot * self.w2_expert_size_bytes : (slot + 1) * self.w2_expert_size_bytes].copy_(
                    self.w2_weights_cpu[layer_idx][eid].untyped_storage()
                )
                # Copy scales/offsets from CPU to NPU
                for attr_name, buffers in self.scale_cpu_buffers.items():
                    if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                        continue
                    dev_tensor = getattr(layer, attr_name, None)
                    if dev_tensor is None:
                        continue
                    dev_tensor.data[slot].copy_(buffers[layer_idx][eid])
                for attr_name, buffers in self.offset_cpu_buffers.items():
                    if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                        continue
                    dev_tensor = getattr(layer, attr_name, None)
                    if dev_tensor is None:
                        continue
                    dev_tensor.data[slot].copy_(buffers[layer_idx][eid])
                # Refresh derived fp32 scale if present (W8A8_DYNAMIC)
                if hasattr(layer, 'w13_weight_scale_fp32'):
                    layer.w13_weight_scale_fp32[slot].copy_(
                        layer.w13_weight_scale.data[slot].to(torch.float32))
                # Update mapping
                if victim is None:
                    victim = slot_owner[slot]
                log2phy_np[victim] = -1             # evict old occupant
                log2phy_np[eid] = slot               # assign slot to new expert
                slot_owner[slot] = eid
                if slot in reusable_slots:
                    reusable_slots.remove(slot)
                n_copies += 1

            self.load_stream.synchronize()

            if _time_miss_load:
                copy_ms = (time.perf_counter() - _tc0) * 1000.0 if n_copies else 0.0
                # CHANGE (#6): "upload" -> "miss_load".
                self._t_step["miss_load"][layer_idx] = copy_ms

    # ------------------------------------------------------------------ #
    #  Next-layer expert prefetch                                          #
    # ------------------------------------------------------------------ #

    def predict_next_layer_experts(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> set[int] | None:
        """Predict which experts layer layer_idx+1 will need.

        Uses the current layer's hidden_states as an approximation of
        the next layer's input, multiplied by the next layer's gate weight
        to get predicted router logits.  A simplified softmax + topk is
        used instead of the full grouped_topk for speed; misses are
        handled by the reactive fallback in update_weights().

        Args:
            layer_idx: Current layer index.
            hidden_states: [num_tokens, hidden_dim] NPU tensor.

        Returns:
            Set of predicted expert IDs, or None if prediction is not
            possible (e.g. last layer, no gate weight).
        """
        # CHANGE: delegate to the pluggable predictor (default FATEPredictor).
        # The FATE implementation is identical to the previous body (early-out
        # before .cpu(), then CPU linear+softmax+topk) — see
        # expert_predictor.py. Callers (the prefetch worker) are unchanged.
        return self._predictor.predict_from_device(layer_idx, hidden_states)

    def _predict_next_layer_experts_cpu(
        self,
        layer_idx: int,
        hs_cpu: torch.Tensor,
    ) -> set[int] | None:
        """Pure-CPU prediction from an already-CPU hidden_states tensor.

        Same approximation as predict_next_layer_experts (simplified softmax +
        topk), but takes a CPU tensor directly — no .cpu() / blocking D2H.
        Used by the graph-mode host callback over the pre-staged pinned
        buffer (_prefetch_hs_h).
        """
        # CHANGE: delegate to the pluggable predictor (default FATEPredictor).
        # Behavior is identical to the previous body. The graph-mode host
        # callback (_do_prefetch_host_callback) is unchanged — it still calls
        # this method over the pre-staged pinned buffer.
        return self._predictor.predict_from_host(layer_idx, hs_cpu)

    def predict_next_layer_experts_npu(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Predict which experts layer layer_idx+1 will need, on NPU.

        Same approximation as predict_next_layer_experts (simplified
        softmax + topk), but runs entirely on the NPU so it can be
        captured in a CUDA/NPU graph.  The returned tensor lives on NPU.

        Args:
            layer_idx: Current layer index.
            hidden_states: [num_tokens, hidden_dim] NPU tensor.

        Returns:
            [num_tokens * topk] NPU int64 tensor of predicted expert IDs,
            or None if prediction is not possible.
        """
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return None  # last layer — nothing to prefetch

        if next_idx >= len(self._gate_weights_npu):
            return None
        gate_w = self._gate_weights_npu[next_idx]
        if gate_w is None:
            return None

        # On-device prediction: [num_tokens, hidden_dim] x [n_experts, hidden_dim]^T
        router_logits = F.linear(hidden_states.float(), gate_w)
        probs = router_logits.softmax(dim=-1)
        _, topk_ids = probs.topk(self.topk, dim=-1)  # [num_tokens, topk]
        predicted = topk_ids.flatten()               # [num_tokens * topk]
        return predicted

    # ------------------------------------------------------------------ #
    #  Threaded prefetch: background thread for expert weight loading     #
    # ------------------------------------------------------------------ #

    def _start_prefetch_thread(self):
        """Start the background prefetch thread (called once, lazily)."""
        self._npu_device = torch_npu.npu.current_stream().device
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
            name="ExpertPrefetch",
        )
        self._prefetch_thread.start()
        self._prefetch_thread_ready.wait()

    def _prefetch_worker(self):
        """Background thread: processes prefetch requests from the queue.

        Each request is a tuple of (layer_idx, hidden_states, compute_event).
        The thread waits for the current layer's GMM kernel to complete
        (via compute_event), then predicts the next layer's needed experts
        and loads them from CPU to NPU.
        """
        torch.npu.set_device(self._npu_device)
        self._prefetch_thread_ready.set()

        while True:
            request = self._prefetch_queue.get()
            if request is None:
                break

            layer_idx, hidden_states, compute_event = request
            next_idx = layer_idx + 1

            # Wait for current layer's GMM kernel to complete before
            # reading hidden_states and touching next layer's weights.
            compute_event.synchronize()

            # NEW: EAGER-path profiling gate. Mirrors _record_layer_time's gate
            # exactly (profiling + cache policy + summary not yet printed) so
            # the stash can never grow without a consumer. When off, the timer
            # calls below are skipped entirely — zero overhead.
            _record_profiling = (self._profile_timing
                                 and self.cache_policy is not None
                                 and not self._seq_stats_done)

            # Predict which experts the next layer needs.
            _t0 = time.perf_counter() if _record_profiling else 0.0
            predicted = self.predict_next_layer_experts(layer_idx,
                                                        hidden_states)
            _compute_ms = ((time.perf_counter() - _t0) * 1000.0 if _record_profiling else 0.0)
            if predicted is None or next_idx >= len(self.moe_layers):
                # No prediction possible — signal done immediately
                done = self._prefetch_layer_done.get(next_idx)
                if done is not None:
                    done.set()
                continue

            # Snapshot next layer's current log2phy to CPU.  This runs on the
            # standalone prefetch thread (after compute_event.synchronize()),
            # so the blocking .cpu() is safe here — unlike the graph host
            # callback, which stages this into a pinned buffer instead.
            next_layer = self.moe_layers[next_idx]
            log2phy_np = next_layer.log2phy.cpu().numpy().copy()

            # Execute the prefetch (H2D copies)
            completion_event = self._do_prefetch(next_idx, predicted,
                                                 log2phy_np, use_sleep=False,
                                                 record_profiling=_record_profiling,
                                                 compute_ms=_compute_ms)

            # Store NPU completion event and signal threading.Event
            with self._prefetch_state_lock:
                if completion_event is not None:
                    self._prefetch_layer_npu_event[next_idx] = completion_event
            done = self._prefetch_layer_done.get(next_idx)
            if done is not None:
                done.set()

    def _do_prefetch(
        self,
        next_idx: int,
        predicted_experts: set[int],
        log2phy_np,
        use_sleep: bool = True,
        record_profiling: bool = False,
        compute_ms: float = 0.0,
    ) -> torch_npu.npu.Event | None:
        """Load predicted experts for layer next_idx from CPU to NPU.

        Runs on the prefetch thread (non-graph) or from the graph host
        callback (graph mode), using self._prefetch_stream for the H2D
        copies.  Returns an NPU Event that signals when all copies are
        complete, or None if no copies were needed.

        Args:
            next_idx: Index of the layer to prefetch experts for.
            predicted_experts: Set of expert IDs predicted to be needed.
            log2phy_np: numpy view of next_idx's current log2phy (CPU,
                int32).  Provided by the caller — the graph path stages it
                into a pinned buffer before the callback so this function
                never does a blocking .cpu() on a live graph tensor.
            use_sleep: If True, insert a small delay between w13 and w2 copies
                to avoid burst traffic.  Should be False when called from a
                graph host callback to avoid blocking graph execution.
            record_profiling: NEW. EAGER worker only. When True, time the H2D
                copy loop (perf_counter + prefetch-stream synchronize, mirroring
                the on-demand "upload" timer in _update_weights) and stash the
                {predicted, already_there, need_to_load, compute_ms, h2d_ms}
                sample for the main thread. Always False from the graph host
                callback, so graph mode adds no profiling work and no sync.
            compute_ms: NEW. CPU prediction time measured by the caller (worker),
                stashed alongside the H2D time.
        """
        next_layer = self.moe_layers[next_idx]

        # Copy out so the async H2D write-back (next_layer.log2phy.copy_)
        # reads an independent buffer.  In graph mode log2phy_np is the shared
        # _prefetch_log2phy_np view; without this copy the async H2D would
        # race the next layer's staging D2H that overwrites it.  Tiny:
        # num_total_experts * 4 bytes.
        log2phy_np = log2phy_np.copy()

        # Determine which predicted experts are not already on device
        slot_owner: dict[int, int] = {}
        for eid, slot in enumerate(log2phy_np):
            if slot >= 0:
                slot_owner[slot] = eid
        on_device = set(slot_owner.values())
        need_to_load = predicted_experts - on_device
        already_there = on_device & predicted_experts

        if not need_to_load:
            # prediction still happened — record it so the prediction
            # accuracy reflects this step.
            if record_profiling:
                self._stash_prefetch_profile(
                    next_idx, predicted_experts, already_there,
                    need_to_load, compute_ms, 0.0)
            return None

        # Protected set: experts we must not evict
        protected = set(predicted_experts)
        reusable_slots = [s for s, e in slot_owner.items()
                          if e not in protected]
        if not reusable_slots:
            # prediction recorded;
            if record_profiling:
                self._stash_prefetch_profile(
                    next_idx, predicted_experts, already_there,
                    need_to_load, compute_ms, 0.0)
            return None  # all resident experts are protected — skip
        
        if self._debug:
                logger.info("[PREFETCH-W] l=%d prefetch_expert_hit=%s prefetch_expert_miss=%s hit_rate=%.2f",
                            next_idx, sorted(already_there),
                            sorted(need_to_load), len(already_there) / 6)
                if need_to_load and len(need_to_load) > len(reusable_slots):
                    logger.info("[PREFETCH-W] l=%d SHORTFALL: need %d load but only %d slots, "
                                "to_load=%s",
                                next_idx, len(need_to_load), len(reusable_slots),
                                sorted(need_to_load)[:20])

        # bracket the H2D copy loop for the pf_h2d metric.
        _tp0 = time.perf_counter() if record_profiling else 0.0

        with torch_npu.npu.stream(self._prefetch_stream):
            n_copies = 0
            for eid in need_to_load:
                # Use cache_policy.choose_victim() for eviction, same as
                # _update_weights.  Pass next_idx so it queries L+1's
                # hotness statistics.
                if self.cache_policy is not None:
                    victim = self.cache_policy.choose_victim(
                        next_idx,
                        slot_owner,
                        protected=protected,
                        loading=need_to_load,
                    )
                    slot = (int(log2phy_np[victim])
                            if victim is not None else -1)
                elif reusable_slots:
                    slot = reusable_slots.pop()
                    victim = slot_owner[slot]
                else:
                    slot = -1
                    victim = None

                if slot < 0:
                    if self._debug:
                        logger.info(
                            "[PREFETCH] l=%d NO SLOTS: %d experts could not be prefetched, missed= %s", 
                        next_idx , len(need_to_load)-n_copies,sorted(list(need_to_load))[n_copies:][:20]
                        )
                    break

                # CPU → NPU async copy w13 weights
                next_layer.w13_weight.data.untyped_storage()[
                    slot * self.w13_expert_size_bytes
                    : (slot + 1) * self.w13_expert_size_bytes
                ].copy_(
                    self.w13_weights_cpu[next_idx][eid].untyped_storage())

                if use_sleep:
                    time.sleep(0.00025)  # 0.25ms delay, avoid burst in thread mode

                # CPU → NPU async copy w2 weights
                next_layer.w2_weight.data.untyped_storage()[
                    slot * self.w2_expert_size_bytes
                    : (slot + 1) * self.w2_expert_size_bytes
                ].copy_(
                    self.w2_weights_cpu[next_idx][eid].untyped_storage())

                # Copy scales/offsets from CPU to NPU (W8A8)
                for attr_name, buffers in self.scale_cpu_buffers.items():
                    if next_idx >= len(buffers):
                        continue
                    if eid >= len(buffers[next_idx]):
                        continue
                    dev_tensor = getattr(next_layer, attr_name, None)
                    if dev_tensor is not None:
                        dev_tensor.data[slot].copy_(
                            buffers[next_idx][eid])
                for attr_name, buffers in self.offset_cpu_buffers.items():
                    if next_idx >= len(buffers):
                        continue
                    if eid >= len(buffers[next_idx]):
                        continue
                    dev_tensor = getattr(next_layer, attr_name, None)
                    if dev_tensor is not None:
                        dev_tensor.data[slot].copy_(
                            buffers[next_idx][eid])

                # Refresh fp32 scale (W8A8_DYNAMIC)
                if hasattr(next_layer, 'w13_weight_scale_fp32'):
                    next_layer.w13_weight_scale_fp32[slot].copy_(
                        next_layer.w13_weight_scale.data[slot].to(
                            torch.float32))

                # Update log2phy mapping
                if victim is None:
                    victim = slot_owner[slot]
                log2phy_np[victim] = -1
                log2phy_np[eid] = slot
                slot_owner[slot] = eid
                if slot in reusable_slots:
                    reusable_slots.remove(slot)
                n_copies += 1

            # Write modified log2phy back to next layer's NPU tensor
            next_layer.log2phy.copy_(
                torch.from_numpy(log2phy_np).to(
                    device=next_layer.log2phy.device))

            # EAGER-only. Synchronize the prefetch stream to measure actual
            # H2D completion (worker thread, off the compute path), then stash
            # the full sample for the main thread.
            if record_profiling:
                self._prefetch_stream.synchronize()
                _h2d_ms = (time.perf_counter() - _tp0) * 1000.0
                self._stash_prefetch_profile(
                    next_idx, predicted_experts, already_there,
                    need_to_load, compute_ms, _h2d_ms)

            # Record prefetch completion event
            completion_event = torch_npu.npu.Event()
            self._prefetch_stream.record_event(completion_event)
            return completion_event

    def _stash_prefetch_profile(self, next_idx: int,
                                predicted: set[int],
                                already_there: set[int],
                                need_to_load: set[int],
                                compute_ms: float,
                                h2d_ms: float) -> None:
        """Stash one layer's prefetch profiling sample for the main thread.

        EAGER worker only (callers gate on record_profiling, which is
        cache_profile_timing-gated). The main thread pops this in
        _update_weights once it knows the ground-truth routed set, computes the
        accuracy fractions, and folds compute/h2d ms + the fractions into the
        _t_* summary. Keyed by the TARGET layer (next_idx). Guarded by
        _prefetch_state_lock for the cross-thread handoff. Last-writer-wins:
        each step overwrites the prior sample for this layer, and the main
        thread pops it, so the dict stays bounded by num_layers.
        """
        with self._prefetch_state_lock:
            self._prefetch_profile_pending[next_idx] = {
                "predicted": set(predicted),
                "already_there": set(already_there),
                "need_to_load": set(need_to_load),
                "compute_ms": compute_ms,
                "h2d_ms": h2d_ms,
            }

    def _ai_layer_index(self, experts) -> int:
        return self.moe_layers.index(experts)
     
    def ai_predict_start(self, experts, pre_attn):
        """Before attention: predict THIS layer's experts on the prefetch stream
        (no host block) and open the prefetch handshake. Eager + decode only.
    
        When profiling is on, bracket the on-NPU predict with timing events so
        ai_prefetch_finish can fold the predictor's NPU time into pf_compute."""
        if not self._ai_active or _EXTRA_CTX.capturing:
            return None
        predictor = self._predictor
        if not getattr(predictor, "ready", False):
            return None
        n = pre_attn.shape[0]
        if n > self.offload_threshold:                 # prefill / large batch
            return None
        moe_idx = self._ai_layer_index(experts)
        head_idx = predictor.head_index(moe_idx)       # map MoE index -> head
        if head_idx is None:                           # leading hash/dense layer: no AI prefetch
            return None
        spec = predictor.input_spec
    
        pa = None
        if "pre_attn" in spec:
            pa = self._ai_pa_buf[:n]
            pa.copy_(pre_attn.to(torch.float32))
        # One event after the copy orders this copy AND the prior router_input save
        # (same compute stream) before the prefetch-stream reads.
        copy_event = torch_npu.npu.current_stream().record_event()
    
        router_prev = None
        if "router_input_prev" in spec and head_idx > 0:   # head 0 uses zeros
            router_prev = self._ai_ri_buf[(moe_idx - 1) % 2][:n]
    
        # Mirror the FATE worker's profiling gate exactly (profiling + cache policy +
        # summary not yet printed) so the stash can never grow without a consumer.
        _record_profiling = (self._profile_timing and self.cache_policy is not None
                            and not self._seq_stats_done)
        # NEW: NPU timing events bracketing the predict (the pf_compute source for
        # the learned predictor). Only created when profiling — zero overhead off.
        predict_ev0 = predict_ev1 = None
        if _record_profiling:
            predict_ev0 = torch_npu.npu.Event(enable_timing=True)
            predict_ev1 = torch_npu.npu.Event(enable_timing=True)
    
        self._prefetch_layer_done[moe_idx] = threading.Event()
        ctx = AIPredictCtx(layer_idx=head_idx, n_tokens=n,
                        pre_attn=pa, router_input_prev=router_prev)
    
        with torch_npu.npu.stream(self._prefetch_stream):
            self._prefetch_stream.wait_event(copy_event)
            if predict_ev0 is not None:
                predict_ev0.record()
            logits = predictor.predict_logits_npu(head_idx, ctx)      # [n, E]
            topk_ids = logits.topk(predictor.top_k, dim=-1).indices    # [n, k]
            if predict_ev1 is not None:
                predict_ev1.record()
        self._ai_pending[moe_idx] = {"moe_idx": moe_idx, "topk_ids": topk_ids,
                                    "predict_ev0": predict_ev0,
                                    "predict_ev1": predict_ev1}
        return self._ai_pending[moe_idx]
    
    def ai_prefetch_finish(self, ctx):
        """After the attention CALL is issued: D2H topk (hidden behind attention),
        reuse _do_prefetch for eviction + H2D, record the handshake update_weights
        waits on. Folds the predictor's NPU compute time into pf_compute. Eager only."""
        if ctx is None:
            return
        moe_idx = ctx["moe_idx"]
        # .tolist() blocks until the predict finishes -> by the line after, the
        # predict's NPU timing events are complete and elapsed_time() won't block.
        predicted = set(int(e) for e in ctx["topk_ids"].flatten().tolist())
    
        compute_ms = 0.0
        ev0 = ctx.get("predict_ev0")
        ev1 = ctx.get("predict_ev1")
        if ev0 is not None and ev1 is not None:
            # NPU time of the predict (runs on _prefetch_stream, overlapping attn).
            compute_ms = ev0.elapsed_time(ev1)
    
        _record_profiling = (self._profile_timing and self.cache_policy is not None
                            and not self._seq_stats_done)
    
        next_layer = self.moe_layers[moe_idx]
        log2phy_np = next_layer.log2phy.cpu().numpy()
        # _do_prefetch stashes {predicted, ..., compute_ms, h2d_ms}; the main thread
        # folds compute_ms into the pf_compute metric in _update_weights — the SAME
        # path FATE uses, so FATE (CPU ms) and the learned predictor (NPU ms) share
        # the metric.
        event = self._do_prefetch(
            moe_idx, predicted, log2phy_np,
            use_sleep=False, record_profiling=_record_profiling,
            compute_ms=compute_ms,
        )
        with self._prefetch_state_lock:
            if event is not None:
                self._prefetch_layer_npu_event[moe_idx] = event
        done = self._prefetch_layer_done.get(moe_idx)
        if done is not None:
            done.set()
        self._ai_pending.pop(moe_idx, None)

    def ai_save_router_input(self, experts, router_input: torch.Tensor):
        """Post-attention: copy this layer's gate input into ping-pong slot ℓ%2
        for ℓ+1's predict. Only when the active predictor consumes it. Eager +
        decode only."""
        if not self._ai_active or _EXTRA_CTX.capturing:
            return
        if "router_input_prev" not in self._predictor.input_spec:
            return
        if not getattr(self._predictor, "ready", False):
            return
        n = router_input.shape[0]
        if n > self.offload_threshold:
            return
        moe_idx = self._ai_layer_index(experts)
        if self._predictor.head_index(moe_idx) is None:   # <-- NEW: don't save for leading layers
            return
        self._ai_ri_buf[moe_idx % 2][:n].copy_(router_input.to(torch.float32))

    def trigger_next_layer_prefetch(self, layer,
                                    hidden_states: torch.Tensor):
        """在 GMM kernel 提交后触发下一层专家预加载。

        必须在 fused_experts() 之后调用，使 compute_event 捕获
        GMM kernel 的 NPU 工作，实现预加载与计算的真正并行。

        图模式下通过 _launch_host_func 提交到 host callback 执行；
        非图模式下提交到后台线程，主线程立即返回，不被 aclrtMemcpy 阻塞。

        Args:
            layer: 当前 MoE 层的 AscendFusedMoE 实例。
            hidden_states: [num_tokens, hidden_dim] NPU tensor。
        """
        if not self.offload_config.expert_prefetch_enabled:
            return
        if self._ai_active:
            return
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return

        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return

        # Record compute event on main thread's compute stream
        # (captures GMM kernel progress so prefetch waits for it)
        compute_event = torch_npu.npu.Event()
        torch_npu.npu.current_stream().record_event(compute_event)

        if _EXTRA_CTX.capturing:
            # Graph mode: launch a host callback that runs prediction and
            # H2D copies on the prefetch stream.  The callback itself is
            # synchronous, but the copies it enqueues are asynchronous.
            #
            # Stage the data the callback needs into pinned CPU *before*
            # launching it, with non_blocking D2H on the compute stream.
            # _launch_host_func guarantees the callback runs only after the
            # stream ops queued before it complete, so the callback reads
            # already-ready memory and never does a blocking .cpu() on a live
            # graph tensor (which would deadlock the stream's host thread).
            # Mirrors update_weights' staging pattern exactly.
            next_layer = self.moe_layers[next_idx]
            num_tokens = hidden_states.size(0)
            hs_h = self._prefetch_hs_h[:num_tokens]
            # Cast to fp32 on-device first, then D2H into the fp32 pinned
            # buffer — mirrors update_weights (topk_weights.to(float32)),
            # keeping the captured op identical to the proven path rather
            # than relying on a cross-dtype copy_.
            hs_h.copy_(hidden_states.to(torch.float32),
                       non_blocking=_EXTRA_CTX.capturing)
            self._prefetch_log2phy_h.copy_(next_layer.log2phy,
                                           non_blocking=_EXTRA_CTX.capturing)
            args = (layer_idx, hs_h, compute_event)
            torch_npu.npu._launch_host_func(
                torch_npu.npu.current_stream(),
                self._do_prefetch_host_callback,
                args,
            )
        else:
            # Non-graph mode: use the background prefetch thread.
            if self._prefetch_thread is None:
                self._start_prefetch_thread()

            # Create per-layer threading.Event for completion signaling
            self._prefetch_layer_done[next_idx] = threading.Event()

            # Submit to background thread — returns immediately!
            self._prefetch_queue.put((layer_idx, hidden_states, compute_event))

    def _do_prefetch_host_callback(self, args):
        """Host callback for graph-mode prefetch.

        Runs on the CPU during graph replay (the compute stream's host
        thread).  It predicts the next layer's experts, waits for the
        current layer's GMM on the prefetch stream, and enqueues async H2D
        copies.  The returned completion event is stored for
        update_weights() to wait on.

        Must NOT block or do any synchronous D2H: the data it needs
        (hidden_states and next layer's log2phy) is pre-staged into pinned
        buffers by trigger_next_layer_prefetch on the compute stream before
        this callback is launched, so reads here are always ready.
        """
        layer_idx, hs_cpu, compute_event = args
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return

        # Pure-CPU prediction over the already-staged hidden_states — no .cpu().
        predicted = self._predict_next_layer_experts_cpu(layer_idx, hs_cpu)
        if predicted is None:
            return

        # Enqueue prefetch work on the prefetch stream after the GMM event.
        with torch_npu.npu.stream(self._prefetch_stream):
            self._prefetch_stream.wait_event(compute_event)
            completion_event = self._do_prefetch(
                next_idx, predicted, self._prefetch_log2phy_np,
                use_sleep=False)

        # Store completion event for update_weights() to wait on.
        if completion_event is not None:
            with self._prefetch_state_lock:
                self._prefetch_layer_npu_event[next_idx] = completion_event

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stats4(values: list[float]) -> tuple:
        """avg, median, min, max of a non-empty list."""
        return (sum(values) / len(values), statistics.median(values),
                min(values), max(values))

    def _record_prefetch_stats(self, layer_idx: int, needed: set[int]) -> None:
        """Harvest this layer's prefetch profiling sample into the step (EAGER).

        Computes the three prefetch accuracy fractions against the ground-truth
        routed set `needed` (= G) and folds compute/h2d ms + the fractions into
        the current step's _t_step dict — same alignment and timing as the
        "upload" metric (both recorded inside _update_weights), so they land in
        the same decode step.

        No-op when profiling is off, the cache policy is absent, the summary
        already printed, or there is no sample for this layer (layer 0 is never
        a prefetch target; graph mode never populated the stash). The gate
        matches _record_layer_time exactly. All denominators are guarded so an
        empty set yields 0.0 — never NaN/Inf.
        """
        if (not self._profile_timing or self.cache_policy is None
                or self._seq_stats_done):
            return
        with self._prefetch_state_lock:
            entry = self._prefetch_profile_pending.pop(layer_idx, None)
        if entry is None:
            return

        predicted = entry["predicted"]
        # Denominator is len(P), the deduped predicted set — general for
        # batch > 1, unlike the hardcoded "/ 6" in the [PREFETCH-W] debug log.
        n_pred = len(predicted)
        need_to_load = entry["need_to_load"]
        n_ntl = len(need_to_load)

        # 1) Prediction accuracy (precision): of what we predicted, how much is
        #    actually routed this step.
        pred_acc = len(predicted & needed) / n_pred if n_pred else 0.0
        # 2) Predicted experts already resident in the LRC cache (overlap of the
        #    two mechanisms): high => LRC already covered the prediction;
        #    low => prefetch and LRC complement each other.
        pf_in_lrc = len(entry["already_there"]) / n_pred if n_pred else 0.0
        # 3) Usefulness of the experts prefetch actually loads (the "miss" set):
        #    of those, how many are truly needed. When need_to_load is empty no
        #    transfer happened this step → 0.0 (kept in the average so no-op
        #    steps are visible).
        pf_useful = len(need_to_load & needed) / n_ntl if n_ntl else 0.0

        self._t_step["pf_compute"][layer_idx] = entry["compute_ms"]
        self._t_step["pf_h2d"][layer_idx] = entry["h2d_ms"]
        self._t_step["pred_acc"][layer_idx] = pred_acc
        self._t_step["pf_in_lrc"][layer_idx] = pf_in_lrc
        self._t_step["pf_useful"][layer_idx] = pf_useful

    def _record_cache_stats(
        self,
        layer_idx: int,
        hit_experts: set[int],
        miss_experts: set[int],
        needed: set[int],
        on_device: set[int],
        num_tokens: int = 1,
    ):
        # Feed the per-(layer, step) hit rate + decode batch size into the
        # end-of-test summary (decode path only; main thread only).
        self._record_seq_layer_hit(layer_idx, len(hit_experts), len(needed), num_tokens)
        self.cache_calls[layer_idx] += 1
        self.cache_requests[layer_idx] += len(needed)
        self.cache_hits[layer_idx] += len(hit_experts)
        self.cache_misses[layer_idx] += len(miss_experts)
        self.last_hit_experts[layer_idx] = sorted(hit_experts)
        self.last_miss_experts[layer_idx] = sorted(miss_experts)

        interval = self.offload_config.cache_stats_log_interval
        if interval == 0 or self.cache_calls[layer_idx] % interval != 0:
            return

        requests = self.cache_requests[layer_idx]
        hit_rate = self.cache_hits[layer_idx] / requests if requests else 0.0
        policy_step = -1
        if self.cache_policy is not None:
            policy_step = self.cache_policy.layer_step(layer_idx)
        logger.info(
            "[EXPERT-OFFLOAD-CACHE] layer=%d cache_step=%d calls=%d policy_step=%d "
            "hit_rate=%.4f hits=%d misses=%d last_hit=%s last_miss=%s resident=%s",
            layer_idx,
            self.cache_calls[layer_idx],
            self.cache_calls[layer_idx],
            policy_step,
            hit_rate,
            self.cache_hits[layer_idx],
            self.cache_misses[layer_idx],
            self.last_hit_experts[layer_idx],
            self.last_miss_experts[layer_idx],
            sorted(on_device),
        )

    def _record_layer_time(self, metric: str, layer, ms: float):
        """Generic per-(layer, step) timing record.

        Pre-hook metrics (attn, router) go to a pending slot, harvested at the
        hit; post-hook metrics (upload, compute, shared) stash directly. No-op
        unless timing + cache policy are on and the summary hasn't printed.
        """
        if (not self._profile_timing or self.cache_policy is None
                or self._seq_stats_done):
            return
        try:
            layer_idx = self.moe_layers.index(layer)
        except (ValueError, AttributeError):
            return
        if metric in self._t_pending:
            self._t_pending[metric][layer_idx] = ms
        else:
            self._t_step[metric][layer_idx] = ms

    # Public call sites (thin wrappers, named for readability).
    def record_attention_time(self, layer, ms): self._record_layer_time("attn", layer, ms)
    def record_router_time(self, layer, ms):    self._accumulate_pending("router", layer, ms)
    def record_compute_time(self, layer, ms):   self._record_layer_time("compute", layer, ms)
    def record_shared_time(self, layer, ms):    self._record_layer_time("shared", layer, ms)
    # "miss_load" is recorded internally in _update_weights.

    # NEW (#2,#3,#6): router/gate now share one metric, so a pending value may be
    # written by more than one call site (gate timer + select timer) before the
    # hit harvests it. Accumulate instead of overwrite. Same gate as
    # _record_layer_time (profiling + cache policy + summary not printed).
    def _accumulate_pending(self, metric: str, layer, ms: float) -> None:
        if (not self._profile_timing or self.cache_policy is None
                or self._seq_stats_done):
            return
        try:
            idx = self.moe_layers.index(layer)
        except (ValueError, AttributeError):
            return
        self._t_pending[metric][idx] = self._t_pending[metric].get(idx, 0.0) + ms

    # NEW (#2): gate GEMM time folds into the router stage. Called from the gate
    # site (fused_moe.shared_forward_impl for is_internal_router, else
    # DeepseekV4MoE.forward). Accumulates into the same pending slot select_experts
    # writes to via record_router_time.
    def record_gate_time(self, layer, ms):
        self._accumulate_pending("router", layer, ms)

    # NEW (#2,#4): pure-isolation stage timer. Drains the COMPUTE stream before and
    # after the wrapped block so `metric` reflects the block's cost with nothing
    # else running on that stream (combine with quiesce_for_timing, which drains
    # the prefetch/load streams + the FATE worker, for full isolation). prehook
    # metrics land in _t_pending (harvested at the cache hit); post=True writes
    # _t_step directly (for ops that run after the hit, e.g. the FFN-side hc_post).
    # accumulate=True adds to any existing value for this layer/step (several
    # segments -> one metric). No-op off profiling / capture / large batch.
    @contextmanager
    def time_stage(self, metric: str, experts, n_tokens: int,
                   accumulate: bool = False, post: bool = False):
        if (not self._profile_timing or self.cache_policy is None
                or self._seq_stats_done or _EXTRA_CTX.capturing
                or n_tokens > self.offload_threshold):
            yield
            return
        try:
            idx = self.moe_layers.index(experts)
        except (ValueError, AttributeError):
            yield
            return
        torch_npu.npu.current_stream().synchronize()
        _t0 = time.perf_counter()
        try:
            yield
        finally:
            torch_npu.npu.current_stream().synchronize()
            ms = (time.perf_counter() - _t0) * 1000.0
            d = self._t_step if post else self._t_pending
            slot = d.setdefault(metric, {})
            slot[idx] = (slot.get(idx, 0.0) + ms) if accumulate else ms

    # NEW (#4): pure-isolation prerequisite. Before timing this layer's stages,
    # drain everything that would otherwise run concurrently on another stream:
    #   - FATE: this layer's prefetch runs on the background worker (triggered by
    #     the PREVIOUS layer's compute). Wait for it to finish, then drain the
    #     prefetch stream so the H2D + log2phy write-back are complete. We PEEK the
    #     threading.Event (no clear/pop) — update_weights still consumes it later.
    #   - NN: the predict was launched on the prefetch stream in ai_predict_start
    #     (this layer, before attention). Draining the prefetch stream forces it to
    #     complete; its NPU cost is still captured by the predict's own events.
    # Also drain the on-demand load stream (idle here, cheap insurance). No-op off
    # profiling / capture / large batch. Must run AFTER ai_predict_start and BEFORE
    # the attention timer. NOTE: FATE's pf_h2d can still include contention with the
    # PREVIOUS layer's epilogue (the worker H2D may begin before this drain); see
    # the guide's "FATE pf_h2d" note.
    def quiesce_for_timing(self, experts, n_tokens: int):
        if (not self._profile_timing or self.cache_policy is None
                or self._seq_stats_done or _EXTRA_CTX.capturing
                or n_tokens > self.offload_threshold):
            return
        try:
            layer_idx = self.moe_layers.index(experts)
        except (ValueError, AttributeError):
            return
        if not self._ai_active:
            done = self._prefetch_layer_done.get(layer_idx)
            if done is not None:
                done.wait()           # PEEK: do not clear/del; update_weights does
        self._prefetch_stream.synchronize()
        self.load_stream.synchronize()

    def _record_seq_layer_hit(self, layer_idx, num_hits, num_requested, num_tokens=1):
        if self._seq_stats_done or num_requested <= 0:
            return
        if layer_idx in self._seq_token_layer_hits:
            self._close_token_stats()
        self._seq_token_layer_hits[layer_idx] = num_hits / num_requested
        self._pending_token_batch = num_tokens
        # Harvest pre-hook timings (attn, router) recorded for THIS layer just
        # before the hook, into the current step's dict (after any close).
        if self._profile_timing:
            for m in self._t_prehook:
                if layer_idx in self._t_pending[m]:
                    self._t_step[m][layer_idx] = self._t_pending[m].pop(layer_idx)

    def _close_token_stats(self):
        rates_by_layer = self._seq_token_layer_hits
        if not rates_by_layer:
            return
        # CHANGE: cross-layer summary stats (avg/median/min/max + per-step total)
        # are computed over MoE layers only — drop the leading num_hash_layers
        # "hash" layers (layer_idx < self._num_hash_layers). The per-layer loops
        # below still record ALL layers, so the per-layer tables / CSV are
        # unaffected. nh==0 (no field) reproduces the old behavior exactly.
        nh = self._num_hash_layers
        rates = [r for li, r in rates_by_layer.items() if li >= nh]
        if rates:  # guard: empty only in the degenerate all-hash-layer case
            self._seq_token_stats.append(self._stats4(rates))
            self._seq_token_batch.append(self._pending_token_batch)
        for lidx, rate in rates_by_layer.items():
            self._seq_layer_rates.setdefault(lidx, []).append(rate)
        # Fold every timing metric present this step.
        if self._profile_timing:
            for m in self._t_metrics:
                d = self._t_step[m]
                if not d:
                    continue
                # CHANGE: exclude hash layers from the cross-layer reduction AND
                # the per-step total (shared filtered list).
                vals = [v for li, v in d.items() if li >= nh]
                if vals:
                    self._t_seq_stats[m].append(self._stats4(vals))
                    self._t_seq_total[m].append(sum(vals))
                for lidx, v in d.items():
                    self._t_seq_layer[m].setdefault(lidx, []).append(v)
            self._t_step = {m: {} for m in self._t_metrics}
        self._seq_token_layer_hits = {}

    def flush_sequence_cache_stats(self, finished_reqs: int = 0):
        """Close out one window into the summary accumulators. SILENT.

        Measured windows accumulate hit-rate and timing stats; on reaching
        seq_stats_num_seqs the one-shot summary prints. Idempotent.
        """
        self._close_token_stats()
        token_stats = self._seq_token_stats
        if not token_stats:
            return

        def _reset():
            self._seq_token_stats = []
            self._seq_token_batch = []
            self._seq_layer_rates = {}
            self._t_seq_stats = {m: [] for m in self._t_metrics}
            self._t_seq_total = {m: [] for m in self._t_metrics}
            self._t_seq_layer = {m: {} for m in self._t_metrics}

        if self._seq_stats_done:
            _reset()
            return
        # REMOVED (#10): the `if self._seq_warmup_remaining > 0:` warmup-discard
        # branch. It referenced self._debug_update_weights, which is never defined
        # (latent AttributeError if warmup>0), and warmup is unused (SEQ_WARMUP=0).
        n = len(token_stats)
        self._summary_seq_stats.append((
            sum(t[0] for t in token_stats) / n,
            sum(t[1] for t in token_stats) / n,
            sum(t[2] for t in token_stats) / n,
            sum(t[3] for t in token_stats) / n,
        ))
        for i in range(4):
            self._summary_step_sum[i] += sum(t[i] for t in token_stats)
        self._summary_step_cnt += n
        self._summary_gen_tokens += sum(self._seq_token_batch)
        self._summary_request_cnt += finished_reqs
        for lidx, rates in self._seq_layer_rates.items():
            self._summary_layer_rate_sum[lidx] = (
                self._summary_layer_rate_sum.get(lidx, 0.0) + sum(rates))
            self._summary_layer_rate_cnt[lidx] = (
                self._summary_layer_rate_cnt.get(lidx, 0) + len(rates))
        # Accumulate every timing metric for this window.
        if self._profile_timing:
            for m in self._t_metrics:
                cs = self._t_seq_stats[m]
                if not cs:
                    continue
                k = len(cs)
                for i in range(4):
                    self._t_sum_step[m][i] += sum(t[i] for t in cs)
                self._t_sum_win[m].append(tuple(sum(t[i] for t in cs) / k for i in range(4)))
                self._t_sum_total[m] += sum(self._t_seq_total[m])
                for lidx, vals in self._t_seq_layer[m].items():
                    self._t_sum_layer_sum[m][lidx] = self._t_sum_layer_sum[m].get(lidx, 0.0) + sum(vals)
                    self._t_sum_layer_cnt[m][lidx] = self._t_sum_layer_cnt[m].get(lidx, 0) + len(vals)
            # step count for timing means: use the compute metric's step count
            # if present, else the hit-rate step count for this window.
            self._t_sum_stepcnt += len(self._t_seq_stats["compute"]) or len(self._seq_token_stats)

        _reset()
        if (self._seq_stats_num_seqs > 0
                and len(self._summary_seq_stats) >= self._seq_stats_num_seqs):
            self._print_summary_stats()

    def _print_summary_stats(self):
        """Print the one-shot slide-ready end-of-test summary and latch done.

        Hit-rate sections always; per-metric timing sections + per-layer timing
        table only when timing data was collected (cache_profile_timing on).
        """
        if self._seq_stats_done:
            return
        seq_stats = self._summary_seq_stats
        if not seq_stats:
            return
        self._seq_stats_done = True
        nw = len(seq_stats)
        nstep = self._summary_step_cnt
        window_mean = [sum(s[i] for s in seq_stats) / nw for i in range(4)]
        step_mean = [self._summary_step_sum[i] / nstep for i in range(4)]

        layers = sorted(self._summary_layer_rate_sum)
        pl_lines: list[str] = []
        row: list[str] = []
        for lidx in layers:
            r = self._summary_layer_rate_sum[lidx] / self._summary_layer_rate_cnt[lidx]
            row.append("L%02d=%.2f" % (lidx, r))
            if len(row) == 6:
                pl_lines.append("   " + " ".join(row))
                row = []
        if row:
            pl_lines.append("   " + " ".join(row))

        meta = self._run_meta
        bar = "=" * 64
        lines = ["", bar, " EXPERT-OFFLOAD CACHE HIT-RATE SUMMARY", bar]
        if meta:
            lines += [
                " Run config",
                "   model            : %s" % meta.get("model", "?"),
                "   dtype / quant     : %s / %s" % (meta.get("dtype", "?"), meta.get("quant", "?")),
                "   parallel          : tp=%s  dp=%s" % (meta.get("tp", "?"), meta.get("dp", "?")),
                "   max_num_seqs      : %s    max_model_len: %s    eager: %s"
                    % (meta.get("max_num_seqs", "?"), meta.get("max_model_len", "?"),
                       meta.get("enforce_eager", "?")),
            ]
        lines += [
            " Offload config",
            "   device_experts    : %d / %s routed    device_layers: %d    moe_layers: %d"
                % (self.num_device_experts, self.num_total_experts,
                   self.num_device_layers, len(self.moe_layers)),
            "   top_k             : %d    offload_threshold: %d  (decode cache when batch<=%d)"
                % (self.topk, self.offload_threshold, self.offload_threshold),
            "   cache_policy      : %s"
                % ("LRC" if self.cache_policy is not None else "off (arbitrary eviction)"),
            "   prefetch          : %s"
                % ("on" if getattr(self.offload_config, "expert_prefetch_enabled", False) else "off"),
            " Workload measured (decode phase only)",
            "   requests          : %d%s"
                % (self._summary_request_cnt,
                   "" if self._summary_request_cnt > 0 else "  (finished-request hook not applied)"),
            "   flush_windows     : %d" % nw,
            "   decode_steps      : %d" % nstep,
            "   gen_tokens        : %d" % self._summary_gen_tokens,
            " Hit rate (routed experts resident / routed experts needed)",
            "   over decode_steps : avg=%.4f  median=%.4f  min=%.4f  max=%.4f" % tuple(step_mean),
            "   over windows      : avg=%.4f  median=%.4f  min=%.4f  max=%.4f" % tuple(window_mean),
        ]

        # Timing — one block per metric, then a per-layer table. Only when data
        # was collected. NOTE: uses 'mt' as the metric loop var so the 'meta'
        # dict above is not shadowed.
        if self._profile_timing and self._t_sum_stepcnt > 0 and any(self._t_sum_win[mt] for mt in self._t_metrics):
            ns = self._t_sum_stepcnt
            lines += [" Per-layer timing (ms), decode path  [eager; relative breakdown, not absolute latency]"]
            # CHANGE: iterate timing-only metrics so the fraction metrics aren't
            # printed/labelled as ms (they get their own block below).
            for mt in self._t_timing_metrics:
                if not self._t_sum_win[mt]:
                    continue
                step = [self._t_sum_step[mt][i] / ns for i in range(4)]
                nwin = len(self._t_sum_win[mt])
                win = [sum(w[i] for w in self._t_sum_win[mt]) / nwin for i in range(4)]
                tot = self._t_sum_total[mt] / ns
                lines += [
                    "   %-26s total/step=%.3f" % (self._t_label[mt] + ":", tot),
                    "     over steps  : avg=%.3f median=%.3f min=%.3f max=%.3f" % tuple(step),
                    "     over windows: avg=%.3f median=%.3f min=%.3f max=%.3f" % tuple(win),
                ]
            # per-layer means: fixed-width columns aligned under their headers.
            # CHANGE: pf_compute/pf_h2d abbreviations added; iterate timing metrics.
            abbr = {"attn": "att", "router": "rtr", "hc": "hc", "lrc": "lrc",
                    "miss_load": "mld", "compute": "cmp", "shared": "shr",
                    "pf_compute": "pfc", "pf_h2d": "pfh"}
            all_layers = sorted({li for mt in self._t_timing_metrics for li in self._t_sum_layer_sum[mt]})
            if all_layers:
                col = 9  # per-column width (right-aligned)
                header = "     %-6s" % "layer" + "".join("%*s" % (col, abbr[mt]) for mt in self._t_timing_metrics)
                lines += ["   per-layer mean (ms):", header]
                for li in all_layers:
                    cells = ""
                    for mt in self._t_timing_metrics:
                        s = self._t_sum_layer_sum[mt].get(li)
                        c = self._t_sum_layer_cnt[mt].get(li)
                        cells += ("%*.3f" % (col, s / c)) if c else ("%*s" % (col, "-"))
                    lines += ["     %-6s" % ("L%02d" % li) + cells]

            # NEW: prefetch accuracy fractions (0..1), printed separately from
            # the ms table. Same over-steps / over-windows aggregation; reuses
            # ns (total measured steps) from above.
            if any(self._t_sum_win[mt] for mt in self._t_rate_metrics):
                lines += [" Prefetch accuracy (decode path)  [eager; fraction 0..1, prefetch targets layers 1..N-1]"]
                for mt in self._t_rate_metrics:
                    if not self._t_sum_win[mt]:
                        continue
                    step = [self._t_sum_step[mt][i] / ns for i in range(4)]
                    nwin = len(self._t_sum_win[mt])
                    win = [sum(w[i] for w in self._t_sum_win[mt]) / nwin for i in range(4)]
                    lines += [
                        "   %-44s" % (self._t_label[mt] + ":"),
                        "     over steps  : avg=%.4f median=%.4f min=%.4f max=%.4f" % tuple(step),
                        "     over windows: avg=%.4f median=%.4f min=%.4f max=%.4f" % tuple(win),
                    ]
                rate_abbr = {"pred_acc": "acc", "pf_in_lrc": "lrc", "pf_useful": "use"}
                all_rlayers = sorted({li for mt in self._t_rate_metrics for li in self._t_sum_layer_sum[mt]})
                if all_rlayers:
                    col = 9
                    header = "     %-6s" % "layer" + "".join("%*s" % (col, rate_abbr[mt]) for mt in self._t_rate_metrics)
                    lines += ["   per-layer mean (fraction):", header]
                    for li in all_rlayers:
                        cells = ""
                        for mt in self._t_rate_metrics:
                            s = self._t_sum_layer_sum[mt].get(li)
                            c = self._t_sum_layer_cnt[mt].get(li)
                            cells += ("%*.4f" % (col, s / c)) if c else ("%*s" % (col, "-"))
                        lines += ["     %-6s" % ("L%02d" % li) + cells]

        lines += [" Per-layer mean hit rate"] + pl_lines
        lines += [bar, ""]
        logger.info("[EXPERT-OFFLOAD-FINAL]\n%s", "\n".join(lines))

        # Markdown-table variant.
        md = [
            "",
            "| field | value |",
            "|---|---|",
            "| model | %s |" % meta.get("model", "?"),
            "| device_experts / routed | %d / %s |"
                % (self.num_device_experts, self.num_total_experts),
            "| prefetch | %s | %s"
                % ("on" if getattr(self.offload_config, "expert_prefetch_enabled", False) else "off", getattr(self.offload_config, "expert_predictor")),
            "| max_num_seqs / top_k / threshold | %s / %d / %d |"
                % (meta.get("max_num_seqs", "?"), self.topk, self.offload_threshold),
            "| requests / decode_steps / gen_tokens | %d / %d / %d |"
                % (self._summary_request_cnt, nstep, self._summary_gen_tokens),
            "| hit_rate over steps (avg/med/min/max) | %.4f / %.4f / %.4f / %.4f |"
                % tuple(step_mean),
            "| hit_rate over windows (avg/med/min/max) | %.4f / %.4f / %.4f / %.4f |"
                % tuple(window_mean),
        ]
        if self._profile_timing and self._t_sum_stepcnt > 0:
            ns = self._t_sum_stepcnt
            # CHANGE: ms metrics use ms labels...
            for mt in self._t_timing_metrics:
                if not self._t_sum_win[mt]:
                    continue
                step = [self._t_sum_step[mt][i] / ns for i in range(4)]
                md += ["| %s ms total/step (mean) | %.3f |"
                       % (self._t_label[mt], self._t_sum_total[mt] / ns)]
                md += ["| %s ms over steps (avg/med/min/max) | %.3f / %.3f / %.3f / %.3f |"
                       % (self._t_label[mt], step[0], step[1], step[2], step[3])]
            # NEW: fraction metrics get fraction rows (no "ms").
            for mt in self._t_rate_metrics:
                if not self._t_sum_win[mt]:
                    continue
                step = [self._t_sum_step[mt][i] / ns for i in range(4)]
                md += ["| %s over steps (avg/med/min/max) | %.4f / %.4f / %.4f / %.4f |"
                       % (self._t_label[mt], step[0], step[1], step[2], step[3])]
        md += [""]
        logger.info("[EXPERT-OFFLOAD-FINAL-MD]\n%s", "\n".join(md))

        # export the per-layer aggregates to CSV (no-op unless CSV=1)
        self._dump_summary_csv()

    def _dump_summary_csv(self) -> None:
        """Export the per-layer summary aggregates to CSV, one row per MoE layer
        (layer 0 .. len(moe_layers)-1). Enabled by CSV=1; path from CSV_PATH.
        Called once at the end of _print_summary_stats.
 
        Each value is that layer's MEAN over the measured decode steps (the same
        per-layer means the printed summary tables show): hit_rate from
        _summary_layer_rate_{sum,cnt}, the timing metrics (<m>_ms) and the rate
        metrics (pred_acc/pf_in_lrc/pf_useful) from _t_sum_layer_{sum,cnt}[m]. A
        layer with no samples for a metric (e.g. the leading hash layers have no
        pf_*/pred_* data) gets a blank cell -> pandas/numpy read it as NaN. The
        common sample size (measured decode steps) is logged once below, not
        repeated as a column. Header is built from the metric tuples, so any
        metric added to _t_timing_metrics / _t_rate_metrics shows up here
        automatically.
 
        NOTE: the timing/rate columns only carry data when cache_profile_timing
        is also on; hit_rate is populated whenever cache stats were collected.
        """
        if not self._dump_csv or self._csv_written:
            return
        import csv  # stdlib, only used here.
 
        n_layers = len(self.moe_layers)
        timing = list(self._t_timing_metrics)   # attn,router,upload,compute,shared,pf_compute,pf_h2d
        rates = list(self._t_rate_metrics)       # pred_acc,pf_in_lrc,pf_useful
 
        # One column per metric: hit_rate, then each timing metric as <m>_ms,
        # then each rate metric (fraction). No per-metric count columns.
        header = ["layer", "hit_rate"]
        header += ["%s_ms" % m for m in timing]
        header += list(rates)
 
        def _mean(sum_d, cnt_d, lidx):
            # blank (-> NaN) when this layer has no samples for the metric.
            c = cnt_d.get(lidx, 0)
            return ("%.6f" % (sum_d.get(lidx, 0.0) / c)) if c else ""
 
        # Representative sample size for the log line: measured decode steps.
        n_steps = self._summary_step_cnt
 
        try:
            with open(self._csv_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                for li in range(n_layers):
                    row = [li, _mean(self._summary_layer_rate_sum,
                                     self._summary_layer_rate_cnt, li)]
                    for m in timing:
                        row.append(_mean(self._t_sum_layer_sum[m],
                                         self._t_sum_layer_cnt[m], li))
                    for m in rates:
                        row.append(_mean(self._t_sum_layer_sum[m],
                                         self._t_sum_layer_cnt[m], li))
                    w.writerow(row)
            self._csv_written = True
            logger.info("[EXPERT-OFFLOAD-CSV] wrote %d per-layer rows "
                        "(each value = mean over %d measured decode steps) -> %s",
                        n_layers, n_steps, self._csv_path)
        except Exception as e:
            logger.warning("[EXPERT-OFFLOAD-CSV] failed to write %s: %s",
                           self._csv_path, e)

    def _dump_final_stats_at_exit(self):
        """atexit backstop: print the summary at engine teardown.

        Offline benchmarks (vllm bench throughput / latency) tear the
        in-process engine down without a trailing prompt or, possibly, a
        finished-request step — so without this the accumulated stats would
        be lost. Folds the in-flight sequence (warmup accounting still
        applies) and prints if anything was measured and the summary hasn't
        already printed. Exceptions swallowed: logging may be partially
        torn down at interpreter exit.
        """
        try:
            if self._seq_stats_done:
                return
            self.flush_sequence_cache_stats()
            self._print_summary_stats()
        except Exception:
            pass


_EXPERT_OFFLOAD_MANAGER: ExpertOffloadManager = None


def maybe_init_expert_offload_manager(vllm_config: VllmConfig):
    global _EXPERT_OFFLOAD_MANAGER
    if _EXPERT_OFFLOAD_MANAGER is None:
        _EXPERT_OFFLOAD_MANAGER = ExpertOffloadManager(vllm_config)


def has_expert_offload_manager():
    return _EXPERT_OFFLOAD_MANAGER is not None


def get_expert_offload_manager():
    assert _EXPERT_OFFLOAD_MANAGER is not None, (
        "Expert Offload Manager is not initialized"
    )
    return _EXPERT_OFFLOAD_MANAGER