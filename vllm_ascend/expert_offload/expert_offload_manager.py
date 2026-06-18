"""Expert Offload Manager — manages CPU-side expert weights and NPU paging."""

import queue
import threading
import time

import atexit
import statistics

import torch
import torch_npu
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


_SUBSCRIBED_COMPUTE_STREAMS = set()
def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS


class ExpertOffloadManager:
    """Singleton manager for expert weight offloading.

    Stores all expert weights on CPU and pages the needed experts to NPU
    during forward based on routing topk_ids.
    """

    _instance: "ExpertOffloadManager | None" = None

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

        # Temporary storage for weights loaded before create_weights()
        self._pending_weights: dict = {}

        # CPU buffers for quantized model scale/offset parameters.
        # Keyed by attr_name (e.g. "w13_weight_scale", "w2_weight_offset").
        # Each value is a list of layers, each layer is a list of expert tensors.
        self.scale_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}
        self.offset_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}

        # Temporary storage for scale/offset weights loaded before
        # maybe_create_scale_buffers runs.
        self._pending_scales: dict[tuple, dict[str, torch.Tensor]] = {}

        self.num_device_layers = self.offload_config.num_device_layers
        self.num_total_experts = None  # set in create_weights
        self.cache_policy: LRCExpertCachePolicy | None = None
        self.cache_requests: list[int] = []
        self.cache_hits: list[int] = []
        self.cache_misses: list[int] = []
        self.cache_calls: list[int] = []
        self.last_hit_experts: list[list[int]] = []
        self.last_miss_experts: list[list[int]] = []
        self._debug_update_weights = self.offload_config.cache_debug_log_updates

        # End-of-test cache hit-rate summary (decode path only)
        self._seq_token_layer_hits: dict[int, float] = {}     # layer_idx -> hit rate, current step
        self._seq_token_stats: list[tuple[float, float, float, float]] = []  # per step of current seq
        self._seq_token_batch: list[int] = []                 # per step of current seq: batch size
        self._seq_layer_rates: dict[int, list[float]] = {}    # layer_idx -> per-step rates, current seq
        self._pending_token_batch: int = 0                    # batch size of the step in flight
        self._seq_stats_warmup_seqs = self.offload_config.seq_stats_warmup_seqs
        self._seq_stats_num_seqs = self.offload_config.seq_stats_num_seqs
        self._seq_warmup_remaining = self._seq_stats_warmup_seqs
        self._summary_seq_stats: list[tuple[float, float, float, float]] = []  # per measured window
        self._summary_step_sum: list[float] = [0.0, 0.0, 0.0, 0.0]
        self._summary_step_cnt: int = 0
        self._summary_gen_tokens: int = 0
        self._summary_request_cnt: int = 0
        self._summary_layer_rate_sum: dict[int, float] = {}
        self._summary_layer_rate_cnt: dict[int, int] = {}
        self._seq_stats_done: bool = False

        # per-layer timing (decode path, profiling only)
        self._profile_timing = self.offload_config.cache_profile_timing
        self._seq_token_layer_compute: dict[int, float] = {}   # current step: layer -> compute ms
        self._seq_token_layer_copy: dict[int, float] = {}      # current step: layer -> upload ms
        self._seq_token_compute_stats: list[tuple] = []        # per step of current seq
        self._seq_token_copy_stats: list[tuple] = []
        self._seq_token_compute_total: list[float] = []        # per step: sum across layers
        self._seq_token_copy_total: list[float] = []
        self._seq_layer_compute: dict[int, list[float]] = {}   # current seq: layer -> per-step ms
        self._seq_layer_copy: dict[int, list[float]] = {}
        self._summary_compute_step_sum: list[float] = [0.0, 0.0, 0.0, 0.0]
        self._summary_copy_step_sum: list[float] = [0.0, 0.0, 0.0, 0.0]
        self._summary_timing_step_cnt: int = 0                 # measured steps with timing
        self._summary_compute_total_sum: float = 0.0           # sum over steps of per-step total
        self._summary_copy_total_sum: float = 0.0
        self._summary_compute_win: list[tuple] = []            # per measured window
        self._summary_copy_win: list[tuple] = []
        self._summary_layer_compute_sum: dict[int, float] = {}
        self._summary_layer_compute_cnt: dict[int, int] = {}
        self._summary_layer_copy_sum: dict[int, float] = {}
        self._summary_layer_copy_cnt: dict[int, int] = {}

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

        # Threaded prefetch: daemon thread processes prefetch requests
        # so the main forward-pass thread is never blocked by H2D copies.
        self._prefetch_queue: queue.Queue = queue.Queue()
        self._prefetch_thread_ready: threading.Event = threading.Event()
        self._prefetch_state_lock = threading.Lock()
        self._prefetch_layer_done: dict[int, threading.Event] = {}
        self._prefetch_layer_npu_event: dict[int, torch_npu.npu.Event] = {}
        self._prefetch_thread: threading.Thread | None = None
        self._npu_device: torch.device | None = None

    # ------------------------------------------------------------------ #
    #  Lifecycle: called from NPUModelRunner during model loading         #
    # ------------------------------------------------------------------ #

    def create_weights(
        self,
        num_moe_layers: int,
        num_total_experts: int,
        w13_up_dim: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
    ):
        """Allocate CPU buffers for all MoE layers."""
        for _ in range(num_moe_layers):
            w13_list = [
                torch.empty(hidden_size, w13_up_dim, dtype=params_dtype, device="cpu", pin_memory=True)
                for _ in range(num_total_experts)
            ]
            w2_list = [
                torch.empty(intermediate_size_per_partition, hidden_size,
                            dtype=params_dtype, device="cpu", pin_memory=True)
                for _ in range(num_total_experts)
            ]
            self.w13_weights_cpu.append(w13_list)
            self.w2_weights_cpu.append(w2_list)
        self._drain_pending_weights()
        self.process_weights_after_loading()

        self.num_total_experts = num_total_experts
        if self.offload_config.cache_policy_enabled:
            self.cache_requests = [0 for _ in range(num_moe_layers)]
            self.cache_hits = [0 for _ in range(num_moe_layers)]
            self.cache_misses = [0 for _ in range(num_moe_layers)]
            self.cache_calls = [0 for _ in range(num_moe_layers)]
            self.last_hit_experts = [[] for _ in range(num_moe_layers)]
            self.last_miss_experts = [[] for _ in range(num_moe_layers)]
            self.cache_policy = LRCExpertCachePolicy(
                num_layers=num_moe_layers,
                num_experts=num_total_experts,
                cache_size=self.num_device_experts,
                topk=self.topk,
                recent_window=self.offload_config.cache_recent_window,
                ema_beta=self.offload_config.cache_ema_beta,
                recent_weight=self.offload_config.cache_recent_weight,
                ema_weight=self.offload_config.cache_ema_weight,
                router_weight=self.offload_config.cache_router_weight,
                age_weight=self.offload_config.cache_age_weight,
            )

        # update weights related buffers
        self.topk_ids_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.topk_weights_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )
        self.log2phy_h = torch.zeros(num_total_experts, dtype=torch.int32, device='cpu', pin_memory=True)
        self.log2phy_np = self.log2phy_h.numpy()

    def process_weights_after_loading(self):
        first_w13 = self.w13_weights_cpu[0][0]
        first_w2 = self.w2_weights_cpu[0][0]
        self.w13_expert_size_bytes = first_w13.nelement() * first_w13.element_size()
        self.w2_expert_size_bytes = first_w2.nelement() * first_w2.element_size()
        if first_w13.dtype != torch.int8:
            return
        # for w8a8, npu weight tensor is cast to NZ format,
        # so we also store NZ format weight in weights_cpu,
        # and copy_ tensor's underlying storage instead of tensor itself
        # to avoid implicit format conversion during h2d.
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

    def register_moe_layer(self, layer):
        self.moe_layers.append(layer)

    def register_gate_weights(self, model):
        """Store fp32 CPU copies of gate.weight for each MoE layer.

        Called from _register_offload_layers() after all MoE layers are
        registered.  The gate weights are used by predict_next_layer_experts()
        to predict which experts the next layer will need.
        """
        from vllm_ascend.models.deepseek_v4 import DeepseekV4MoE
        moe_wrappers = [m for m in model.modules()
                        if isinstance(m, DeepseekV4MoE)]
        for wrapper in moe_wrappers:
            gate_cpu = wrapper.gate.weight.data.cpu().float().clone()
            self._gate_weights_cpu.append(gate_cpu)
        logger.info("[PREFETCH] registered gate weights for %d MoE layers",
                    len(self._gate_weights_cpu))

    def load_w13(self, layer_moe_idx: int, expert_id: int,
                 loaded_weight: torch.Tensor, shard_id: str):
        """Store w1/w3 shard to CPU buffer (with transpose to post format)."""
        if not self.w13_weights_cpu:
            key = (layer_moe_idx, expert_id)
            self._pending_weights.setdefault(key, {})[f"w13_{shard_id}"] = \
                loaded_weight.cpu().clone()
            return
        cpu = self.w13_weights_cpu[layer_moe_idx][expert_id]
        intermed = cpu.shape[1] // 2
        w = loaded_weight.cpu()
        if shard_id == "w1":
            cpu[:, :intermed].copy_(w.t())
        elif shard_id == "w3":
            cpu[:, intermed: intermed + w.shape[0]].copy_(w.t())

    def load_w2(self, layer_moe_idx: int, expert_id: int,
                loaded_weight: torch.Tensor):
        """Store w2 weight to CPU buffer (with transpose to post format)."""
        if not self.w2_weights_cpu:
            key = (layer_moe_idx, expert_id)
            self._pending_weights.setdefault(key, {})["w2"] = \
                loaded_weight.cpu().clone()
            return
        self.w2_weights_cpu[layer_moe_idx][expert_id].copy_(loaded_weight.cpu().t())

    # ------------------------------------------------------------------ #
    #  Scale / offset helpers (quantized models only)                     #
    # ------------------------------------------------------------------ #

    def _add_pending_scale(self, layer_moe_idx: int, expert_id: int,
                           attr_name: str, shard_id: str,
                           loaded_weight: torch.Tensor):
        """Store a scale/offset weight that arrived before CPU buffers exist."""
        key = (layer_moe_idx, expert_id)
        sub_key = f"{attr_name}_{shard_id}"
        self._pending_scales.setdefault(key, {})[sub_key] = \
            loaded_weight.cpu().clone()

    def maybe_create_scale_buffers(self, layer, layer_moe_idx: int):
        """Inspect layer for scale/offset params and allocate CPU buffers.

        Called from _register_offload_layers AFTER process_weights_after_loading
        has transformed device tensor shapes, so we detect the final per-expert
        shape from the device tensor.
        """
        attr_names = [
            ("scale_cpu_buffers", "w13_weight_scale"),
            ("scale_cpu_buffers", "w2_weight_scale"),
            ("offset_cpu_buffers", "w13_weight_offset"),
            ("offset_cpu_buffers", "w2_weight_offset"),
        ]
        created_any = False
        global_num_experts = len(self.w13_weights_cpu[layer_moe_idx])

        for buffer_dict_name, attr_name in attr_names:
            if not hasattr(layer, attr_name):
                continue
            dev_tensor = getattr(layer, attr_name)
            per_expert_shape = dev_tensor.shape[1:]
            dtype = dev_tensor.dtype
            buffer_dict: dict = getattr(self, buffer_dict_name)
            if attr_name not in buffer_dict:
                buffer_dict[attr_name] = []
            buffers = buffer_dict[attr_name]
            while len(buffers) <= layer_moe_idx:
                buffers.append([])
            for _ in range(global_num_experts):
                buffers[layer_moe_idx].append(
                    torch.empty(per_expert_shape, dtype=dtype, device="cpu", pin_memory=True))
            created_any = True

        if created_any:
            self._drain_pending_scales()

    def _drain_pending_scales(self):
        """Drain _pending_scales into CPU buffers, assembling w1/w3 shards.

        Only removes entries that were successfully copied to CPU buffers.
        Entries for layers whose buffers haven't been created yet are left
        in _pending_scales for the next call.
        """
        if not self._pending_scales:
            return
        processed_keys: list[tuple] = []
        for (layer_idx, eid), items in self._pending_scales.items():
            if layer_idx >= len(self.w13_weights_cpu):
                continue
            if eid >= len(self.w13_weights_cpu[layer_idx]):
                continue
            # Group shards by attr_name
            attr_shards: dict[str, dict[str, torch.Tensor]] = {}
            for sub_key, w in items.items():
                # sub_key format: "{attr_name}_{shard_id}"
                # attr_name may contain underscores (e.g. "w13_weight_scale")
                # shard_id is always "w1", "w2", or "w3" (no underscores)
                parts = sub_key.rsplit("_", 1)
                if len(parts) == 2 and parts[1] in ("w1", "w2", "w3"):
                    attr_name, shard = parts[0], parts[1]
                else:
                    attr_name, shard = parts[0], parts[1] if len(parts) > 1 else ""
                attr_shards.setdefault(attr_name, {})[shard] = w

            copied_any = False
            for attr_name, shards in attr_shards.items():
                target_dict = None
                if "scale" in attr_name:
                    target_dict = self.scale_cpu_buffers
                elif "offset" in attr_name:
                    target_dict = self.offset_cpu_buffers
                if target_dict is None or attr_name not in target_dict:
                    continue
                buffers = target_dict[attr_name]
                if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                    continue
                target = buffers[layer_idx][eid]

                if attr_name.startswith("w13_"):
                    # w13 scale/offset: assemble w1 + w3 shards along dim 0
                    if "w1" in shards and "w3" in shards:
                        assembled = torch.cat(
                            [shards["w1"].cpu(), shards["w3"].cpu()], dim=0)
                        # squeeze trailing dim-1 if present (W8A8_DYNAMIC)
                        assembled = assembled.reshape(target.shape)
                        target.copy_(assembled)
                        copied_any = True
                elif attr_name.startswith("w2_"):
                    # w2 scale/offset: single shard
                    if "w2" in shards:
                        w_cpu = shards["w2"]
                        if w_cpu.device.type != "cpu":
                            w_cpu = w_cpu.cpu()
                        w_cpu = w_cpu.reshape(target.shape)
                        target.copy_(w_cpu)
                        copied_any = True
            if copied_any:
                processed_keys.append((layer_idx, eid))
        # Only remove successfully processed entries
        for key in processed_keys:
            del self._pending_scales[key]

    def init_device_experts(self):
        """Refresh derived fp32 scale after weight loading.

        Device experts are already loaded by the weight loader and
        process_weights_after_loading. Only refresh w13_weight_scale_fp32.
        """
        for i, layer in enumerate(self.moe_layers):
            ndev = min(self.num_device_experts, layer.w13_weight.shape[0])
            if hasattr(layer, 'w13_weight_scale_fp32'):
                for j in range(ndev):
                    layer.w13_weight_scale_fp32[j].copy_(
                        layer.w13_weight_scale.data[j].to(torch.float32))

    def create_prefill_pool(self):
        """Allocate prefill pool tensors on NPU with full expert count.

        Called from _register_offload_layers after decode buffers are set up.
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
        logger.warning("[PREFILL_POOL] allocated %d layers × %d experts, "
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

        # import logging
        # _dbg = logging.getLogger(__name__)
        # _dbg.warning("[PREFILL_LOAD] layer=%d pool_slot=%d ntotal=%d is_w8a8=%s",
        #              layer_idx, pool_slot, ntotal, is_w8a8)

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

        # Wait for any pending threaded prefetch for this layer to complete
        layer_done = self._prefetch_layer_done.get(layer_idx)
        if layer_done is not None:
            layer_done.wait()           # Block until prefetch thread finishes
            layer_done.clear()
            del self._prefetch_layer_done[layer_idx]
            with self._prefetch_state_lock:
                npu_event = self._prefetch_layer_npu_event.pop(layer_idx,
                                                                None)
            if npu_event is not None:
                npu_event.synchronize()  # Block until NPU DMA copies complete

        topk_ids_h = self.topk_ids_h[:num_tokens]
        topk_weights_h = None
        if (self.cache_policy is not None and topk_weights is not None and not _EXTRA_CTX.capturing
                and self.offload_config.cache_router_weight != 0):
            topk_weights_h = self.topk_weights_h[:num_tokens]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32), non_blocking=False)
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
                needed = self.cache_policy.observe(
                    layer_idx,
                    topk_ids_h.tolist(),
                    router_scores=router_scores,
                )
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
                self._record_cache_stats(layer_idx, already_there, need_to_load, needed, on_device, topk_ids_h.shape[0])
            reusable_slots = [s for s, e in slot_owner.items()
                            if e not in needed]          # slots to recycle

            if self.cache_policy is not None and self._debug_update_weights:
                import logging
                _dbg = logging.getLogger(__name__)
                # _dbg.warning(
                #     "[UPDATE-W] l=%d call=%d topk_shape=%s |needed|=%d |on_dev|=%d "
                #     "|to_load|=%d reusable=%d needed=%s",
                #     layer_idx, self.cache_calls[layer_idx], tuple(topk_ids_h.shape),
                #     len(needed), len(on_device),
                #     len(need_to_load), len(reusable_slots),
                #     sorted(needed)[:30],
                # )
                # _dbg.warning("[UPDATE-W] l=%d cache_hit=%s cache_miss=%s hit_rate=%.2f",
                #              layer_idx, sorted(already_there), sorted(need_to_load),len(already_there) / 6)
                if need_to_load and len(need_to_load) > len(reusable_slots):
                    _dbg.warning("[UPDATE-W] l=%d SHORTFALL: need %d load but only %d slots, to_load=%s",
                                layer_idx, len(need_to_load), len(reusable_slots),
                                sorted(need_to_load)[:20])

            dev = layer.w13_weight.device

            # start the upload timer just before the copy loop
            _time_upload = self._profile_timing and self.cache_policy is not None
            _tc0 = time.perf_counter() if _time_upload else 0.0

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
                    import logging
                    logging.getLogger(__name__).warning(
                        "[UPDATE-W] l=%d NO SLOTS: %d experts could not be loaded, missed=%s",
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

            if _time_upload:
                copy_ms = (time.perf_counter() - _tc0) * 1000.0 if n_copies else 0.0
                self._seq_token_layer_copy[layer_idx] = copy_ms

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
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return None  # last layer — nothing to prefetch

        if next_idx >= len(self._gate_weights_cpu):
            return None
        gate_w = self._gate_weights_cpu[next_idx]
        if gate_w is None:
            return None

        # Move hidden_states to CPU for prediction (tiny in decode: 1-8 tokens)
        hs_cpu = hidden_states.float().cpu()
        router_logits = F.linear(hs_cpu, gate_w)  # [num_tokens, n_experts]

        # Simplified routing: softmax + topk (approximation)
        probs = router_logits.softmax(dim=-1)
        _, topk_ids = probs.topk(self.topk, dim=-1)
        return set(topk_ids.flatten().tolist())

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

            # Predict which experts the next layer needs
            predicted = self.predict_next_layer_experts(layer_idx,
                                                        hidden_states)
            if predicted is None or next_idx >= len(self.moe_layers):
                # No prediction possible — signal done immediately
                done = self._prefetch_layer_done.get(next_idx)
                if done is not None:
                    done.set()
                continue

            # Execute the prefetch (H2D copies)
            completion_event = self._do_prefetch(next_idx, predicted)

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
    ) -> torch_npu.npu.Event | None:
        """Load predicted experts for layer next_idx from CPU to NPU.

        Runs on the prefetch thread using self._prefetch_stream.
        Returns an NPU Event that signals when all copies are complete,
        or None if no copies were needed.
        """
        next_layer = self.moe_layers[next_idx]

        # Snapshot next layer's current log2phy to CPU
        log2phy_np = next_layer.log2phy.cpu().numpy().copy()

        # Determine which predicted experts are not already on device
        slot_owner: dict[int, int] = {}
        for eid, slot in enumerate(log2phy_np):
            if slot >= 0:
                slot_owner[slot] = eid
        on_device = set(slot_owner.values())
        need_to_load = predicted_experts - on_device

        if not need_to_load:
            return None

        # Protected set: experts we must not evict
        protected = set(predicted_experts)
        reusable_slots = [s for s, e in slot_owner.items()
                          if e not in protected]
        if not reusable_slots:
            return None  # all resident experts are protected — skip

        with torch_npu.npu.stream(self._prefetch_stream):
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
                    logger.warning(
                        "[PREFETCH] l=%d NO SLOTS: %d experts could not "
                        "be prefetched", next_idx - 1, next_idx,
                        len(need_to_load))
                    break

                # CPU → NPU async copy w13 weights
                next_layer.w13_weight.data.untyped_storage()[
                    slot * self.w13_expert_size_bytes
                    : (slot + 1) * self.w13_expert_size_bytes
                ].copy_(
                    self.w13_weights_cpu[next_idx][eid].untyped_storage())

                time.sleep(0.00025)  # 0.25ms delay

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

            # Write modified log2phy back to next layer's NPU tensor
            next_layer.log2phy.copy_(
                torch.from_numpy(log2phy_np).to(
                    device=next_layer.log2phy.device))

            # Record prefetch completion event
            completion_event = torch_npu.npu.Event()
            self._prefetch_stream.record_event(completion_event)
            return completion_event

    def trigger_next_layer_prefetch(self, layer,
                                    hidden_states: torch.Tensor):
        """在 GMM kernel 提交后触发下一层专家预加载。

        必须在 fused_experts() 之后调用，使 compute_event 捕获
        GMM kernel 的 NPU 工作，实现预加载与计算的真正并行。

        将预测和 H2D 拷贝提交到后台线程，主线程立即返回，
        不被 aclrtMemcpy 阻塞。

        Args:
            layer: 当前 MoE 层的 AscendFusedMoE 实例。
            hidden_states: [num_tokens, hidden_dim] NPU tensor。
        """
        if not self.offload_config.expert_prefetch_enabled:
            return
        if _EXTRA_CTX.capturing:
            return

        # Lazy-start the prefetch thread on first call
        if self._prefetch_thread is None:
            self._start_prefetch_thread()

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

        # Create per-layer threading.Event for completion signaling
        self._prefetch_layer_done[next_idx] = threading.Event()

        # Submit to background thread — returns immediately!
        self._prefetch_queue.put((layer_idx, hidden_states, compute_event))

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stats4(values: list[float]) -> tuple:
        """avg, median, min, max of a non-empty list. NEW (v6); also usable
        by the hit-rate path but left there as-is to keep this delta minimal."""
        return (sum(values) / len(values), statistics.median(values),
                min(values), max(values))

    def _record_cache_stats(
        self,
        layer_idx: int,
        hit_experts: set[int],
        miss_experts: set[int],
        needed: set[int],
        on_device: set[int],
        num_tokens: int = 1,
    ):
        self._record_seq_layer_hit(layer_idx, len(hit_experts), len(needed))
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

    def record_compute_time(self, layer, compute_ms: float):
        """Stash one (layer, decode-step) MoE compute time for the summary.

        NEW (v6): called from fused_moe.py apply after the routed
        fused_experts call, on the decode path only. Does NOT trigger a token
        close (compute is recorded after the hit/upload hook within the same
        step); the close in _record_seq_layer_hit (next step) folds it in.
        Guarded so it is a no-op unless timing + policy are on and the
        summary hasn't printed.
        """
        if (not self._profile_timing or self.cache_policy is None
                or self._seq_stats_done):
            return
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return
        self._seq_token_layer_compute[layer_idx] = compute_ms

    def _record_seq_layer_hit(self, layer_idx: int, num_hits: int,
                              num_requested: int, num_tokens: int = 1):
        """Record one (layer, decode-step) hit rate for the final summary.

        hit rate = |hit_experts| / |needed| for this layer this step. A
        repeated layer_idx means a new decode forward STEP has started (each
        step visits each MoE layer once), so close out the previous step
        first. num_tokens is the step's batch size, captured once per step
        (constant across layers) to count generated tokens at flush time.
        """
        if self._seq_stats_done or num_requested <= 0:
            return
        if layer_idx in self._seq_token_layer_hits:
            self._close_token_stats()
        self._seq_token_layer_hits[layer_idx] = num_hits / num_requested
        self._pending_token_batch = num_tokens

    def _close_token_stats(self):
        """Fold the in-flight decode step into per-step stats.

        Hit rate: avg/median/min/max across this step's layers (v4). NEW (v6):
        when timing is on, also fold per-step compute and upload 4-stats,
        per-step totals (sum across layers), and per-layer series.
        """
        rates_by_layer = self._seq_token_layer_hits
        if not rates_by_layer:
            return
        rates = list(rates_by_layer.values())
        self._seq_token_stats.append(self._stats4(rates))
        self._seq_token_batch.append(self._pending_token_batch)
        for lidx, rate in rates_by_layer.items():
            self._seq_layer_rates.setdefault(lidx, []).append(rate)
        # NEW (v6): timing fold.
        if self._profile_timing:
            comp = self._seq_token_layer_compute
            if comp:
                vals = list(comp.values())
                self._seq_token_compute_stats.append(self._stats4(vals))
                self._seq_token_compute_total.append(sum(vals))
                for lidx, v in comp.items():
                    self._seq_layer_compute.setdefault(lidx, []).append(v)
            cop = self._seq_token_layer_copy
            if cop:
                vals = list(cop.values())
                self._seq_token_copy_stats.append(self._stats4(vals))
                self._seq_token_copy_total.append(sum(vals))
                for lidx, v in cop.items():
                    self._seq_layer_copy.setdefault(lidx, []).append(v)
        self._seq_token_layer_hits = {}
        self._seq_token_layer_compute = {}
        self._seq_token_layer_copy = {}

    def flush_sequence_cache_stats(self, finished_reqs: int = 0):
        """Close out one window into the summary accumulators. SILENT.

        v4 semantics unchanged (warmup discard / measured accumulate / summary
        on reaching seq_stats_num_seqs). NEW (v6): also accumulate the timing
        per-window means, per-step sums/totals, and per-layer series.
        """
        self._close_token_stats()
        token_stats = self._seq_token_stats
        if not token_stats:
            return

        def _reset():
            self._seq_token_stats = []
            self._seq_token_batch = []
            self._seq_layer_rates = {}
            # NEW (v6): timing per-seq buffers.
            self._seq_token_compute_stats = []
            self._seq_token_copy_stats = []
            self._seq_token_compute_total = []
            self._seq_token_copy_total = []
            self._seq_layer_compute = {}
            self._seq_layer_copy = {}

        if self._seq_stats_done:
            _reset()
            return
        if self._seq_warmup_remaining > 0:
            self._seq_warmup_remaining -= 1
            if self._debug_update_weights:
                logger.info(
                    "[EXPERT-OFFLOAD-FINAL] warmup window discarded "
                    "(decode_steps=%d, warmup_remaining=%d)",
                    len(token_stats), self._seq_warmup_remaining,
                )
            _reset()
            return
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
        # NEW (v6): timing accumulation for this window.
        if self._profile_timing:
            cs = self._seq_token_compute_stats
            if cs:
                m = len(cs)
                for i in range(4):
                    self._summary_compute_step_sum[i] += sum(t[i] for t in cs)
                self._summary_compute_win.append(tuple(
                    sum(t[i] for t in cs) / m for i in range(4)))
                self._summary_compute_total_sum += sum(self._seq_token_compute_total)
                self._summary_timing_step_cnt += m
                for lidx, vals in self._seq_layer_compute.items():
                    self._summary_layer_compute_sum[lidx] = (
                        self._summary_layer_compute_sum.get(lidx, 0.0) + sum(vals))
                    self._summary_layer_compute_cnt[lidx] = (
                        self._summary_layer_compute_cnt.get(lidx, 0) + len(vals))
            cps = self._seq_token_copy_stats
            if cps:
                m2 = len(cps)
                for i in range(4):
                    self._summary_copy_step_sum[i] += sum(t[i] for t in cps)
                self._summary_copy_win.append(tuple(
                    sum(t[i] for t in cps) / m2 for i in range(4)))
                self._summary_copy_total_sum += sum(self._seq_token_copy_total)
                for lidx, vals in self._seq_layer_copy.items():
                    self._summary_layer_copy_sum[lidx] = (
                        self._summary_layer_copy_sum.get(lidx, 0.0) + sum(vals))
                    self._summary_layer_copy_cnt[lidx] = (
                        self._summary_layer_copy_cnt.get(lidx, 0) + len(vals))
        _reset()
        if (self._seq_stats_num_seqs > 0
                and len(self._summary_seq_stats) >= self._seq_stats_num_seqs):
            self._print_summary_stats()

    def _print_summary_stats(self):
        """Print the one-shot end-of-test summary (slide-ready) and latch done.

        v5 layout + NEW (v6) timing sections, shown only when timing data was
        collected. step_mean denominator for timing is the count of timed
        steps (may equal the hit-rate step count).
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

        m = self._run_meta
        bar = "=" * 64
        lines = ["", bar, " EXPERT-OFFLOAD CACHE HIT-RATE SUMMARY", bar]
        if m:
            lines += [
                " Run config",
                "   model            : %s" % m.get("model", "?"),
                "   dtype / quant     : %s / %s" % (m.get("dtype", "?"), m.get("quant", "?")),
                "   parallel          : tp=%s  dp=%s" % (m.get("tp", "?"), m.get("dp", "?")),
                "   max_num_seqs      : %s    max_model_len: %s    eager: %s"
                    % (m.get("max_num_seqs", "?"), m.get("max_model_len", "?"),
                       m.get("enforce_eager", "?")),
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

        # NEW (v6): timing sections, only if timing data was collected.
        nts = self._summary_timing_step_cnt
        if self._profile_timing and self._summary_compute_win and nts > 0:
            c_step = [self._summary_compute_step_sum[i] / nts for i in range(4)]
            u_step = [self._summary_copy_step_sum[i] / nts for i in range(4)]
            ncw = len(self._summary_compute_win)
            nuw = max(1, len(self._summary_copy_win))
            c_win = [sum(w[i] for w in self._summary_compute_win) / ncw for i in range(4)]
            u_win = [sum(w[i] for w in self._summary_copy_win) / nuw for i in range(4)] \
                if self._summary_copy_win else [0.0, 0.0, 0.0, 0.0]
            c_total = self._summary_compute_total_sum / nts
            u_total = self._summary_copy_total_sum / nts
            lines += [
                " MoE timing (ms)  [compute = routed fused_experts; upload = cache-miss H2D]",
                "   total per step    : compute=%.3f   upload=%.3f   (mean over steps, summed across layers)"
                    % (c_total, u_total),
                "   compute over steps: avg=%.3f  median=%.3f  min=%.3f  max=%.3f" % tuple(c_step),
                "   compute over wins : avg=%.3f  median=%.3f  min=%.3f  max=%.3f" % tuple(c_win),
                "   upload  over steps: avg=%.3f  median=%.3f  min=%.3f  max=%.3f" % tuple(u_step),
                "   upload  over wins : avg=%.3f  median=%.3f  min=%.3f  max=%.3f" % tuple(u_win),
            ]

        lines += [" Per-layer mean hit rate"] + pl_lines

        # NEW (v6): per-layer mean timing (compute / upload), chunked 4/row.
        if self._profile_timing and self._summary_layer_compute_sum:
            lines += [" Per-layer mean time (ms): compute / upload"]
            trow: list[str] = []
            for lidx in sorted(self._summary_layer_compute_sum):
                c = self._summary_layer_compute_sum[lidx] / self._summary_layer_compute_cnt[lidx]
                u = (self._summary_layer_copy_sum.get(lidx, 0.0)
                     / self._summary_layer_copy_cnt[lidx]) if self._summary_layer_copy_cnt.get(lidx) else 0.0
                trow.append("L%02d=%.3f/%.3f" % (lidx, c, u))
                if len(trow) == 4:
                    lines.append("   " + " ".join(trow))
                    trow = []
            if trow:
                lines.append("   " + " ".join(trow))

        lines += [bar, ""]
        logger.info("[EXPERT-OFFLOAD-FINAL]\n%s", "\n".join(lines))

        # Markdown-table variant (v5) + NEW (v6) timing rows when present.
        md = [
            "",
            "| field | value |",
            "|---|---|",
            "| model | %s |" % m.get("model", "?"),
            "| device_experts / routed | %d / %s |"
                % (self.num_device_experts, self.num_total_experts),
            "| device_layers / moe_layers | %d / %d |"
                % (self.num_device_layers, len(self.moe_layers)),
            "| max_num_seqs / top_k / threshold | %s / %d / %d |"
                % (m.get("max_num_seqs", "?"), self.topk, self.offload_threshold),
            "| requests / decode_steps / gen_tokens | %d / %d / %d |"
                % (self._summary_request_cnt, nstep, self._summary_gen_tokens),
            "| hit_rate over steps (avg/med/min/max) | %.4f / %.4f / %.4f / %.4f |"
                % tuple(step_mean),
            "| hit_rate over windows (avg/med/min/max) | %.4f / %.4f / %.4f / %.4f |"
                % tuple(window_mean),
        ]
        if self._profile_timing and self._summary_compute_win and nts > 0:
            md += [
                "| compute ms total/step (mean) | %.3f |" % c_total,
                "| upload ms total/step (mean) | %.3f |" % u_total,
                "| compute ms over steps (avg/med/min/max) | %.3f / %.3f / %.3f / %.3f |" % tuple(c_step),
                "| upload ms over steps (avg/med/min/max) | %.3f / %.3f / %.3f / %.3f |" % tuple(u_step),
            ]
        md += [""]
        logger.info("[EXPERT-OFFLOAD-FINAL-MD]\n%s", "\n".join(md))

    def _dump_final_stats_at_exit(self):
        """atexit backstop: print the summary at engine teardown.

        NEW: offline benchmarks (vllm bench throughput / latency) tear the
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

    def _drain_pending_weights(self):
        if not self._pending_weights:
            return
        for (layer_idx, eid), weights in self._pending_weights.items():
            if layer_idx >= len(self.w13_weights_cpu):
                continue
            if eid >= len(self.w13_weights_cpu[layer_idx]):
                continue
            cpu_w13 = self.w13_weights_cpu[layer_idx][eid]
            intermed = cpu_w13.shape[1] // 2
            for key, w in weights.items():
                w_cpu = w if w.device.type == "cpu" else w.cpu()
                if key.startswith("w13_"):
                    shard = key.split("_")[1]
                    if shard == "w1":
                        cpu_w13[:, :intermed].copy_(w_cpu.t())
                    elif shard == "w3":
                        cpu_w13[:, intermed: intermed + w_cpu.shape[0]].copy_(w_cpu.t())
                elif key == "w2":
                    self.w2_weights_cpu[layer_idx][eid].copy_(w_cpu.t())
        self._pending_weights.clear()


_EXPERT_OFFLOAD_MANAGER: ExpertOffloadManager = None


def maybe_init_expert_offload_manager(vllm_config: VllmConfig):
    # if no need to init offload manager:
    #     return
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
