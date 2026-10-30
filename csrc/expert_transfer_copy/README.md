# Device-owned expert byte copy (Ascend950PR)

The default AccOffloadSparseCopy ABI uses 2 AIV blocks and 64 KiB tiles. Source/destination pointers and lengths remain NPU tensors. No CPU expert decision callback or runtime weight conversion is introduced. The launcher submits eight fixed waves on the current stream, partitioning each field by tile index. This bounds continuous AIV occupancy while preserving a fixed graph topology and NPU-owned descriptors. Device prefetch and load-balanced expert uploads share this entry point.

Diagnostic copies use an independent unsliced kernel with the same block count and tile size and additional instrumentation fences; their timing does not represent the production wave schedule. Use the profiler for production scheduling.

Build from repository root after sourcing CANN:

    bash csrc/expert_transfer_copy/build.sh "$PWD/build/expert_transfer_copy" ascend950pr_9579

This copies both shared libraries into vllm_ascend/expert_offload/lib. CPU MoE transfer_scheduler=npu resolves this library when transfer_copy_library is omitted or empty. Explicit paths override it. Pass the target SoC explicitly: this change was validated on Ascend950PR_9579 with CANN 9.1.0. The build script default remains ascend950pr_9599.

There is no core-count, tile-size or pipeline-mode selector in the production copy path. Native MXFP4 conversion remains separate. Small tests cover full expert bytes, dynamic graph source/slot changes, inactive transfers, tail guards and diagnostic ring replay. Microbenchmark improvement is not a full-model TPOT claim.
