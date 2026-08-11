### memfabric_hybrid 使用代码
https://github.com/ader47/vllm-ascend/blob/e4f2dd3e663c4d44b3c770f59424b83252df8608/vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py#L5


### 任务进展
已完成：
1、A5  流程打通 --完成
2、w4a8
3、MTP、dflash； 一个开关；mtp2；
4、H2D优化 -- tpot 提升20ms
	1、mte   高--A5不可用，使用copy_添加non_blocking去除传输空泡
5、调整单批次token：--max-num-batched-tokens 8192 ，会使kvcache变小
	prefill分析，
	1、TTFT的时长，与分的chunk有关，一个chunk 8k，传输时间约3s；
	2、计算与传输掩盖情况；传输时长固定；每个chunk3s；每层传输256个专家，每个专家250us，每层总时间（68ms=64+4）传输时间64ms，传输启动时间4ms；总计算时间22ms；
	3、  8k的chunk时间为3.0s，每层传输64ms，传输启动时间4ms，计算时间22ms；  64k启动时间：64/8*3s=24s
		16k的chunk时间为4.5s，每层传输64ms，传输启动时间18ms，计算时间54ms； 64k启动时间：64/16*4.5s=18s
		24k的chunk时间为3.3s，每层传输64ms，传输启动时间4ms，计算时间70ms；  64k启动时间：64/24*3.3s=8.8s
		32k的chunk时间为4.5s，每层传输64ms，传输启动时间5ms，计算时间95ms；  64k启动时间：64/32*4.5s=9s
6、通算掩盖；--完成优化；
	0、放在fused_moe前面进行传输  -- done w4a8_mxfp4  / undo w4a8/w8a8
	1、要让传进来的数字有效；
	2、测试时间； 传一个：xx；传两个：xx；
	3、启动开销；--单次开销：250um，没有办法再减小； done
7、prefill优化 ，调整chunk为32k，保证256k输入，prefix cache时，输入能够单chunk完成，ttft：6.7s
8、换成c8的精度；自动生效，ds v4走dsa，sfa c8 走 sfa；
9、、前三层专家加载  -- 前三层使用hash路由，后面40层使用学习型路由，保证前三层不会错误加载
10、index cache -- 加速推理 attention 耗时  -- 可以直接叠；GLM5已经适配； ttft提升1s，5.6s
--hf-overrides '{"use_index_cache": true, "index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}' \
精度与速度的黄金平衡：对于 43 层的 Flash 模型，频率设为 2 可以在完全不伤及精度的情况下，将文本预填（Prefill）阶段的延迟砍掉将近一半；如果追求极限速度且不在乎极微小的精度波动，也可以尝试设为 4。
11、进行20条数据集测试；ttft=7s，tpot=30ms，吞吐=30; --done
	疑问：单独采集prefiler的时候看着也就4s，不知道为什么aisbench测性能的时候居然7s左右；
12、动态：每层专家命中情况统计；（前提，关掉prefetch，关掉mtp），针对coding和general两种数据集，按层对不同命中率所需专家统计；（gsm8k）
	1、获得统计数据；
	2、提前将静态的专家加载；
13、测精度
	1、开启index cache的情况下，输出5k；gsm8k：94.6；GPQA：68；
	2、不开启index cache，输出 128k；gsm8k：；GPQA：；



待完成
2、测试完成后，按照 静态+动态 进行开发和测试；；32步是好策略吗？--和孟博讨论下； -- 暂时用32
按照不同数据集进行--初始化  -- 先用 gsm8k打样
5、kv offload 卸载
6、测精度
7、rebase代码
8、走读代码逻辑
9、三个算子入参
10、和agent hint，在线热点专家更新；
11、去掉命中率低谷，提升命中率
12、测试不同数据集，提升泛化性

优先级-高，开关适配即可
10、index cache -- 加速推理 attention 耗时  -- 可以直接叠；GLM5已经适配； ttft提升1s，5.6s
9、prefix cache加速优化 - 
0.22.1里开关 VLLM_USE_FASTOKENS=1
0.21.0里--tokenizer-mode fastokens vLLM 里也没有 fastokens 这个 tokenizer mode（合法值是 auto/slow/mistral/deepseek_v4 这类）