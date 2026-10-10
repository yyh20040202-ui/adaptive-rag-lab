# Model card

本卡由实测日志与本地公开模型下载记录生成。实际训练、上游预训练和本地推理分别说明。

| 模型 | 来源 / revision | 用途 | 许可 |
|---|---|---|---|
| MiniLM encoder | `sentence-transformers/all-MiniLM-L6-v2` · `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | 384维mean-pool / L2检索 | Apache-2.0 |
| Retriever LoRA | 项目训练，Q/V层r8/alpha16/dropout0.05 | 73,728参数，192-token输入 | Apache-2.0 |
| V1 neural ranker | 项目训练，771→64→16→1 MLP | 池化交互特征重排，不是token cross-encoder | MIT |
| V1 router | 项目训练，7→32→16→3 MLP | BM25可得特征预测预算路径 | MIT |
| V2 cross-encoder L6 | `cross-encoder/ms-marco-MiniLM-L6-v2` · `233902d25c440f23af6f7d6e94d2946bac0bee0a` | MS MARCO预训练成对相关性；本项目未训练该模型 | Apache-2.0 |
| V2 cross-encoder L12 | `cross-encoder/ms-marco-MiniLM-L12-v2` · `7b0235231ca2674cb8ca8f022859a6eba2b1c968` | MS MARCO预训练成对相关性；本项目未训练该模型 | Apache-2.0 |
| V2 BGE v2-m3 | `gpustack/bge-reranker-v2-m3-GGUF` · `3093af03b1a635e67b084b1d8c03c5f5e020fd05` | BAAI多语种预训练重排器，GPUStack Q8_0转换；本项目未训练 | Apache-2.0 |
| Qwen GGUF / qwen | `ggml-org/Qwen3-0.6B-GGUF` · `b5f37287796e5be0ea3dab2e7430873fb3f73e49` | `Qwen3-0.6B-Q4_0.gguf`本地受约束生成；没有Qwen SFT/LoRA | Apache-2.0 |
| Qwen GGUF / qwen17 | `ggml-org/Qwen3-1.7B-GGUF` · `daeb8e2d528a760970442092f6bf1e55c3b659eb` | `Qwen3-1.7B-Q4_K_M.gguf`本地受约束生成；没有Qwen SFT/LoRA | Apache-2.0 |
| Qwen GGUF / qwen4 | `lmstudio-community/Qwen3-4B-Instruct-2507-GGUF` · `4edb920b6f14e3b9284d4502a6485103d72cde05` | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`本地受约束生成；没有Qwen SFT/LoRA | Apache-2.0 |

## 实际训练与验证

LoRA底座冻结，PyTorch CPU, float32, 4 threads，训练4轮 / 324步，batch8，验证损失选第1轮。训练集已知正例从BM25难负例中剔除，batch中的已知假负例屏蔽；未标注相关文献仍可能被误当负例。

训练数据与模型记录见 `reports/training.json`、`reports/split.json`、`reports/retrieval.json`。单种子与小数据集结果不代表跨领域鲁棒性。

分组验证集选定 `gpustack/bge-reranker-v2-m3-GGUF`（revision `3093af03b1a635e67b084b1d8c03c5f5e020fd05`），重排深度 50，α=0.75，成对输入最多 384 tokens，实际推理 `GGUF-Q8_0-Vulkan`；验证 nDCG@10=0.7662。

V2公开cross-encoder对query/document联合token编码输出相关性原始logit。MiniLM L6/L12以FP32 safetensors下载，运行时仅Linear层做CPU动态INT8量化；BGE以Q8_0 GGUF通过Vulkan推理。没有量化感知训练或FP32/量化质量等价假设。最大384 tokens，batch≤8、CPU默认2线程；仅在163条分组验证查询上选三模型、候选深度和融合比例。实际选定backend/inference见 `checkpoints/quality_config.json`，完整45项网格见 `reports/quality_selection.json`。

## 模型文件与来源

下载记录保存revision、URL、SHA256与字节数；权重仅使用safetensors/GGUF，完整基础权重不上传仓库。0.6B/1.7B量化权重来自ggml-org，4B Instruct量化权重来自LM Studio community，上游基础模型均来自Qwen；llama.cpp使用固定b11517 Windows Vulkan运行时。

- `cross_encoder_l6/model.safetensors` SHA256：`821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`。
- `cross_encoder_l12/model.safetensors` SHA256：`1ed84b90cdf3518f76ec9bb93a16f97887eea7c4e7ee5dfb03cc297e394fbbc3`。
- `bge_reranker/bge-reranker-v2-m3-Q8_0.gguf` SHA256：`a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3`。
- `qwen/Qwen3-0.6B-Q4_0.gguf` SHA256：`da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4`。
- `qwen17/Qwen3-1.7B-Q4_K_M.gguf` SHA256：`d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5`。
- `qwen4/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` SHA256：`8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc`。

上游模型卡：[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)、[cross-encoder L6](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)、[L12](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2)、[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)、[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)。[BGE基础模型](https://huggingface.co/BAAI/bge-reranker-v2-m3)、[GPUStack量化转换](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF)。感谢Sentence Transformers/MiniLM、BAAI、GPUStack、Qwen与ggml-org贡献者。

## 推理、引用和限制

生成模型只进行本地推理，型号由 `LLAMA_MODEL_LABEL` 显式配置并写入报告。V2模型选择原文跨度ID，客户端恢复逐字quote并核对source_id及原文字符偏移；JSON互斥回答/拒答分支。缺少可引用证据、模型不可用或审计失败时拒答。

SciFact额外使用保守的字面标识守卫：数字复合词、英文缩写或混合大小写实体缺失时拒答。它可能错拒只有同义表达的证据，不是语义蕴含模型。支持标签后只重复声明的回答也会被拒绝。

GGUF服务RAM提示缓存与上下文检查点均关闭（--cache-ram 0 / --ctx-checkpoints 0）。历史内存守卫事件完整保留；重复请求后的资源采样见运行报告。

同一组固定真实文献声明：3 / 6 条输出通过机械来源/引用审计。

实际生成配置：Qwen3-4B-Instruct-2507 Q4_K_M; no thinking; temperature=0。

2 / 6 条的打包上下文包含 qrels 标注相关来源；这仍不是答案正确标签。

拒答/审计记录：`missing_query_anchor`=2，`model_abstained`=1。

约束原文跨度选择会提高引用形式的可控性，但合法引用仍可能不支持结论；这些数量不能换算为 QA accuracy、语义支持率或幻觉率。

逐条回答、早期因果混淆和更换模型的诊断说明见 `docs/generation_review.md`；该复查使用同一已观察样例，并非独立标注评测。

- 原文引用存在不等于结论受支持；跨度约束也不提供语义蕴含证明。
- V2复用了已查看过的BEIR测试查询；配置由验证集选定，但不能宣称全新盲测。
- 上游MiniLM/MS MARCO/BGE/Qwen训练数据污染未知；英语生物医学结果不能外推中文企业资料。
- 本项目未进行独立QA金标准评估，不宣称幻觉为零、QA准确率、SOTA或生产级事实性。
- 路由成本0/1/1.25是代理单位，不是实测token费用、焦耳或同比能耗收益。
- CPU两线程、batch八、可用内存2GiB与RSS3GiB守卫是批前采样，不是操作系统硬限额。生成器建议通过 `python -m adaptive_rag.launch_generator` 启动：单并发、2秒RAM看门狗，只停止自身子进程。
