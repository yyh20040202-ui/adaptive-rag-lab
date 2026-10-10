"""Render V2 documentation and figures from measured logs, preserving V1 files.

Plotting is imported only by the final report command. --text-only can prepare
honest documentation while V2 evaluation is pending; it never fills missing
results with predicted values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


NAMES = {
    "bm25": "BM25",
    "dense_frozen": "Frozen dense",
    "hybrid_frozen": "Frozen hybrid",
    "dense_lora": "LoRA dense",
    "hybrid_lora": "LoRA hybrid",
    "neural_rerank": "V1 pooled MLP",
    "adaptive": "V1 adaptive budget",
    "cross_encoder": "V2 token cross-encoder",
}


def read_json(path: str | Path, required: bool = False) -> dict | None:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required measured report is missing: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_table(v1: dict, quality: dict | None) -> str:
    rows = ["| 版本 / 方法 | nDCG@10 | MRR@10 | Recall@10 |",
            "|---|---:|---:|---:|"]
    values = dict(quality["metrics"] if quality is not None else v1["metrics"])
    for method, metrics in values.items():
        name = NAMES.get(method, method)
        rows.append(f"| {name} | {metrics['ndcg@10']:.4f} | {metrics['mrr@10']:.4f} | {metrics['recall@10']*100:.2f}% |")
    return "\n".join(rows)


def comparisons_text(quality: dict | None) -> str:
    if quality is None:
        return "V2 测试尚未完成；不填写新指标、提升百分比或置信区间。"
    rows = []
    value = quality["metrics"]["cross_encoder"]["ndcg@10"]
    for baseline in ["bm25", "neural_rerank"]:
        base = quality["metrics"][baseline]["ndcg@10"]
        relative = (value/base-1)*100 if base else None
        delta = value-base
        compared = quality.get("paired_ndcg_comparisons", {}).get(f"cross_encoder-vs-{baseline}")
        ci = (f"，成对查询 bootstrap 95% 区间 [{compared['ci95'][0]:.4f}, {compared['ci95'][1]:.4f}]"
              if compared else "；该差值未记录置信区间")
        relative_text = f"，相对变化 {relative:+.2f}%" if relative is not None else ""
        rows.append(f"相对 {NAMES[baseline]}：nDCG@10 绝对变化 {delta:+.4f}{relative_text}{ci}。")
    return "\n\n".join(rows)


def chosen_text(quality: dict | None, selection: dict | None) -> str:
    selected = quality.get("config") if quality else selection.get("selected") if selection else None
    if selected is None:
        return "验证选择尚未结束；候选为 MiniLM L6/L12 与 BGE v2-m3、深度20/50/100、α=0/0.25/0.5/0.75/1。"
    return (f"分组验证集选定 `{selected['model']}`（revision `{selected['revision']}`），"
            f"重排深度 {selected['rerank_depth']}，α={selected['alpha']:g}，"
            f"成对输入最多 {selected['max_length']} tokens，实际推理 `{selected['inference']}`；"
            f"验证 nDCG@10={selected['validation_metrics']['ndcg@10']:.4f}。")


def generation_text(generation: dict | None, original: dict | None) -> str:
    if generation is None:
        return ("V2 真实文献生成报告尚未产生。固定六条声明的原始 V1 结果保留在 "
                "`reports/real_corpus_generation.json`；不使用模拟制度样例填补当前结果。")
    if generation.get("status", "complete") != "complete":
        return (f"生成诊断尚未完成：仅保存 {generation.get('cases', 0)} 条进度，"
                "不能作为完整实验结果或问答正确率。")
    models = sorted({row["result"]["model"] for row in generation.get("results", []) if row.get("result", {}).get("model")})
    rows = [f"同一组固定真实文献声明：{generation['answered']} / {generation['cases']} 条输出通过机械来源/引用审计。",
            f"实际生成配置：{'；'.join(models)}。"]
    if original:
        rows.append(f"原始 V1 为 {original['answered']} / {original['cases']} 条；两版原始回答及拒答原因均保留。")
    if "relevant_source_in_packed_context" in generation:
        rows.append(f"{generation['relevant_source_in_packed_context']} / {generation['cases']} 条的打包上下文包含 qrels 标注相关来源；这仍不是答案正确标签。")
    failures = generation.get("audit_failure_counts", {})
    if failures:
        rows.append("拒答/审计记录："+"，".join(f"`{name}`={count}" for name, count in sorted(failures.items()))+"。")
    rows.append("约束原文跨度选择会提高引用形式的可控性，但合法引用仍可能不支持结论；这些数量不能换算为 QA accuracy、语义支持率或幻觉率。")
    rows.append("逐条回答、早期因果混淆和更换模型的诊断说明见 `docs/generation_review.md`；该复查使用同一已观察样例，并非独立标注评测。")
    return "\n\n".join(rows)


def resource_text(selection: dict | None) -> str:
    if not selection:
        return "V2 资源采样尚未记录。默认 CPU 2线程、batch上限8、可用内存低于2 GiB停止。"
    rows = ["| 验证候选 | 采样进程 RSS 峰值 / GiB | 采样系统可用内存最低值 / GiB | 采样耗时 / s |",
            "|---|---:|---:|---:|"]
    for variant, value in sorted(selection.get("resources", {}).items()):
        resource = value.get("resources", {})
        if resource.get("cached_scores"):
            rows.append(f"| {variant} | 缓存复用，未重新采样 | 未重新采样 | 未重新采样 |")
        else:
            peak = resource.get("peak_observed_process_rss_gib")
            available = resource.get("minimum_observed_available_gib")
            seconds = resource.get("seconds")
            rows.append(f"| {variant} | {peak:.3f} | {available:.3f} | {seconds:.2f} |"
                        if all(v is not None for v in [peak, available, seconds]) else f"| {variant} | 未记录 | 未记录 | 未记录 |")
    rows += ["", "采样仅覆盖本次选择调用：CPU候选的100篇分数可复用先前完整计算，只重新计算短batch的尾部；`full_scores_reused`标记复用情况。GGUF客户端RSS不包含独立llama.cpp服务的RSS或显存。采样在批前执行，不是OS强制分配上限。此耗时不是单次在线查询p50；系统剩余内存仍受到其他应用影响。"]
    return "\n".join(rows)


def render_analysis(v1: dict, quality: dict | None, selection: dict | None,
                    training: dict, generation: dict | None, original: dict | None) -> str:
    lines = ["# V2 检索与真实文献生成分析", "",
             "本文件由 `python -m adaptive_rag.quality_analysis` 根据本地实测 JSON 生成。缺失结果明确标记，不补写预期数字。V1 的 `benchmark.png/svg`、`analysis.md` 和原始报告保留。", "",
             f"BEIR SciFact：{v1['corpus']:,} 篇文献，{v1['train']} 训练 / {v1['validation']} 分组验证 / {v1['test']} 测试查询。", "",
             metrics_table(v1, quality), "", comparisons_text(quality), "",
             "## 选择协议", "", chosen_text(quality, selection), "",
             "LoRA hybrid 使用RRF depth100、constant60产生候选。比较MiniLM L6/L12的CPU动态INT8 Linear推理与BGE v2-m3的Q8_0 GGUF Vulkan推理，重排20/50/100篇、五个融合权重，共45组配置；未进行QAT或声称FP32质量相同。CPU动态INT8的短batch尾部单独重算，确保与线上实际批次一致。相关性原始logit与RRF分数分别min-max归一化，不解释为校准概率。验证nDCG@10选配置，同分优先浅候选、L6。BGE候选是在发现本机量化GPU推理可行后加入，加入及选参数均早于V2测试。冻结配置见 `checkpoints/quality_config.json`，完整网格见 `quality_selection.json`。", "",
             "V1 的300条测试查询已被查看；V2 是同一集合的迭代比较。配置仍只在分组验证上选择，不能将这次测试称为新盲测。置信区间仅描述给定模型下成对查询重采样，不覆盖训练种子、配置搜索、预训练污染或领域漂移。", "",
             "## 与轻量训练的关系", "",
             f"MiniLM adapter 实际训练 {training['epochs_requested']} 轮 / {training['optimizer_steps']} 步，更新 {training['trainable_parameters']:,} 个参数，验证损失选第 {training['selected_epoch']} 轮。原始后几轮出现训练损失下降、验证损失上升，未用延长训练凑高分。", "",
             "V1 ranker 是池化交互特征 MLP；V2 cross-encoder 是预训练的 query/document 联合 token 编码相关性模型，没有在本项目对它或 Qwen 进行新的 SFT。新增排序模型只重排已经召回的候选。", "",
             f"V1 预算路径减少 {(1-v1['dense_call_fraction'])*100:.2f}% 的向量检索调用，同时 nDCG@10 从 {v1['metrics']['neural_rerank']['ndcg@10']:.4f} 降至 {v1['metrics']['adaptive']['ndcg@10']:.4f}；这项成本取舍未被当成质量不变。", "",
             "## 本机资源", "", resource_text(selection), "",
             "V1 时延来自早期40条固定抽样查询的CPU热启动串行计时；V2采用新的线程/模型配置，旧时延不能直接充当新版时延。两者均不包含生成服务启动、索引构建和 Qwen 输出。没有新版查询计时字段时不推算 p50/p95。", "",
             "## 同一组六条真实声明", "", generation_text(generation, original), "",
             "模型输出使用互斥回答/拒答 JSON 分支。来源正文被切成有 ID 的连续原文跨度，模型只能选择已有 ID，程序恢复精确 quote，再核验 source_id 与原文字符位置；标题不能单独作为证明。证据预算包含实际模型上下文的跨度 JSON 开销。机械引用校验不是语义蕴含判断。", "",
             "## 边界", "",
             "- 英语生物医学检索不能外推中文企业知识库；BGE支持多语种不代表本项目验证了中文效果，用户资料的学习路由默认禁用。",
             "- BEIR SciFact test 来自原任务有标注验证集，未运行隐藏榜单。",
             "- 预训练数据污染未知，单数据集/单种子结果不支持 zero-shot 或 SOTA 声明。",
             "- 六条生成诊断与12条历史模拟回归样例都没有独立 QA 金标准。",
             "- 资源守卫是批前采样与安全停止，不能保证所有其他进程不造成瞬时峰值。", ""]
    return "\n".join(lines)


def generator_instructions() -> tuple[str, str, str]:
    stronger = read_json("models/qwen4/provenance.json")
    if stronger:
        return "qwen4", Path(stronger["file"]["path"]).name, "Qwen3-4B-Instruct-2507 Q4_K_M"
    bigger = read_json("models/qwen17/provenance.json")
    if bigger:
        filename = Path(bigger["file"]["path"]).name
        quantization = filename.removeprefix("Qwen3-1.7B-").removesuffix(".gguf")
        return "qwen17", filename, f"Qwen3-1.7B {quantization}"
    return "qwen", "Qwen3-0.6B-Q4_0.gguf", "Qwen3-0.6B Q4_0"


def render_readme(v1: dict, quality: dict | None, selection: dict | None,
                  training: dict, generation: dict | None, original: dict | None) -> str:
    folder, filename, label = generator_instructions()
    prepare_command = "python -m adaptive_rag.prepare --llama-win" if folder in {"qwen17", "qwen4"} else "python -m adaptive_rag.prepare --generator --llama-win"
    generator_download = ("python -m adaptive_rag.prepare_quality --generator-4b\n" if folder == "qwen4" else
                          "python -m adaptive_rag.prepare_quality --generator\n" if folder == "qwen17" else "")
    offload = 30 if folder == "qwen4" else 99
    load_flag = " --no-mmap" if folder == "qwen4" else ""
    selected = quality.get("config", {}) if quality else selection.get("selected", {}) if selection else {}
    gpu_reranker = selected.get("backend") == "llama_vulkan"
    reranker_download = "python -m adaptive_rag.prepare_quality --gpu-reranker\n" if gpu_reranker else ""
    reranker_start = ("额外终端先启动选定的GPU重排服务，生成服务和重排服务可同时驻留，重任务按请求串行执行：\n\n```powershell\npython -m adaptive_rag.launch_generator --model-directory models/bge_reranker --reranker\n```\n\n" if gpu_reranker else "")
    if quality:
        headline = (f"验证冻结后的 V2 cross-encoder 在同一300条测试查询上取得 "
                    f"**nDCG@10 {quality['metrics']['cross_encoder']['ndcg@10']:.4f}**、"
                    f"**Recall@10 {quality['metrics']['cross_encoder']['recall@10']*100:.2f}%**。")
        chart = "![V1/V2 measured retrieval quality and paired differences](reports/quality_benchmark.png)"
    else:
        headline = "**V2 检索评测尚未完成**；下面只列已实测的 V1 数字。新增模型不预设提升。"
        chart = "![Archived V1 retrieval benchmark](reports/benchmark.png)"
    return f'''# Adaptive RAG Lab

**真实文献检索、模型微调、cross-encoder 重排和原文引用核验的本地 RAG 实验台。**

[![offline-checks](https://github.com/yyh20040202-ui/adaptive-rag-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/yyh20040202-ui/adaptive-rag-lab/actions/workflows/ci.yml)

在 BEIR SciFact 的 {v1['corpus']:,} 篇真实摘要上比较稀疏、向量、LoRA、轻量重排、预算路由与真正的 token cross-encoder；同一模型配置接入 FastAPI 问答。项目保存训练权重、逐查询结果、原始生成和失败原因，而不是只展示一次 API 调用。

这是使用 AI 辅助开发的个人学习与实验项目。未声称独立完成全部代码、企业投产、复现论文结果或 SOTA；面试前需要能独立解释、改动和重跑其中的模型与评估流程。

## 实测结果

{headline}

{metrics_table(v1, quality)}

{comparisons_text(quality)}

{chart}

{v1['train']}训练 / {v1['validation']}相关文档分组验证 / {v1['test']}测试查询，seed42。**V1测试已被观察；V2是同一集合的迭代比较，不是全新盲测。** 置信区间只覆盖给定模型的查询重采样，不能代表训练种子或新领域。nDCG/Recall 是检索指标，不是问答准确率。

{chosen_text(quality, selection)}

完整记录：[V2分析](reports/quality_analysis.md)、[V2结果](reports/quality.json)、[验证网格](reports/quality_selection.json)、[逐查询结果](reports/quality_per_query.jsonl)。原始 [V1分析](reports/analysis.md) 和图表保留。

## 实现内容

- **真实检索器训练**：MiniLM query/value 层 LoRA，{training['trainable_parameters']:,}个参数，BM25难负样本、温度0.05对比损失和已知假负例屏蔽。验证选第{training['selected_epoch']}轮；本项目没有微调Qwen。
- **两种重排**：V1使用池化交互特征MLP；V2比较MiniLM L6/L12 CPU动态INT8和BGE v2-m3 Q8_0 Vulkan的query/document联合token编码。只在验证集选择，实际量化运行参加测试；没有QAT或FP32质量等价声明。重排仅排序召回候选。
- **质量与预算分开呈现**：轻量路由在BM25路径实际跳过向量编码；V1减少{(1-v1['dense_call_fraction'])*100:.2f}%向量调用，同时nDCG@10从{v1['metrics']['neural_rerank']['ndcg@10']:.4f}降至{v1['metrics']['adaptive']['ndcg@10']:.4f}。
- **可追溯引用**：原文跨度ID约束选择→恢复逐字quote→核对source_id和字符偏移；标题不是证据，失败则拒答。引用存在不证明语义结论正确。
- **普通笔记本资源控制**：CPU默认2线程，成对推理batch≤8；可用RAM低于2GiB或进程RSS超过3GiB时停止重计算。批前采样不是OS硬限额，多进程重任务需串行启动。
- **真实本地资料流程**：默认真实SciFact语料；用户知识库初始为空，支持当前会话文字/PDF导入、执行路径、来源展开与token预算，不把模拟制度载入主页面。

## 架构

```mermaid
flowchart LR
    Q[查询或声明] --> B[BM25]
    Q --> D[MiniLM LoRA向量]
    B --> F[RRF候选]
    D --> F
    F --> C[V2 token cross-encoder]
    B --> R[V1预算路由与轻量基线]
    C --> E[原文窗口与token预算]
    R --> E
    E --> G[本地Qwen选择原文跨度]
    G --> V[来源与逐字引用核验]
    V --> A[回答或拒答]
```

训练、来源与限制见 [设计](docs/design.md)、[数据卡](DATA_CARD.md)、[模型卡](MODEL_CARD.md)、[面试准备](docs/interview.md)。

## Windows本地运行

Python3.10–3.12。CPU PyTorch已经足够，不需要安装CUDA训练包或购买云服务。首次下载公开数据、编码器与可选生成模型；基础模型不进Git仓库。下载脚本保存固定revision与文件SHA256，只加载本地safetensors/GGUF。

```powershell
git clone https://github.com/yyh20040202-ui/adaptive-rag-lab.git
cd adaptive-rag-lab
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
$env:ADAPTIVE_CPU_THREADS='2'
$env:ADAPTIVE_MIN_AVAILABLE_GIB='2.0'
{prepare_command}
python -m adaptive_rag.prepare_quality --l12
{generator_download}{reranker_download}python -m adaptive_rag.index
```

生成模型可选0.6B、1.7B与4B Instruct；分别使用 `prepare --generator`、`prepare_quality --generator`、`prepare_quality --generator-4b` 下载。基础权重只在本机保存，大小/哈希核验后才启动。4B采用部分GPU offload以给独立BGE服务留出显存；先确认RAM和磁盘余量，其他硬件应重新测量。下面的启动型号来自本机下载记录，诊断结果以报告为准。

{reranker_start}另一个终端使用带资源看门狗的启动器开启本地生成模型（Vulkan、CPU2线程、context3072、batch256/ubatch128、单并发）：

```powershell
python -m adaptive_rag.launch_generator --model-directory models/{folder} --gpu-layers {offload} --context 3072{load_flag}
```

启动器每2秒检查RAM，可用系统内存低于2GiB或自己启动的生成子进程RSS超过3GiB时停止该子进程；Ctrl+C也只停止自己启动的子进程。不会终止其他应用。模型选择和降级参数见`python -m adaptive_rag.launch_generator --help`；文件名来自下载记录，而不是假设固定量化文件名。本机当前记录为`models/{folder}/{filename}`。看门狗采样仍不是OS内存硬限额。

两个GGUF服务都显式使用 `--cache-ram 0 --ctx-checkpoints 0`，关闭跨提示RAM缓存与上下文检查点。本机早期重复诊断曾触发2GiB余量守卫；原始事件与处理保留在 [资源事件](reports/resource_incident.json)，不能把中断的三条输出写成完整六条结果。

4B启动器的 `--no-mmap` 对固定b11517运行时翻译为 `--load-mode none`。本机最初24层GPU/mmap加载同时驻留时触及RAM守卫；30层GPU加直接加载后重新运行。不是所有机器都受益，资源与GPU采样见 [运行核验](reports/runtime_verification.json)，需要按自身硬件复测。

第二个已激活环境的终端显式设置型号，再启动实验台：

```powershell
$env:LLAMA_MODEL_LABEL='{label}'
$env:ADAPTIVE_CPU_THREADS='2'
python -m adaptive_rag.service
```

打开 [http://127.0.0.1:8088](http://127.0.0.1:8088)，接口文档`/docs`。只接受loopback生成地址；未启动模型时明确拒答。文字/PDF导入只存当前进程内存，重启清空。当前实现未提供公开部署所需的鉴权、多租户隔离和持久化。

## 重现实验

公开仓库提供小型LoRA/ranker/router权重。要重训和比较模型，在启动生成器前串行运行：

```powershell
python -m adaptive_rag.train --epochs 4 --seed 42
python -m adaptive_rag.benchmark --seed 42
python -m adaptive_rag.index
python -m adaptive_rag.prepare_quality --l12
python -m adaptive_rag.prepare_quality --gpu-reranker
# 在另一个终端启动BGE服务后，执行下面两步
python -m adaptive_rag.quality --stage select --variants l6 l12 bge
python -m adaptive_rag.quality --stage evaluate
python -m adaptive_rag.verify_reference
python -m pytest -q
```

`select`仅使用163条分组验证查询，在L6/L12/BGE、候选深度20/50/100、融合权重0/0.25/0.5/0.75/1中选择nDCG@10；同分优先浅候选、L6。没有Vulkan GPU时可先只用默认L6/L12重新选择。冻结配置写入`checkpoints/quality_config.json`，随后`evaluate`才比较同一300条测试查询。缓存指纹包含文档ID/正文、模型/tokenizer、adapter/config和池化设置，改变这些输入需要重建，旧版缺少元数据的缓存会被拒绝。

启动已选择的生成器后，另在设置过`LLAMA_MODEL_LABEL`的终端执行：

```powershell
python -m adaptive_rag.real_generation --mode cross_encoder --output reports/real_corpus_generation_v2.json
python -m adaptive_rag.quality_analysis
```

新图为`quality_benchmark.png/svg`，保留旧图不覆盖。`quality_analysis --text-only`可在V2待完成时生成明确标注缺失结果的文本，不生成假分数。

## 真实生成的范围

{generation_text(generation, original)}

两版使用相同六条SciFact测试声明，qrels只表示检索相关性。该集合已用于查看失败，不能称独立QA评测。12条历史手写制度样例只是调试回归，保存在原报告中，不进入主页面。

## 来源与许可

- [SciFact/AllenAI](https://github.com/allenai/scifact) 与 [BEIR](https://github.com/beir-cellar/beir)：声明/证据与摘要许可分别见数据卡；BEIR test不是原任务隐藏榜单。
- [MiniLM编码器](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)、[cross-encoder L6](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)、[L12](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2)：公开模型卡与本地固定版本/哈希。
- [BGE v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)、[GPUStack GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF)：多语种预训练重排器的Q8_0本地推理，未在本项目微调。
- [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B)、[4B Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)、[ggml-org GGUF](https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF)、[LM Studio GGUF](https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF)、[llama.cpp](https://github.com/ggml-org/llama.cpp)：本地推理，不进行Qwen微调。
- [LoRA](https://arxiv.org/abs/2106.09685) 与 [Adaptive Re-Ranking](https://arxiv.org/abs/2606.25249)：借鉴参数高效训练、按查询权衡质量与计算的思路；模型/数据/方法不同，不声称复现论文结果。

代码与自训练ranker/router为MIT，MiniLM adapter为Apache-2.0，数据与基础模型保留上游许可。仓库不上传完整基础模型、私人资料、API密钥或个人简历。
'''


def render_model_card(training: dict, quality: dict | None, selection: dict | None,
                      generation: dict | None) -> str:
    lines = ["# Model card", "", "本卡由实测日志与本地公开模型下载记录生成。实际训练、上游预训练和本地推理分别说明。", "",
             "| 模型 | 来源 / revision | 用途 | 许可 |", "|---|---|---|---|",
             f"| MiniLM encoder | `{training['base_model']}` · `{training['base_revision']}` | 384维mean-pool / L2检索 | Apache-2.0 |",
             f"| Retriever LoRA | 项目训练，Q/V层r8/alpha16/dropout0.05 | {training['trainable_parameters']:,}参数，192-token输入 | Apache-2.0 |",
             "| V1 neural ranker | 项目训练，771→64→16→1 MLP | 池化交互特征重排，不是token cross-encoder | MIT |",
             "| V1 router | 项目训练，7→32→16→3 MLP | BM25可得特征预测预算路径 | MIT |"]
    for variant in ["l6", "l12"]:
        provenance = read_json(f"models/cross_encoder_{variant}/provenance.json")
        if provenance:
            lines.append(f"| V2 cross-encoder {variant.upper()} | `{provenance['model']}` · `{provenance['revision']}` | MS MARCO预训练成对相关性；本项目未训练该模型 | {provenance.get('license', '见上游')} |")
    bge = read_json("models/bge_reranker/provenance.json")
    if bge:
        lines.append(f"| V2 BGE v2-m3 | `{bge['model']}` · `{bge['revision']}` | BAAI多语种预训练重排器，GPUStack Q8_0转换；本项目未训练 | {bge.get('license', '见上游')} |")
    for folder in ["qwen", "qwen17", "qwen4"]:
        provenance = read_json(f"models/{folder}/provenance.json")
        if provenance:
            filename = Path(provenance["file"]["path"]).name
            lines.append(f"| Qwen GGUF / {folder} | `{provenance['model']}` · `{provenance['revision']}` | `{filename}`本地受约束生成；没有Qwen SFT/LoRA | {provenance.get('license', '见上游')} |")
    lines += ["", "## 实际训练与验证", "",
              f"LoRA底座冻结，{training['backend']}，训练{training['epochs_requested']}轮 / {training['optimizer_steps']}步，batch{training['batch_size']}，验证损失选第{training['selected_epoch']}轮。训练集已知正例从BM25难负例中剔除，batch中的已知假负例屏蔽；未标注相关文献仍可能被误当负例。", "",
              "训练数据与模型记录见 `reports/training.json`、`reports/split.json`、`reports/retrieval.json`。单种子与小数据集结果不代表跨领域鲁棒性。", "",
              chosen_text(quality, selection), "",
              "V2公开cross-encoder对query/document联合token编码输出相关性原始logit。MiniLM L6/L12以FP32 safetensors下载，运行时仅Linear层做CPU动态INT8量化；BGE以Q8_0 GGUF通过Vulkan推理。没有量化感知训练或FP32/量化质量等价假设。最大384 tokens，batch≤8、CPU默认2线程；仅在163条分组验证查询上选三模型、候选深度和融合比例。实际选定backend/inference见 `checkpoints/quality_config.json`，完整45项网格见 `reports/quality_selection.json`。", "",
              "## 模型文件与来源", "",
              "下载记录保存revision、URL、SHA256与字节数；权重仅使用safetensors/GGUF，完整基础权重不上传仓库。0.6B/1.7B量化权重来自ggml-org，4B Instruct量化权重来自LM Studio community，上游基础模型均来自Qwen；llama.cpp使用固定b11517 Windows Vulkan运行时。", ""]
    for folder in ["cross_encoder_l6", "cross_encoder_l12", "bge_reranker", "qwen", "qwen17", "qwen4"]:
        provenance = read_json(f"models/{folder}/provenance.json")
        if not provenance:
            continue
        files = provenance.get("files", [provenance["file"]] if "file" in provenance else [])
        for file in files:
            if Path(file["path"]).name.endswith((".safetensors", ".gguf")):
                lines.append(f"- `{folder}/{Path(file['path']).name}` SHA256：`{file['sha256']}`。")
    lines += ["", "上游模型卡：[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)、[cross-encoder L6](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)、[L12](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2)、[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)、[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)。[BGE基础模型](https://huggingface.co/BAAI/bge-reranker-v2-m3)、[GPUStack量化转换](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF)。感谢Sentence Transformers/MiniLM、BAAI、GPUStack、Qwen与ggml-org贡献者。", "",
              "## 推理、引用和限制", "",
              "生成模型只进行本地推理，型号由 `LLAMA_MODEL_LABEL` 显式配置并写入报告。V2模型选择原文跨度ID，客户端恢复逐字quote并核对source_id及原文字符偏移；JSON互斥回答/拒答分支。缺少可引用证据、模型不可用或审计失败时拒答。", ""]
    lines += ["SciFact额外使用保守的字面标识守卫：数字复合词、英文缩写或混合大小写实体缺失时拒答。它可能错拒只有同义表达的证据，不是语义蕴含模型。支持标签后只重复声明的回答也会被拒绝。", "",
              "GGUF服务RAM提示缓存与上下文检查点均关闭（--cache-ram 0 / --ctx-checkpoints 0）。历史内存守卫事件完整保留；重复请求后的资源采样见运行报告。", ""]
    if generation:
        lines.append(generation_text(generation, None))
    else:
        lines.append("V2真实文献生成尚未完成，不把已下载的大模型或约束格式当作已验证的问答能力。")
    lines += ["", "- 原文引用存在不等于结论受支持；跨度约束也不提供语义蕴含证明。",
              "- V2复用了已查看过的BEIR测试查询；配置由验证集选定，但不能宣称全新盲测。",
              "- 上游MiniLM/MS MARCO/BGE/Qwen训练数据污染未知；英语生物医学结果不能外推中文企业资料。",
              "- 本项目未进行独立QA金标准评估，不宣称幻觉为零、QA准确率、SOTA或生产级事实性。",
              "- 路由成本0/1/1.25是代理单位，不是实测token费用、焦耳或同比能耗收益。",
              "- CPU两线程、batch八、可用内存2GiB与RSS3GiB守卫是批前采样，不是操作系统硬限额。生成器建议通过 `python -m adaptive_rag.launch_generator` 启动：单并发、2秒RAM看门狗，只停止自身子进程。", ""]
    return "\n".join(lines)


def plot_quality(v1: dict, quality: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    values = dict(quality["metrics"])
    methods = list(values)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f9fc", "axes.facecolor": "#f7f9fc"})
    fig, (bars, differences) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
    reverse = methods[::-1]
    colors = ["#315de4" if m == "cross_encoder" else "#14a08d" if m == "neural_rerank" else "#94a3b8" for m in reverse]
    bars.barh([NAMES[m] for m in reverse], [values[m]["ndcg@10"] for m in reverse], color=colors, height=.66)
    for i, method in enumerate(reverse):
        score = values[method]["ndcg@10"]
        bars.text(score+.009, i, f"{score:.4f}", va="center", color="#24314c")
    bars.set_xlim(0, 1.02)
    bars.set_xlabel("nDCG@10")
    bars.set_title("Measured retrieval quality", loc="left", fontweight="bold", pad=15)
    bars.grid(axis="x", alpha=.12)
    bars.set_axisbelow(True)
    comparisons = quality.get("paired_ndcg_comparisons", {})
    display = [(key, comparisons[key]) for key in ["cross_encoder-vs-bm25", "cross_encoder-vs-neural_rerank"] if key in comparisons]
    for i, (key, comparison) in enumerate(display):
        delta, (low, high) = comparison["delta"], comparison["ci95"]
        # A percentile bootstrap interval need not contain the observed estimate.
        differences.plot([low, high], [i, i], color="#315de4", linewidth=3)
        differences.scatter([delta], [i], color="#315de4", s=65, zorder=4)
        differences.annotate(f"{delta:+.4f}\n95% [{low:.4f}, {high:.4f}]", (high, i), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
    differences.set_yticks(range(len(display)), ["V2 vs BM25" if key.endswith("bm25") else "V2 vs V1 MLP" for key, _ in display])
    differences.axvline(0, color="#8a97ad", linestyle="--", linewidth=1)
    differences.set_ylim(-.7, max(len(display)-.3, .7))
    if display:
        low = min(min(row["ci95"]) for _, row in display)
        high = max(max(row["ci95"]) for _, row in display)
        width = max(high-low, .04)
        differences.set_xlim(min(0, low)-width*.15, max(0, high)+width*.75)
    else:
        differences.text(.5, .5, "Paired intervals not recorded", transform=differences.transAxes, ha="center")
    differences.set_xlabel("Absolute nDCG@10 difference")
    differences.set_title("Paired query bootstrap", loc="left", fontweight="bold", pad=15)
    differences.grid(axis="x", alpha=.12)
    fig.suptitle("Adaptive RAG Lab · BEIR SciFact · V1 / V2", x=.02, ha="left", fontsize=15, fontweight="bold", color="#14213d")
    fig.text(.02, .018, "Same previously inspected 300 test queries; V2 configuration selected on 163 grouped validation queries.\nRetrieval metrics are not QA accuracy. Intervals exclude training-seed and dataset-shift uncertainty.", fontsize=8, color="#5b6781")
    fig.tight_layout(rect=[0, .10, 1, .92])
    fig.savefig("reports/quality_benchmark.png", dpi=180, metadata={"Description": "Real V1/V2 retrieval reports; reused test disclosed"})
    fig.savefig("reports/quality_benchmark.svg")
    plt.close(fig)


def run(text_only: bool = False) -> dict:
    v1 = read_json("reports/retrieval.json", required=True)
    training = read_json("reports/training.json", required=True)
    quality = read_json("reports/quality.json", required=not text_only)
    selection = read_json("reports/quality_selection.json")
    generation = read_json("reports/real_corpus_generation_v2.json")
    original = read_json("reports/real_corpus_generation.json")
    Path("README.md").write_text(render_readme(v1, quality, selection, training, generation, original), encoding="utf-8")
    Path("MODEL_CARD.md").write_text(render_model_card(training, quality, selection, generation), encoding="utf-8")
    Path("reports/quality_analysis.md").write_text(render_analysis(v1, quality, selection, training, generation, original), encoding="utf-8")
    if not text_only:
        plot_quality(v1, quality)
    return {"v2_retrieval_measured": quality is not None, "v2_generation_measured": generation is not None,
            "figures_written": not text_only, "v1_graph_preserved": True,
            "text_outputs": ["README.md", "MODEL_CARD.md", "reports/quality_analysis.md"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-only", action="store_true")
    print(json.dumps(run(parser.parse_args().text_only), indent=2), flush=True)
