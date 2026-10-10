"""Generate figures, extra paired comparisons and readable analysis from real logs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import file_sha256
from .metrics import paired_bootstrap


if __name__ == "__main__":
    report = json.loads(Path("reports/retrieval.json").read_text(encoding="utf-8"))
    training = json.loads(Path("reports/training.json").read_text(encoding="utf-8"))
    generation = json.loads(Path("reports/generation.json").read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path("reports/per_query.jsonl").read_text().splitlines()]
    extras = {}
    for a, b in [("neural_rerank", "bm25"), ("neural_rerank", "hybrid_lora")]:
        extras[f"{a}-vs-{b}"] = paired_bootstrap(np.array([r["metrics"][a]["ndcg@10"] for r in details]),
                                                np.array([r["metrics"][b]["ndcg@10"] for r in details]))
    files = [Path("checkpoints/config.json"), Path("checkpoints/ranker.safetensors"),
             Path("checkpoints/router.safetensors"), Path("checkpoints/retriever_lora/adapter_model.safetensors"),
             Path("checkpoints/retriever_lora/adapter_config.json")]
    provenance = {"artifacts": {str(p): file_sha256(p) for p in files},
                  "source_code": {str(p): file_sha256(p) for p in sorted(Path("adaptive_rag").glob("*.py"))},
                  "input_sha256": json.loads(Path("reports/split.json").read_text())["input_sha256"]}
    Path("reports/artifact_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    Path("reports/additional_comparisons.json").write_text(json.dumps(extras, indent=2), encoding="utf-8")
    methods = list(report["metrics"])
    names = ["BM25", "Frozen dense", "Frozen hybrid", "LoRA dense", "LoRA hybrid", "Neural rerank", "Adaptive"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#d7dbd5", "figure.facecolor": "#fafaf6",
                         "axes.facecolor": "#fafaf6"})
    fig, (ax, scatter) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    scores = [report["metrics"][m]["ndcg@10"] for m in methods]
    colors = ["#9caf9e"]*5+["#bd7743", "#536f78"]
    ax.barh(names[::-1], scores[::-1], color=colors[::-1], height=.65)
    for i, value in enumerate(scores[::-1]):
        ax.text(value+.004, i, f"{value:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, .8)
    ax.set_xlabel("nDCG@10 (300 held-out queries)")
    ax.set_title("Retrieval quality", loc="left", fontweight="bold", pad=15)
    for method in ["bm25", "hybrid_lora", "neural_rerank", "adaptive"]:
        ms = report["latency"][method]["p50_ms"]
        score = report["metrics"][method]["ndcg@10"]
        color = "#bd7743" if method == "neural_rerank" else "#536f78" if method == "adaptive" else "#9caf9e"
        scatter.scatter(ms, score, color=color, s=100, edgecolor="white", linewidth=1.5, zorder=3)
        scatter.annotate({"bm25":"BM25", "hybrid_lora":"LoRA hybrid", "neural_rerank":"Neural rerank", "adaptive":"Adaptive"}[method],
                         (ms, score), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    scatter.set_ylim(.63, .72)
    scatter.set_xlim(0, max(r["p50_ms"] for r in report["latency"].values())*1.2)
    scatter.grid(axis="y", alpha=.2)
    scatter.set_ylabel("nDCG@10")
    scatter.set_xlabel("Warm serial CPU p50 / ms (40 sampled queries)")
    scatter.set_title("Quality / compute trade-off", loc="left", fontweight="bold", pad=15)
    fig.suptitle("Adaptive RAG Lab · BEIR SciFact · seed 42", x=.02, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, .02, 1, .92])
    fig.savefig("reports/benchmark.png", dpi=180)
    fig.savefig("reports/benchmark.svg")
    plt.close(fig)
    gain = (report["metrics"]["neural_rerank"]["ndcg@10"] / report["metrics"]["bm25"]["ndcg@10"] - 1)*100
    saved = (1-report["dense_call_fraction"])*100
    ci = extras["neural_rerank-vs-bm25"]["ci95"]
    lines = ["# 实验分析", "", "所有数字由本项目真实运行结果生成，未补写理想指标。", "",
             f"BEIR SciFact：{report['corpus']} 篇文档；{report['train']} 条训练、{report['validation']} 条验证、{report['test']} 条测试查询。", "",
             "| 方法 | nDCG@10 | MRR@10 | Recall@10 | p50 (ms) |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        m = report["metrics"][method]
        timing = f"{report['latency'][method]['p50_ms']:.2f}" if method in report["latency"] else "未单独计时"
        lines.append(f"| {method} | {m['ndcg@10']:.4f} | {m['mrr@10']:.4f} | {m['recall@10']*100:.2f}% | {timing} |")
    lines += ["", f"完整重排序流程相对 BM25 的 nDCG@10 提升 {gain:.2f}%，绝对差的成对查询 bootstrap 95% 区间为 [{ci[0]:.4f}, {ci[1]:.4f}]。这是查询重采样的不确定性，不能代表训练种子或其他数据集的方差。", "",
              f"LoRA 实际训练 {training['epochs_requested']} 轮 / {training['optimizer_steps']} 步，只更新 {training['trainable_parameters']:,} 个参数。验证损失选择第 {training['selected_epoch']} 轮；后续轮次训练损失下降、验证损失上升，出现过拟合。", "",
              "重排序模型是基于池化向量交互特征的小型 MLP，不能称作 token cross-encoder。单独 LoRA 的提升与重排序的提升都列在对照中。", "",
              f"自适应路由对 {report['route_counts']['bm25']} / {report['test']} 条查询跳过向量编码与检索，减少 {saved:.2f}% 的向量检索调用。但 nDCG@10 从完整流程的 {report['metrics']['neural_rerank']['ndcg@10']:.4f} 降至 {report['metrics']['adaptive']['ndcg@10']:.4f}。它是预算策略，不能写成准确率不变。", "",
              "时延采用 40 条固定随机查询、预热后的 CPU 串行计时，计入查询分词/编码、相似度计算和必要的重排，排除模型加载、索引构建和 Qwen 生成。不代表吞吐量或线上尾延迟保证。", "",
              "## 本地生成检查", "",
              f"Qwen3-0.6B Q4_0 在 12 个手写模拟制度样例上通过 {generation['passed']} / {generation['cases']} 个固定原子信息/来源/拒答检查。原始回答和失败案例完整保存在 generation.json。这个小样例集参与了输出约束和拒答规则的调试，只是功能回归检查，不能视为独立 QA 准确率。", "",
              "最初宽松格式允许答案没有引用；随后改成回答/拒答两个互斥的 JSON 分支。增加无主题词重合、问题复述和显式缺失事实的保守拦截。中间结果保留在 initial_generation.json 与 schema_only_generation.json，避免把格式修复当作模型能力提升。", "",
              "引用核验只证明 quote 在提供的上下文中存在，并把它映射回原文位置。模型仍可能引用无关片段或生成不被引文支持的结论；离线测试专门保留这一限制。当前小模型和规则不能支撑生产级事实性保证。", "",
              "## 排序数值修复", "",
              "首版 RRF 使用 float32，两个相等的 rank sum 在不同加法顺序下可能产生一个 ULP 差异。离线测试发现后改用 float64，并按文档位置打破精确同分。相同种子和验证规则重新运行，首版报告保留在 initial_float32_retrieval.json；没有根据测试标签选配置。", "",
              "## 适用边界", "",
              "- BEIR 的 SciFact test 来自原 SciFact 有标注验证集，不是原任务隐藏测试榜单。", "- 共享检索语料合理；训练/验证的监督查询按相关文档连通分组。测试查询不进入训练与配置选择。", "- MiniLM 预训练中的数据污染未知，项目不能声称严格 zero-shot 或 SOTA。", "- 英语生物医学检索结果不能外推中文企业制度；演示默认关闭跨领域学习路由。", "- 单种子、小训练集、代理计算成本及小生成模型限制都需要在后续实验中检查。", ""]
    Path("reports/analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ndcg_relative_gain_percent": gain, "dense_calls_saved_percent": saved,
                      "neural_vs_bm25_ci95": ci, "generation_smoke_passed": generation["passed"]}, indent=2), flush=True)
