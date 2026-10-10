# Data card · BEIR SciFact

来源：[AllenAI SciFact](https://github.com/allenai/scifact)、[BEIR](https://github.com/beir-cellar/beir)。下载地址由 `adaptive_rag.prepare` 固定，原始压缩包 SHA256 为 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`。

5,183篇科学论文题目/摘要，1,109条查询；官方BEIR train 809条、test 300条。项目把809条train按相关文档的连通分组拆为646训练/163验证；引用同一文档的查询不能跨这两个集合。分组清单和四个输入文件哈希见 `reports/split.json`。

检索语料是共享的，这是固定语料检索设置。测试查询不用于LoRA/重排/路由训练、负样本挖掘、早停或配置选择。基座模型预训练是否接触过SciFact未知，无法排除预训练污染。训练阶段屏蔽的假负样本仅使用已知train qrels，不能发现所有未标注相关文献。

BEIR test 是原始SciFact的有标注验证集，不是原始SciFact隐藏测试集。标签只说明文档相关，不是问答答案标签；生成结果也不能借此声称正确率。

许可：[SciFact LICENSE](https://github.com/allenai/scifact/blob/master/LICENSE.md) 将claims/evidence列为 CC BY 4.0、论文摘要列为S2ORC的 ODC-By 1.0、源码列为Apache-2.0。感谢David Wadden等SciFact作者、AllenAI、S2ORC和BEIR贡献者。仓库不重新分发完整原始语料，仅记录公开标识、排名和少量演示引文。

领域为英语生物医学研究文献。它便于检验检索算法，不能证明中文企业问答或其他领域效果。`examples/policies.json`是明确标记的人工测试夹具，仅用于格式回归，不作为真实企业经历或算法测试语料。
