"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const methodLabels = {
  bm25: "BM25 稀疏检索", dense_frozen: "原始 MiniLM", hybrid_frozen: "BM25 + 原始 MiniLM",
  dense_lora: "MiniLM LoRA", hybrid_lora: "混合检索 + LoRA", neural_rerank: "混合检索 + 神经重排序",
  adaptive: "预算路由", cross_encoder: "Cross-Encoder 重排序"
};
const modeNotes = {
  cross_encoder: "逐对编码问题与候选文献，优先检索质量；本机会串行执行并限制批大小。",
  neural_rerank: "在混合候选上运行已训练的神经重排序，优先查看检索质量。",
  hybrid_lora: "融合 BM25 与经过 LoRA 适配的 MiniLM 排名，兼顾词匹配和语义检索。",
  adaptive: "用轻量路由器选择检索路径以减少计算；可能牺牲部分排名质量。",
  bm25: "仅运行关键词稀疏检索，开销较低；分数只在当前检索路径内解释。"
};
const reasonLabels = {
  no_lexical_evidence: "检索未找到可用的词匹配证据", no_topical_overlap: "文献与问题缺少足够的主题关联",
  local_generator_unavailable: "本地生成模型尚未启动；下方检索证据仍可阅读",
  model_abstained: "模型判断现有文献不足以回答", missing_query_anchor: "证据中缺少问题中的关键标识", budget_has_no_evidence: "当前 token 预算未能容纳证据",
  generation_or_format_error: "生成或格式检查未通过", missing_or_invalid_citations: "回答缺少合格引用",
  quote_not_in_context: "引用片段未与原文完全匹配", unknown_source: "引用来源不在检索证据中",
  question_echo: "模型重复了问题，未提供有效回答", invalid_answer: "模型未给出合格回答"
};
const metricNotes = {
  "ndcg@10": "nDCG@10：相关文档排得越靠前，分数越高；取值 0–1。",
  "mrr@10": "MRR@10：衡量前 10 个结果中首篇相关文档的排名；取值 0–1。",
  "recall@10": "Recall@10：前 10 个检索结果覆盖了多少已标注相关文档；取值 0–1。"
};
const samples = { demo: [], scifact: [] };
const evidenceNodes = new Map();
const history = [];
let busy = false;
let elapsedTimer = null;
let toastTimer = null;
let reports = null;
let chosenMetric = "ndcg@10";
let currentResult = null;
let currentRequest = null;
let knowledgeCorpus = "scifact";
let library = null;
let libraryRequestId = 0;
let health = null;
let resourceSnapshot = null;
let modeTouched = false;
let sampleOffset = 0;
let statusLoading = false;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}
function number(value) { return typeof value === "number" && Number.isFinite(value); }
function fixed(value, digits = 2) { return number(value) ? value.toFixed(digits) : "—"; }
function count(value) { return number(value) ? value.toLocaleString("zh-CN") : "—"; }
function label(method) { return methodLabels[method] || method || "未记录路径"; }
function percent(value, digits = 1) { return number(value) ? `${(value * 100).toFixed(digits)}%` : "—"; }
function signed(value, digits = 3) { return number(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(digits)}` : "—"; }
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24"); svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", name === "document" ? "M5 3h10l4 4v14H5V3Zm10 0v5h4M9 12h6m-6 4h6" : "M4 12h16m-6-6 6 6-6 6");
  svg.append(path); return svg;
}
function detailRow(title, value) {
  const row = element("div", undefined, "detail-row");
  row.append(element("span", title), element("span", value)); return row;
}
function toast(message) {
  const area = $("#toast"); area.textContent = message; area.classList.remove("hidden");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => area.classList.add("hidden"), 2600);
}
async function api(path, options = {}) {
  const { timeout = 120000, ...fetchOptions } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, { ...fetchOptions, signal: controller.signal, cache: "no-store" });
    let data;
    try { data = await response.json(); } catch { throw new Error("服务未返回有效数据，请检查本地服务状态。"); }
    if (!response.ok) {
      let detail = typeof data.detail === "string" ? data.detail : data.message;
      if (!detail && Array.isArray(data.detail)) detail = data.detail.map(item => item.msg).filter(Boolean).join("；");
      throw new Error(detail || `请求未完成（HTTP ${response.status}）`);
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("等待超过时限。本地任务可能仍在运行，请先检查执行状态，再重试。");
    if (error instanceof TypeError) throw new Error("无法连接本地服务，请确认工作台服务仍在运行。");
    throw error;
  } finally { clearTimeout(timer); }
}
function post(path, data) { return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) }); }
function showError(message) { $("#error-text").textContent = message; $("#error").classList.remove("hidden"); }
function updateQueryLength() { $("#query-length").textContent = `${$("#query").value.length} / 1000`; }
function setBusy(value, answer = true) {
  busy = value; $("#progress").classList.toggle("hidden", !value);
  ["#ask-button", "#search-only", "#query", "#corpus", "#mode", "#budget", "#top-k", "#open-import"].forEach(selector => { $(selector).disabled = value; });
  $$("#examples button").forEach(button => { button.disabled = value; });
  $("#retry-query").disabled = value;
  $("#query-form").setAttribute("aria-busy", String(value));
  clearInterval(elapsedTimer);
  if (value) {
    const started = performance.now();
    $("#progress-label").textContent = answer ? "正在检索并调用本地模型" : "正在检索与重排序";
    $("#progress-note").textContent = "首个请求可能需要加载模型；这里显示实际等待时间。";
    $("#elapsed").textContent = "0.0s";
    elapsedTimer = setInterval(() => { $("#elapsed").textContent = `${((performance.now() - started) / 1000).toFixed(1)}s`; }, 100);
  }
}
function showTab(name) {
  if (!["workspace", "experiments", "knowledge"].includes(name)) return;
  $$(".nav").forEach(node => { const active = node.dataset.tab === name; node.classList.toggle("active", active); if (active) node.setAttribute("aria-current", "page"); else node.removeAttribute("aria-current"); });
  $$(".tab").forEach(node => node.classList.toggle("active", node.id === name));
  $("#breadcrumb").textContent = { workspace: "问答工作台", experiments: "评测与对照", knowledge: "资料管理" }[name];
  if (name === "knowledge") loadDocuments();
}
function updateModeNote() {
  const corpus = $("#corpus").value;
  $("#mode-note").textContent = corpus === "demo" && $("#mode").value === "adaptive" ? "学习路由器尚未在导入资料上验证；当前本地文档会使用保守检索策略。" : modeNotes[$("#mode").value];
  $("#domain-note").textContent = corpus === "demo" ? "导入资料仅存当前进程；尚未评测跨领域问答效果。" : "真实科学文献；公开查询用于展示检索与引用流程。";
  $("#top-k").title = "仅控制“只检索”的结果数量；回答会按 token 预算选择证据。";
}
function sampleButtons() {
  const corpus = $("#corpus").value, area = $("#examples"); area.replaceChildren();
  const available = samples[corpus];
  if (available.length) {
    for (let index = 0; index < Math.min(3, available.length); index++) {
      const text = available[(sampleOffset + index) % available.length];
      const button = element("button", undefined, "example"); button.type = "button";
      button.append(element("span", text, "example-text"));
      button.addEventListener("click", () => { if (busy) return; $("#query").value = text; updateQueryLength(); $("#query").focus(); });
      area.append(button);
    }
  } else {
    area.append(element("span", corpus === "demo" ? "先导入真实资料，再根据正文提出问题。" : "暂无公开样例，请输入英文科学问题或论断。", "subtle"));
  }
  $("#examples-label").textContent = corpus === "demo" ? "基于你的真实资料提问" : "真实诊断样例 · 包含拒答示例";
  $("#shuffle-examples").disabled = available.length <= 3;
  $("#query").placeholder = corpus === "demo" ? "根据你导入的资料提问…" : "输入英文科学问题或论断，检索相关研究文献…";
  updateModeNote();
}

function renderTrace(data) {
  const trace = $("#trace"); trace.replaceChildren();
  const steps = Array.isArray(data.trace) ? data.trace : [];
  if (!steps.length) {
    const row = element("div", undefined, "trace-item waiting"); row.append(element("i"));
    const body = element("div"); body.append(element("b", "暂无可执行的检索"), element("p", "请确认语料与模型已准备好")); row.append(body); trace.append(row);
  }
  for (const step of steps) {
    const executed = step.executed === true;
    const row = element("div", undefined, `trace-item ${executed ? "done" : "skipped"}`); row.append(element("i"));
    const body = element("div"); body.append(element("b", step.step || step.name || "检索步骤"), element("p", executed ? (number(step.ms) ? `已执行 · ${fixed(step.ms, 1)} ms` : "已执行") : "本次查询跳过")); row.append(body); trace.append(row);
  }
  if (number(data.generation_ms)) {
    const row = element("div", undefined, "trace-item done"); row.append(element("i")); const body = element("div");
    body.append(element("b", "本地模型生成"), element("p", `${fixed(data.generation_ms / 1000)} 秒 · ${data.abstain ? "未提供核验通过的回答" : "已返回回答"}`)); row.append(body); trace.append(row);
  }
  let policy = "按指定路径执行。";
  if (data.policy === "local_conservative_policy") policy = "导入文档采用保守 BM25；跨领域学习路由尚未验证。";
  if (data.policy === "learned_scifact_router") policy = "路由依据为稀疏检索分数、间隔、熵、词覆盖率与查询长度。";
  if (data.policy === "empty_local_corpus") policy = "当前导入语料为空，请先添加资料。";
  $("#route-details").textContent = `本次：${label(data.route)}。${policy}`;
  const stats = $("#run-stats"); stats.replaceChildren();
  for (const [title, value] of [["检索耗时", `${fixed(data.retrieval_ms, 1)} ms`], ["生成耗时", number(data.generation_ms) ? `${fixed(data.generation_ms / 1000)} s` : "未运行"]]) {
    const item = element("div", undefined, "run-stat"); item.append(element("span", title), element("b", value)); stats.append(item);
  }
  stats.classList.remove("hidden");
}
function appendHighlighted(node, text, citations) {
  const ranges = [];
  for (const citation of citations) {
    if (typeof citation.quote !== "string" || !citation.quote) continue;
    const start = text.indexOf(citation.quote);
    if (start >= 0) ranges.push([start, start + citation.quote.length]);
  }
  ranges.sort((left, right) => left[0] - right[0]);
  let cursor = 0;
  for (const [start, end] of ranges) {
    if (start < cursor) continue;
    node.append(document.createTextNode(text.slice(cursor, start)), element("mark", text.slice(start, end))); cursor = end;
  }
  node.append(document.createTextNode(text.slice(cursor)));
}
function focusEvidence(sourceId) {
  const source = evidenceNodes.get(String(sourceId)); if (!source) return;
  source.details.open = true; source.row.classList.remove("highlighted");
  source.row.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
  requestAnimationFrame(() => source.row.classList.add("highlighted"));
  $("#expand-evidence").textContent = $$("#evidence-list details").every(node => node.open) ? "收起全部" : "展开全部";
}
function renderEvidence(items, data) {
  const area = $("#evidence-list"); area.replaceChildren(); evidenceNodes.clear();
  const citations = Array.isArray(data.citations) ? data.citations : [];
  items.forEach((item, index) => {
    const row = element("article", undefined, "evidence"), details = element("details");
    const summary = element("summary"), heading = element("div", undefined, "evidence-summary");
    const title = element("div", undefined, "evidence-title"); title.append(element("b", item.title || "未命名文档"));
    const sourceCitations = citations.filter(citation => citation.source_id === item.source_id);
    const subtitle = element("div"); subtitle.append(element("span", item.source_id || "无来源 ID"));
    if (number(item.score)) subtitle.append(element("span", `分数 ${fixed(item.score, 4)}`));
    if (sourceCitations.length) subtitle.append(element("span", "被回答引用", "cited-tag"));
    title.append(subtitle); heading.append(element("span", String(index + 1).padStart(2, "0"), "evidence-rank"), title, element("span", "›", "evidence-chevron"));
    summary.append(heading, element("p", item.text || "", "evidence-preview")); details.append(summary);
    const text = typeof item.text === "string" ? item.text : "";
    const body = element("div", undefined, "evidence-body"); appendHighlighted(body, text, sourceCitations); details.append(body);
    const footer = element("div", undefined, "evidence-footer");
    footer.append(element("span", number(item.start) && number(item.end) ? `原文片段 · 字符 ${item.start}–${item.end}` : `来源原文 · ${text.length.toLocaleString()} 字符`));
    if (sourceCitations.length) footer.append(element("span", "高亮内容与回答引用完全匹配"));
    details.append(footer); row.append(details); area.append(row);
    evidenceNodes.set(String(item.source_id), { row, details });
    details.addEventListener("toggle", () => { const list = $$("#evidence-list details"); $("#expand-evidence").textContent = list.length && list.every(node => node.open) ? "收起全部" : "展开全部"; });
    if (index === 0) details.open = true;
  });
  if (!items.length) {
    const empty = element("div", undefined, "evidence-empty"); empty.append(element("b", "当前没有可展示的证据"), element("p", data.domain === "demo" ? "先导入资料，再根据正文提问。" : "调整问题，或检查语料是否已加载。")); area.append(empty);
  }
  $("#evidence-count").textContent = items.length;
  $("#expand-evidence").disabled = items.length === 0; $("#expand-evidence").textContent = "展开全部";
  const info = $("#result-summary"); info.replaceChildren(element("span", label(data.route)), element("span", `检索 ${fixed(data.retrieval_ms, 1)} ms`), element("span", "不同路径的分数不可直接比较"));
  $("#evidence-card").classList.remove("hidden");
}
function renderAnswer(data) {
  const card = $("#answer-card"); card.classList.remove("hidden"); card.classList.toggle("abstained", data.abstain === true);
  $("#answer-text").textContent = data.answer || "模型未返回回答。";
  const citations = Array.isArray(data.citations) ? data.citations : [];
  const verified = citations.length > 0 && citations.every(item => item.quote_verified === true);
  const status = $("#answer-status"); status.textContent = data.abstain ? "资料不足 / 核验未通过" : verified ? "引用原文已核验" : "请检查引用状态"; status.className = `pill${data.abstain || !verified ? " warn" : ""}`;
  const abstention = $("#abstention-note"); abstention.classList.toggle("hidden", !data.abstain);
  abstention.textContent = (data.audit_failures || []).map(reason => reasonLabels[reason] || reason).join("；") || "现有证据未能支持经过核验的回答。";
  const area = $("#citations"); area.replaceChildren();
  citations.forEach((item, index) => {
    const citation = element("div", undefined, "citation"), heading = element("div", undefined, "citation-header");
    heading.append(element("span", index + 1, "citation-index"), element("b", item.title || item.source_id)); citation.append(heading, element("blockquote", item.quote));
    const footer = element("div", undefined, "citation-footer"); footer.append(element("small", `${item.source_id} · 字符 ${item.start ?? "—"}–${item.end ?? "—"}${item.quote_verified ? " · 原文匹配" : ""}`));
    if (evidenceNodes.has(String(item.source_id))) { const jump = element("button", "定位到原文 ↘", "text-button"); jump.type = "button"; jump.addEventListener("click", () => focusEvidence(item.source_id)); footer.append(jump); }
    citation.append(footer); area.append(citation);
  });
  const meta = $("#answer-meta"); meta.replaceChildren();
  if (number(data.evidence_tokens)) meta.append(element("span", `证据 ${data.evidence_tokens} / ${data.evidence_budget || currentRequest?.evidence_budget || "—"} tokens`));
  if (data.usage && (number(data.usage.prompt_tokens) || number(data.usage.completion_tokens))) meta.append(element("span", `输入 ${count(data.usage.prompt_tokens)} · 输出 ${count(data.usage.completion_tokens)} tokens`));
  if (number(data.generation_ms)) meta.append(element("span", `生成 ${fixed(data.generation_ms / 1000)} s`));
  if (typeof data.model === "string") meta.append(element("span", data.model.split(";")[0]));
}
function displayResult(data, answer) {
  currentResult = data; $("#initial-state").classList.add("hidden");
  renderTrace(data); renderEvidence(data.evidence || data.hits || [], data);
  $("#answer-card").classList.toggle("hidden", !answer);
  if (answer) renderAnswer(data);
}
function renderHistory() {
  const area = $("#history-list"); area.replaceChildren();
  if (!history.length) area.append(element("p", "当前标签页还没有查询。", "history-empty"));
  history.forEach(item => {
    const button = element("button", undefined, "history-item"); button.type = "button";
    button.append(element("span", item.request.query, "history-query"));
    const info = element("span", undefined, "history-info"); info.append(element("span", `${item.request.corpus === "demo" ? "本地资料" : "SciFact"} · ${item.answer ? "问答" : "检索"}`), element("span", `${fixed(item.result.retrieval_ms, 1)} ms`)); button.append(info);
    button.addEventListener("click", () => {
      if (busy) return;
      currentRequest = item.request; $("#query").value = item.request.query; $("#corpus").value = item.request.corpus;
      if (!$("#mode").querySelector(`option[value="${item.request.mode}"]`)?.disabled) $("#mode").value = item.request.mode;
      $("#budget").value = item.request.evidence_budget; $("#top-k").value = item.request.top_k;
      updateQueryLength(); sampleButtons(); displayResult(item.result, item.answer); showTab("workspace"); $("#error").classList.add("hidden"); toast("已打开当前标签页保存的查询结果");
    }); area.append(button);
  });
  $("#clear-history").disabled = history.length === 0;
}
async function runQuery(answer) {
  if (busy) return;
  const query = $("#query").value.trim(); if (!query) { $("#query").focus(); return; }
  const request = { query, corpus: $("#corpus").value, mode: $("#mode").value, evidence_budget: Number($("#budget").value), top_k: Number($("#top-k").value) };
  currentRequest = { ...request, answer };
  $("#error").classList.add("hidden"); $("#answer-card").classList.add("hidden"); $("#evidence-card").classList.add("hidden"); $("#initial-state").classList.add("hidden"); setBusy(true, answer);
  const pendingTrace = element("div", undefined, "trace-item waiting"), pendingBody = element("div"); pendingTrace.append(element("i")); pendingBody.append(element("b", "请求已提交"), element("p", "等待服务返回实际执行路径")); pendingTrace.append(pendingBody); $("#trace").replaceChildren(pendingTrace); $("#route-details").textContent = "当前查询执行中；检索与生成完成后显示实际步骤和耗时。"; $("#run-stats").classList.add("hidden");
  try {
    const data = await post(answer ? "/api/ask" : "/api/search", request); displayResult(data, answer);
    history.unshift({ request: { ...request }, result: data, answer }); if (history.length > 8) history.pop(); renderHistory();
  } catch (error) { showError(error.message); $("#route-details").textContent = "服务未返回可展示的执行结果。请查看错误信息；等待超时并不表示后台计算已停止。"; }
  finally { setBusy(false); refreshStatus(); }
}

function reportView(data) {
  const base = data.retrieval || {}, quality = data.quality || {};
  return { ...base, ...quality, metrics: { ...(base.metrics || {}), ...(quality.metrics || {}) }, latency: { ...(base.latency || {}), ...(quality.latency || {}) }, route_counts: quality.route_counts || base.route_counts };
}
function methods(report, metric = "ndcg@10") {
  const order = ["bm25", "dense_frozen", "hybrid_frozen", "dense_lora", "hybrid_lora", "neural_rerank", "adaptive", "cross_encoder"];
  return Object.entries(report.metrics || {}).filter(([method, metrics]) => !/oracle|diagnostic/i.test(method) && metrics && number(metrics[metric])).sort((left, right) => {
    const a = order.indexOf(left[0]), b = order.indexOf(right[0]); return (a < 0 ? 99 : a) - (b < 0 ? 99 : b);
  });
}
function bestMethod(entries, metric) { return entries.reduce((best, item) => !best || item[1][metric] > best[1][metric] ? item : best, null); }
function renderBenchmark() {
  if (!reports) return;
  const report = reportView(reports), entries = methods(report, chosenMetric), best = bestMethod(entries, chosenMetric);
  const chart = $("#benchmark-chart"); chart.replaceChildren();
  if (!entries.length) { chart.append(element("p", "还没有可展示的实测报告。", "small-note")); return; }
  for (const [method, metrics] of entries) {
    const row = element("div", undefined, `chart-row${method === best[0] ? " best" : ""}`); row.append(element("span", label(method)));
    const track = element("div", undefined, "bar-track"), bar = element("div", undefined, `bar${method === best[0] ? " best" : method === "adaptive" ? " adaptive" : ""}`);
    bar.style.width = `${Math.max(0, Math.min(1, metrics[chosenMetric])) * 100}%`; track.append(bar); row.append(track, element("span", fixed(metrics[chosenMetric], 3), "chart-value")); chart.append(row);
  }
  $("#metric-explainer").textContent = metricNotes[chosenMetric];
  const baseline = report.metrics.bm25?.[chosenMetric], gain = number(baseline) && baseline > 0 ? (best[1][chosenMetric] / baseline - 1) : null;
  const insight = $("#comparison-insight"); insight.replaceChildren(element("span", "MEASURED GAIN"), element("strong", number(gain) ? `${gain >= 0 ? "+" : ""}${percent(gain)}` : "—"));
  insight.append(element("p", `${label(best[0])} 在 ${chosenMetric} 上相对 BM25 的实测变化。原始分数 ${fixed(best[1][chosenMetric], 4)}；评测集不等同于实际业务数据。`));
  const tbody = $("#results-table tbody"); tbody.replaceChildren();
  const overallBest = bestMethod(methods(report), "ndcg@10");
  for (const [method, metrics] of methods(report)) {
    const row = element("tr", undefined, method === overallBest?.[0] ? "best" : method === "adaptive" ? "adaptive" : "");
    const title = element("td", label(method)); if (method === overallBest?.[0]) title.append(element("span", "最佳 nDCG", "row-badge")); row.append(title);
    for (const metric of ["ndcg@10", "mrr@10", "recall@10"]) row.append(element("td", fixed(metrics[metric], 4)));
    const delta = number(report.metrics.bm25?.["ndcg@10"]) ? metrics["ndcg@10"] - report.metrics.bm25["ndcg@10"] : null;
    row.append(element("td", method === "bm25" ? "基线" : signed(delta), number(delta) ? delta >= 0 ? "positive" : "negative" : ""));
    row.append(element("td", fixed(report.latency[method]?.p50_ms)), element("td", fixed(report.latency[method]?.p95_ms))); tbody.append(row);
  }
  const queryCounts = [...new Set(Object.values(report.latency).map(item => item?.queries).filter(number))];
  const parts = [];
  if (number(report.train) || number(report.validation)) parts.push(`${count(report.train)} 训练 / ${count(report.validation)} 验证 / ${count(report.test)} 测试查询`);
  else if (number(report.test)) parts.push(`${count(report.test)} 条测试查询`);
  if (number(report.seed)) parts.push(`seed=${report.seed}`);
  if (queryCounts.length) parts.push(`延迟采样 ${queryCounts.join(" / ")} 条查询；缺失项显示为 —`);
  parts.push("检索延迟包含查询编码与排序，不包含生成、启动和建库；不同报告的时延样本需按各自协议解读。");
  $("#benchmark-protocol").textContent = parts.join("。 ");
}
function renderRouting(report) {
  const chart = $("#route-chart"), legend = $("#route-legend"); chart.replaceChildren(); legend.replaceChildren();
  const routes = Object.entries(report.route_counts || {}).filter(([, value]) => number(value) && value >= 0);
  const total = routes.reduce((sum, [, value]) => sum + value, 0);
  if (!total) { $("#routing-note").textContent = "当前报告没有路由分布记录。"; return; }
  for (const [method, value] of routes) {
    const fraction = value / total;
    const segment = element("div", fraction > .12 ? percent(fraction, 0) : "", "route-segment"); segment.style.width = `${fraction * 100}%`; segment.title = `${label(method)}：${value} / ${total}`; chart.append(segment);
    const item = element("span", undefined, "legend-item"); item.append(element("i"), element("span", `${label(method)} ${value}`)); legend.append(item);
  }
  const adaptive = report.metrics.adaptive?.["ndcg@10"], full = report.metrics.neural_rerank?.["ndcg@10"];
  let note = `${report.route_counts.bm25 || 0} / ${total} 条查询走 BM25。`;
  if (number(adaptive) && number(full)) note += `预算路由 nDCG@10 为 ${fixed(adaptive, 3)}，相对神经重排序差值 ${signed(adaptive - full)}。`;
  if (number(report.dense_call_fraction)) note += `减少 ${percent(1 - report.dense_call_fraction, 0)} 的向量调用。`;
  note += "该路径以节省计算为目标，工作台默认采用质量优先路径。"; $("#routing-note").textContent = note;
}
function renderGeneration(data) {
  const report = data.generation_real || data.generation;
  if (!report) { $("#generation-report").classList.add("hidden"); return; }
  const real = Boolean(data.generation_real), entries = Array.isArray(report.results) ? report.results : [];
  const answered = number(report.answered) ? report.answered : entries.length ? entries.filter(item => item.result?.abstain === false || item.abstain === false).length : null;
  const cases = number(report.cases) ? report.cases : number(report.queries) ? report.queries : entries.length || null;
  const stats = $("#generation-stats"); stats.replaceChildren();
  const verifiedCitations = report.valid_quote_citations ?? report.quote_verified_answers ?? (real && entries.length ? entries.flatMap(item => item.result?.citations || []).filter(item => item.quote_verified === true).length : null);
  const seconds = report.seconds ?? report.elapsed_seconds;
  const values = [["检查范围", count(cases), real ? "条真实文献查询" : "个人工构造功能用例"], ["返回回答", count(answered), "不等同于答案正确"], ["匹配原文的引用", count(verifiedCitations), "来源与原文位置核验"], ["检查耗时", number(seconds) ? `${fixed(seconds, 1)}s` : "—", "报告记录的总耗时"]];
  for (const [title, value, note] of values) { const item = element("div", undefined, "generation-stat"); item.append(element("span", title), element("b", value), element("small", note)); stats.append(item); }
  $("#generation-label").textContent = real ? "真实文献 · 小规模检查" : "人工用例 · 功能冒烟检查";
  $("#generation-limitations").textContent = real ? "这些查询用于核验真实文献的生成与引用流程；样本量较小，原文匹配不能证明答案的语义正确性。" : "人工构造功能用例只检查回答原子、拒答和引用契约；没有测得真实企业问答准确率。";
  $("#generation-report").classList.remove("hidden");
}
async function loadReports() {
  $("#refresh-reports").disabled = true;
  try {
    const data = await api("/api/reports", { timeout: 15000 }); reports = data;
    const report = reportView(data), best = bestMethod(methods(report), "ndcg@10"), training = data.training;
    if (!best) { $("#benchmark-chart").textContent = "评测尚未完成；实测报告就绪后会在这里展示。"; $("#report-subtitle").textContent = "尚无实测检索报告"; return; }
    $("#corpus-count").textContent = count(report.corpus); $("#best-ndcg").textContent = fixed(best[1]["ndcg@10"], 3); $("#best-method").textContent = label(best[0]);
    const baseline = report.metrics.bm25?.["ndcg@10"], gain = number(baseline) && baseline > 0 ? best[1]["ndcg@10"] / baseline - 1 : null;
    $("#quality-gain").textContent = number(gain) ? `${gain >= 0 ? "+" : ""}${percent(gain)}` : "—";
    $("#lora-fraction").textContent = percent(training?.trainable_fraction, 2); $("#test-count").textContent = `${count(report.test)} 条测试查询`;
    $("#report-subtitle").textContent = `${count(report.corpus)} 篇文献 · ${count(report.test)} 条保留测试查询`;
    const backend = report.config?.backend === "llama_vulkan" ? "CPU + Vulkan GPU" : "CPU";
    $("#report-environment").textContent = `${backend} · ${report.environment?.cpu_threads || training?.cpu_threads || "受限"} 线程 · seed ${report.seed ?? "—"}`;
    renderBenchmark(); renderRouting(report); renderGeneration(data);
    const details = $("#training-details"); details.replaceChildren();
    if (training) {
      details.append(detailRow("LoRA 可训练参数", count(training.trainable_parameters)), detailRow("训练步数", `${count(training.optimizer_steps)} steps`), detailRow("验证集选定轮次", `${training.selected_epoch ?? "—"} / ${training.epochs_requested ?? "—"}`), detailRow("训练后端", training.backend || "未记录"), detailRow("训练耗时", number(training.seconds) ? `${fixed(training.seconds / 60, 1)} 分钟` : "—"), detailRow("负样本", "BM25 难负样本"));
    } else details.append(detailRow("训练记录", "尚未提供"));
    const crossModel = data.quality?.config?.model || data.quality?.cross_encoder_model || data.quality?.model;
    if (crossModel) details.append(detailRow("Cross-Encoder", crossModel), detailRow("重排序模型来源", "预训练权重；未在 SciFact 训练"));
    if (data.quality?.config) details.append(detailRow("推理配置", data.quality.config.inference || "未记录"), detailRow("重排深度 / 融合权重", `${data.quality.config.rerank_depth ?? "—"} / ${data.quality.config.alpha ?? "—"}`), detailRow("配置选择", "仅通过文档分组验证集选择"));
    $("#experiment-limitations").textContent = data.quality ? "V2 在先前查看过结果的同一测试集上迭代；模型与配置仅通过验证集选择，不属于全新盲测。单一英语科学文献基准，模型预训练污染未知。Cross-Encoder 使用公开预训练权重，LoRA 训练作用于 MiniLM 检索编码器。检索质量、引用原文匹配和答案正确性是不同的评测对象。" : "单随机种子、单一英语科学文献基准。BEIR 的 SciFact test 对应原始数据的有标注验证集；模型预训练污染未知。检索质量、引用原文匹配和答案正确性是不同的评测对象。";
  } catch (error) { $("#benchmark-chart").textContent = error.message; $("#report-subtitle").textContent = "报告读取失败"; }
  finally { $("#refresh-reports").disabled = false; }
}

function emptyLibrary(title, note, canImport = false) {
  const card = element("div", undefined, "library-empty"); card.append(element("span", "▤"), element("h2", title), element("p", note));
  if (canImport) { const button = element("button", "＋ 导入第一份资料", "button secondary"); button.type = "button"; button.addEventListener("click", openImport); card.append(button); }
  return card;
}
function renderDocuments() {
  if (!library) return;
  const area = $("#document-list"), filter = $("#document-filter").value.trim().toLocaleLowerCase(); area.replaceChildren();
  const loaded = library.documents || [], filtered = loaded.filter(item => `${item.title} ${item.id}`.toLocaleLowerCase().includes(filter));
  $("#document-count").textContent = `共 ${count(library.count)} 篇${library.count > loaded.length ? ` · 载入前 ${loaded.length} 篇` : ""}${filter ? ` · 筛选 ${filtered.length} 篇` : ""}`;
  $("#library-storage").textContent = knowledgeCorpus === "demo" ? "当前进程内存 · 重启后清除" : "只读公开语料";
  if (!filtered.length) {
    area.append(emptyLibrary(filter ? "未找到匹配标题" : "你的资料库还没有文档", filter ? "筛选仅覆盖已载入的资料；试试其他关键词。" : "导入真实制度、研究笔记或文字版 PDF 后，即可从当前本地资料提问。", !filter && knowledgeCorpus === "demo")); return;
  }
  for (const item of filtered) {
    const card = element("article", undefined, "document-card"), top = element("div", undefined, "document-top"), symbol = element("div", undefined, "doc-icon"); symbol.append(icon("document"));
    top.append(symbol, element("span", knowledgeCorpus === "demo" ? "LOCAL SOURCE" : "SCIENTIFIC PAPER", "doc-label"));
    card.append(top, element("h3", item.title || "未命名文档"), element("p", item.preview || "暂无摘要"), element("small", `${item.id} · ${count(item.characters)} 字符`)); area.append(card);
  }
}
async function loadDocuments() {
  const requestId = ++libraryRequestId, corpus = knowledgeCorpus; const area = $("#document-list");
  area.replaceChildren(emptyLibrary("正在读取资料", "从本机服务载入真实文档列表…"));
  try { const data = await api(`/api/documents?corpus=${corpus}`, { timeout: 45000 }); if (requestId !== libraryRequestId) return; library = data; renderDocuments(); }
  catch (error) { if (requestId !== libraryRequestId) return; library = null; area.replaceChildren(emptyLibrary("资料读取未完成", error.message)); $("#document-count").textContent = "读取失败"; }
}
function setKnowledgeCorpus(corpus) { knowledgeCorpus = corpus; $("#document-filter").value = ""; $$("[data-knowledge-corpus]").forEach(node => { const active = node.dataset.knowledgeCorpus === corpus; node.classList.toggle("active", active); node.setAttribute("aria-pressed", String(active)); }); }
function openImport() { if (!busy) { $("#import-message").textContent = ""; $("#import-dialog").showModal(); } }
function updateRuntime() {
  const details = $("#runtime-details"); details.replaceChildren();
  if (health) {
    details.append(detailRow("生成模型", health.model || "未记录"), detailRow("生成状态", health.generator_available ? "在线" : "未启动"), detailRow("检索编码器", health.encoder_available ? "已准备" : "未准备"), detailRow("重排序模型", health.cross_encoder_available ? "已准备" : "未准备 / 不可用"));
  } else details.append(detailRow("服务状态", "暂未取得状态"));
  if (resourceSnapshot) {
    if (number(resourceSnapshot.cpu_threads)) details.append(detailRow("CPU 线程上限", String(resourceSnapshot.cpu_threads)));
    if (number(resourceSnapshot.max_pair_batch)) details.append(detailRow("成对推理批大小", String(resourceSnapshot.max_pair_batch)));
    if (number(resourceSnapshot.process_rss_gib)) details.append(detailRow("服务内存", `${fixed(resourceSnapshot.process_rss_gib)} GiB`));
    if (resourceSnapshot.heavy_jobs_serial === true) details.append(detailRow("重任务调度", "串行执行"));
  }
}
async function refreshStatus() {
  if (statusLoading) return; statusLoading = true; $("#refresh-status").disabled = true;
  const results = await Promise.allSettled([api("/api/health", { timeout: 10000 }), api("/api/resources", { timeout: 10000 })]);
  const status = $("#model-status");
  if (results[0].status === "fulfilled") {
    health = results[0].value; status.className = `status ${health.generator_available ? "good" : "warn"}`;
    status.replaceChildren(element("i"), element("span", health.generator_available ? "本地生成模型在线" : "生成未启动 · 可检索"));
    const modes = Array.isArray(health.retrieval_modes) ? health.retrieval_modes : ["bm25", "hybrid_lora", "neural_rerank", "adaptive"];
    const crossOption = $("#mode option[value=cross_encoder]");
    $$("#mode option").forEach(option => { option.disabled = !modes.includes(option.value); });
    crossOption.disabled = !health.cross_encoder_available || !modes.includes("cross_encoder");
    if (!modeTouched && !busy) $("#mode").value = health.default_mode && modes.includes(health.default_mode) ? health.default_mode : !crossOption.disabled ? "cross_encoder" : "neural_rerank";
    if (!modes.includes($("#mode").value) && !busy) $("#mode").value = health.default_mode || modes[0] || "bm25";
  } else {
    health = null; status.className = "status offline"; status.replaceChildren(element("i"), element("span", "状态连接失败"));
  }
  const meter = $("#resource-meter");
  resourceSnapshot = results[1].status === "fulfilled" ? results[1].value : null;
  if (resourceSnapshot && number(resourceSnapshot.memory_percent)) {
    meter.classList.remove("hidden");
    $("#memory-label").textContent = `${fixed(resourceSnapshot.memory_percent, 0)}%`;
    $("#memory-bar").style.width = `${Math.max(0, Math.min(100, resourceSnapshot.memory_percent))}%`;
    $("#memory-bar").style.background = resourceSnapshot.memory_percent > 85 ? "#c9a879" : "#a9c888";
    $("#resource-note").textContent = `可用 ${fixed(resourceSnapshot.available_gib)} / ${fixed(resourceSnapshot.total_gib)} GiB${number(resourceSnapshot.min_available_gib) ? ` · 保护阈值 ${fixed(resourceSnapshot.min_available_gib, 1)} GiB` : ""}`;
  } else { meter.classList.add("hidden"); }
  updateRuntime(); updateModeNote(); statusLoading = false; $("#refresh-status").disabled = false;
}

$$(".nav").forEach(button => button.addEventListener("click", () => showTab(button.dataset.tab)));
$$("[data-open-tab]").forEach(button => button.addEventListener("click", () => showTab(button.dataset.openTab)));
$("#query-form").addEventListener("submit", event => { event.preventDefault(); runQuery(true); });
$("#query").addEventListener("input", updateQueryLength);
$("#query").addEventListener("keydown", event => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); runQuery(true); } });
$("#search-only").addEventListener("click", () => runQuery(false));
$("#retry-query").addEventListener("click", () => runQuery(currentRequest?.answer !== false));
$("#corpus").addEventListener("change", () => { sampleOffset = 0; sampleButtons(); });
$("#mode").addEventListener("change", () => { modeTouched = true; updateModeNote(); });
$("#shuffle-examples").addEventListener("click", () => { sampleOffset = (sampleOffset + 3) % Math.max(1, samples[$("#corpus").value].length); sampleButtons(); });
$("#refresh-status").addEventListener("click", refreshStatus);
$("#refresh-reports").addEventListener("click", loadReports);
$$("[data-metric]").forEach(button => button.addEventListener("click", () => { chosenMetric = button.dataset.metric; $$("[data-metric]").forEach(node => { const active = node === button; node.classList.toggle("active", active); node.setAttribute("aria-pressed", String(active)); }); renderBenchmark(); }));
$("#expand-evidence").addEventListener("click", () => { const list = $$("#evidence-list details"), open = !list.every(node => node.open); list.forEach(node => { node.open = open; }); $("#expand-evidence").textContent = open ? "收起全部" : "展开全部"; });
$("#clear-history").addEventListener("click", () => { history.length = 0; renderHistory(); toast("已清空当前标签页的查询记录"); });
$("#copy-answer").addEventListener("click", async () => { if (!currentResult) return; try { await navigator.clipboard.writeText(currentResult.answer || ""); toast("已复制回答"); } catch { toast("浏览器暂不支持复制，可选中回答文本手动复制。"); } });
$$("[data-knowledge-corpus]").forEach(button => button.addEventListener("click", () => { setKnowledgeCorpus(button.dataset.knowledgeCorpus); loadDocuments(); }));
$("#document-filter").addEventListener("input", renderDocuments);
$("#open-import").addEventListener("click", openImport);
$("#close-import").addEventListener("click", () => $("#import-dialog").close());
$("#cancel-import").addEventListener("click", () => $("#import-dialog").close());
$("#import-form").addEventListener("submit", async event => {
  event.preventDefault(); const button = $("#import-form button[type=submit]"), message = $("#import-message"); button.disabled = true; message.textContent = "正在解析并导入资料…";
  try {
    const title = $("#document-title").value.trim(), file = $("#document-file").files[0];
    if (file) {
      if (file.size > 2 * 1024 * 1024) throw new Error("PDF 超过 2 MiB，请选择更小的文字版文件。");
      const encoded = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(",")[1]); reader.onerror = () => reject(new Error("无法读取选择的文件。")); reader.readAsDataURL(file); });
      await post("/api/documents/pdf", { title, content_base64: encoded });
    } else {
      const text = $("#document-text").value.trim(); if (!text) throw new Error("请粘贴正文或选择文字版 PDF。"); await post("/api/documents", { title, text });
    }
    setKnowledgeCorpus("demo"); $("#corpus").value = "demo"; sampleButtons(); $("#import-dialog").close(); $("#import-form").reset(); await loadDocuments(); toast("资料已导入本地当前进程，重启后清除。");
  } catch (error) { message.textContent = error.message; }
  finally { button.disabled = false; }
});

sampleButtons(); renderHistory(); updateQueryLength(); loadReports(); refreshStatus();
api("/api/example-queries", { timeout: 20000 }).then(data => { samples.scifact = (data.queries || []).map(item => item.query).filter(text => typeof text === "string" && text.trim()); sampleButtons(); }).catch(() => { sampleButtons(); });
setInterval(() => { if (document.visibilityState === "visible") refreshStatus(); }, 20000);
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") refreshStatus(); });
