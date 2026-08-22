"""character_reduce — reduce 流水线（Django 模块版）。

输入: staged uploads（`_resolve_staged_uploads` 的产出，每项含
      {name, kind, mime_type, content, file_url}，content 为全文）。
流程:
  1. 按戏份分层（纯规则，0 LLM）: 重头/中/客串
  2. 分批精读（0 LLM 组装 prompt）: 重头完整读；客串只取目标角色出现片段
  3. 每批 LLM 产出结构化笔记（带原文引用）: 性格证据/语言风格/行为演出/情绪触发点/关系网
  4. 汇总合并（LLM）: 处理前后期变化 → 角色属性总结 + 分类台词样本库 + 行为样本

`run_reduce_pipeline()` 返回可直接映射为 PrisMateDraft 的 dict。
llm_call 通过参数注入（真实环境绑 _generate_text）。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, List, Optional

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
TIER_MAIN = 30      # 重头: 目标角色台词 >= 30 句
TIER_MID = 5        # 中: 5~29 句；客串: < 5 句
CAME0_WINDOW = 3    # 客串片段: 台词前后各 N 行（保上下文、控 token）
BATCH_SIZE = 6      # 每批精读的文件数

# ---------------------------------------------------------------------------
# 1. 分层（纯规则，0 LLM）
# ---------------------------------------------------------------------------

def _speaker_line_count(content: str, target: str) -> int:
    """统计目标角色台词条数：以 `目标名:` 开头的行。"""
    if not target:
        return 0
    return sum(1 for ln in content.splitlines() if ln.strip().startswith(target))


def tier_for(line_count: int) -> str:
    if line_count >= TIER_MAIN:
        return "main"
    if line_count >= TIER_MID:
        return "mid"
    return "cameo"


def _tier_uploads(uploads: List[dict], target: str):
    """返回 {tier: [upload]}，每 upload 附 line_count。"""
    tiers = {"main": [], "mid": [], "cameo": []}
    for upload in uploads:
        n = _speaker_line_count(upload.get("content") or "", target)
        tiers[tier_for(n)].append({**upload, "line_count": n})
    for tier in tiers:
        tiers[tier].sort(key=lambda r: -r["line_count"])
    return tiers


# ---------------------------------------------------------------------------
# 2. 精读内容组装（0 LLM）
# ---------------------------------------------------------------------------

def _cameo_segments(content: str, target: str, window: int = CAME0_WINDOW) -> str:
    """客串文件：只取目标角色出现的片段，前后保留 window 行上下文。"""
    if not target:
        return content
    lines = content.splitlines()
    keep = set()
    for idx, ln in enumerate(lines):
        if ln.strip().startswith(target):
            for j in range(max(0, idx - window), min(len(lines), idx + window + 1)):
                keep.add(j)
    return "\n".join(lines[i] for i in sorted(keep)) if keep else ""


def _build_batch_prompt(uploads: List[dict], target: str, tier_label: str) -> str:
    parts = [f"[批次: {tier_label} 文件，共 {len(uploads)} 个]\n"]
    for upload in uploads:
        if tier_label == "cameo":
            body = _cameo_segments(upload.get("content") or "", target, CAME0_WINDOW)
        else:
            body = upload.get("content") or ""
        name = upload.get("name") or upload.get("file_url") or "uploaded-file"
        parts.append(
            f"\n===== 文件: {name}（目标角色台词 {upload['line_count']} 句）=====\n{body}"
        )
    return "\n".join(parts)


BATCH_NOTE_SYSTEM = (
    "你是角色分析师。阅读提供的文件批次，提取目标角色的结构化证据。\n"
    "每条证据必须带原文引用（file + 原句）。输出 JSON，键为：\n"
    "batch_summary(字符串), citations(数组, 每项 {file, quote, note}), "
    "personality_evidence(数组), language_style(数组), behavior_notes(数组), "
    "emotion_triggers(数组), relationships(数组)。"
)


def _produce_batch_notes(uploads, target, tier_label, llm_call) -> dict:
    prompt = _build_batch_prompt(uploads, target, tier_label)
    raw = llm_call(BATCH_NOTE_SYSTEM, prompt)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# 3. 合并（LLM）
# ---------------------------------------------------------------------------

MERGE_SYSTEM = (
    "你是角色档案总编辑。把各批次的结构化笔记合并成最终角色档案，处理前后期变化"
    "（分阶段呈现性格/语气演变）。输出 JSON，键为：\n"
    "profile_summary(对象: name/description/personality/appearance/affiliation/tags), "
    "dialogue_library(对象: 按 日常/提问/情绪/命令拒绝/玩笑 分类的台词样本数组, "
    "每项 {category, quote, file, note}), "
    "behavior_samples(数组, 每项 {scenario, behavior, file}), "
    "evolution(数组, 每项 {phase, summary})。"
)


def _merge_notes(batch_notes: list, llm_call) -> dict:
    payload = json.dumps({"batches": batch_notes}, ensure_ascii=False)
    raw = llm_call(MERGE_SYSTEM, payload)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# 4. 主入口
# ---------------------------------------------------------------------------

def run_reduce_pipeline(
    uploads: List[dict],
    target: str,
    llm_call: Callable[[str, str], str],
    batch_size: int = BATCH_SIZE,
) -> dict:
    """跑完整 reduce 流水线，返回最终档案 dict（可直接映射 PrisMateDraft）。"""
    tiers = _tier_uploads(uploads, target)

    batch_notes: List[dict] = []
    for tier in ("main", "mid", "cameo"):
        recs = tiers[tier]
        for i in range(0, len(recs), batch_size):
            batch = recs[i:i + batch_size]
            notes = _produce_batch_notes(batch, target, tier, llm_call)
            notes["_tier"] = tier
            notes["_files"] = [
                u.get("name") or u.get("file_url") for u in batch
            ]
            batch_notes.append(notes)

    merged = _merge_notes(batch_notes, llm_call)
    return {
        "target": target,
        "tier_counts": {t: len(r) for t, r in tiers.items()},
        "batch_count": len(batch_notes),
        "result": merged,
    }


# ---------------------------------------------------------------------------
# 5. 对齐 PrisMateDraft
# ---------------------------------------------------------------------------

DIALOGUE_CATEGORIES = ["日常", "提问", "情绪", "命令拒绝", "玩笑"]


def _build_example_dialogue(dialogue_library: Optional[dict]) -> str:
    """把 dialogue_library（5 分类台词样本）转成 example_dialogue 文本。

    PrisMateDraft.example_dialogue 约定每段为
    "User: <一句提问或陈述>\\nCharacter: <一句完整回答>"。
    原型阶段：把样本台词作为 Character 侧内容，用「Character: <quote>」段落
    呈现（用户侧留空提示）；后续接真模型时可让 reduce 直接生成完整对白。
    """
    if not dialogue_library:
        return ""
    samples: List[str] = []
    for category in DIALOGUE_CATEGORIES:
        items = dialogue_library.get(category) or []
        for item in items[:2]:
            quote = (item.get("quote") or "").strip()
            if not quote:
                continue
            file_ref = item.get("file") or ""
            source = f"（{file_ref}）" if file_ref else ""
            samples.append(f"User: <触发场景>{source}\nCharacter: {quote}")
    return "\n\n".join(samples)


def reduce_result_to_draft(pipeline_result: dict) -> dict:
    """把 reduce 流水线产出映射为 PrisMateDraft 兼容字段。"""
    result = pipeline_result.get("result") or {}
    summary = result.get("profile_summary") or {}

    example_dialogue = _build_example_dialogue(result.get("dialogue_library"))

    return {
        "name": (summary.get("name") or "Unknown").strip(),
        "description": (summary.get("description") or "").strip(),
        "personality": (summary.get("personality") or "").strip(),
        "appearance": (summary.get("appearance") or "").strip(),
        "affiliation": (summary.get("affiliation") or "").strip(),
        "tags": summary.get("tags") or [],
        "visual_summary": "",
        "example_dialogue": example_dialogue,
    }


def _normalize_target_name(text_context: Optional[str]) -> str:
    """从 text_context 里解析目标角色名。

    前端约定 text_context 含 "目标角色名: xxx" 行；取不到则返回空。
    """
    if not text_context:
        return ""
    m = re.search(r"目标角色名\s*[:：]\s*(\S+)", text_context)
    if m:
        return m.group(1).strip()
    m = re.search(r"target\s*character\s*name\s*[:：]\s*(\S+)", text_context, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""
