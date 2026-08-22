#!/usr/bin/env python3
"""character_reducer.py — reduce 阶段原型（分层 → 精读 → 笔记 → 合并）。

输入: map 阶段产出的 index.json（每文件 {file, chars, content} 全文）。
流程:
  1. 按戏份分层（纯规则，0 LLM）: 重头/中/客串
  2. 分批精读（0 LLM 组装 prompt）: 重头完整读；客串只取目标角色出现片段
  3. 每批 LLM 产出结构化笔记（带原文引用）: 性格证据/语言风格/行为演出/情绪触发点/关系网
  4. 汇总合并（LLM）: 处理前后期变化 → 角色属性总结 + 分类台词样本库 + 行为样本

LLM 调用统一走 llm_call()，当前为 mock（返回占位结构化数据）。
接入真实模型时只需替换 llm_call() 内部实现为 _generate_text 调用。

用法:
    python character_reducer.py <index.json> [目标角色名] [输出目录]
"""
import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
TIER_MAIN = 30      # 重头: 目标角色台词 >= 30 句
TIER_MID = 5        # 中: 5~29 句；客串: < 5 句
CAME0_WINDOW = 3    # 客串片段: 台词前后各 N 行（保上下文、控 token）
BATCH_SIZE = 6      # 每批精读的文件数（重头按批，客串按批）

MOCK_LLM = True     # True = mock 占位；False = 走真实 _generate_text（需先接好）


def mock_llm_call(system_prompt: str, user_prompt: str) -> str:
    """mock 占位：返回结构合法的假笔记。"""
    return json.dumps({
        "batch_summary": "（mock）本批包含角色在该阶段的对话与互动。",
        "citations": [
            {"file": "示例文件.txt", "quote": "示例台词", "note": "（mock）性格证据占位"},
        ],
        "personality_evidence": ["（mock）语气特征占位"],
        "language_style": ["（mock）口头禅/句式占位"],
        "behavior_notes": ["（mock）行为演出占位"],
        "emotion_triggers": ["（mock）情绪触发点占位"],
        "relationships": ["（mock）关系网占位"],
    }, ensure_ascii=False)


def llm_call(system_prompt: str, user_prompt: str) -> str:
    """统一 LLM 入口。mock 模式返回占位；接入真实模型时替换此函数。"""
    return mock_llm_call(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# 1. 分层（纯规则，0 LLM）
# ---------------------------------------------------------------------------

def count_target_lines(content: str, target: str) -> int:
    return sum(1 for ln in content.splitlines() if ln.strip().startswith(target))


def tier_for(line_count: int) -> str:
    if line_count >= TIER_MAIN:
        return "main"
    if line_count >= TIER_MID:
        return "mid"
    return "cameo"


def tier_files(records, target):
    """返回 {tier: [record]}，每记录附 line_count。"""
    tiers = {"main": [], "mid": [], "cameo": []}
    for record in records:
        n = count_target_lines(record["content"], target)
        tiers[tier_for(n)].append({**record, "line_count": n})
    for tier in tiers:
        tiers[tier].sort(key=lambda r: -r["line_count"])
    return tiers


# ---------------------------------------------------------------------------
# 2. 精读内容组装（0 LLM）
# ---------------------------------------------------------------------------

def cameo_segments(content: str, target: str, window: int = CAME0_WINDOW) -> str:
    """客串文件：只取目标角色出现的片段，前后保留 window 行上下文。"""
    lines = content.splitlines()
    keep = set()
    for idx, ln in enumerate(lines):
        if ln.strip().startswith(target):
            for j in range(max(0, idx - window), min(len(lines), idx + window + 1)):
                keep.add(j)
    return "\n".join(lines[i] for i in sorted(keep)) if keep else ""


def build_batch_prompt(records, target, tier_label: str) -> str:
    """组装一个精读批次的 prompt。重头=完整原文；客串=片段。"""
    parts = [f"[批次: {tier_label} 文件，共 {len(records)} 个]\n"]
    for record in records:
        if tier_label == "main":
            body = record["content"]
        elif tier_label == "mid":
            body = record["content"]  # 中量也全文，但文件本就小
        else:
            body = cameo_segments(record["content"], target, CAME0_WINDOW)
        parts.append(f"\n===== 文件: {record['file']}（目标角色台词 {record['line_count']} 句）=====\n{body}")
    return "\n".join(parts)


BATCH_NOTE_SYSTEM = (
    "你是角色分析师。阅读提供的文件批次，提取目标角色的结构化证据。\n"
    "每条证据必须带原文引用（file + 原句）。输出 JSON，键为：\n"
    "batch_summary(字符串), citations(数组, 每项 {file, quote, note}), "
    "personality_evidence(数组), language_style(数组), behavior_notes(数组), "
    "emotion_triggers(数组), relationships(数组)。"
)


def produce_batch_notes(records, target, tier_label: str, llm_call=llm_call) -> dict:
    prompt = build_batch_prompt(records, target, tier_label)
    return json.loads(llm_call(BATCH_NOTE_SYSTEM, prompt))


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


def merge_notes(batch_notes: list, llm_call=llm_call) -> dict:
    merged = {"batches": batch_notes}
    return json.loads(llm_call(MERGE_SYSTEM, json.dumps(merged, ensure_ascii=False)))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="reduce 阶段原型")
    parser.add_argument("index_json", help="map 阶段产出的 index.json")
    parser.add_argument("target", nargs="?", default="圣亚", help="目标角色名")
    parser.add_argument("output_dir", nargs="?", default="reduce_out")
    args = parser.parse_args()

    index_path = Path(args.index_json)
    if not index_path.is_file():
        print(f"index.json 不存在: {index_path}", file=sys.stderr)
        return 1
    data = json.loads(index_path.read_text(encoding="utf-8"))
    records = data["records"]
    target = args.target

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 分层
    tiers = tier_files(records, target)
    print(f"分层（目标角色: {target}）:")
    for tier, recs in tiers.items():
        print(f"  {tier}: {len(recs)} 个文件")

    # 2+3. 分批精读 → 每批笔记
    batch_notes = []
    for tier in ("main", "mid", "cameo"):
        recs = tiers[tier]
        for i in range(0, len(recs), BATCH_SIZE):
            batch = recs[i:i + BATCH_SIZE]
            notes = produce_batch_notes(batch, target, tier)
            notes["_tier"] = tier
            notes["_files"] = [r["file"] for r in batch]
            batch_notes.append(notes)
            print(f"  [批次] {tier} 第{i // BATCH_SIZE + 1}批: {len(batch)} 文件 → 笔记完成")

    # 4. 合并
    print("合并批次笔记...")
    final = merge_notes(batch_notes)

    out_path = out_dir / "character_profile.json"
    out_path.write_text(json.dumps({
        "target": target,
        "source_index": str(index_path),
        "tier_counts": {t: len(r) for t, r in tiers.items()},
        "batch_count": len(batch_notes),
        "result": final,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"最终档案: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
