"""run_character_reduce — 用真实模型配置跑 reduce 流水线。

    python manage.py run_character_reduce <user_id> <index.json> [目标角色名] [输出目录]

复用用户 DB 里的 text 角色模型配置（ModelRoleAssignment），通过
chat.tasks._generate_text 调用真实模型完成「分批精读笔记 → 汇总合并」。
"""
import json
import sys
from pathlib import Path

from django.core.management.base import BaseCommand

from chat.models import ModelRole, ModelRoleAssignment
from chat.tasks import _extract_json_object, _generate_text, _model_config_to_runtime


def _make_llm_call(runtime_config):
    """把 reducer 的 llm_call 绑定到真实 _generate_text。"""
    def llm_call(system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = _generate_text(runtime_config, messages)
        data = _extract_json_object(raw)
        if not data:
            raise ValueError(
                "Model did not return a JSON object for the reduce step. "
                f"Raw output (truncated): {raw[:300]!r}"
            )
        return json.dumps(data, ensure_ascii=False)
    return llm_call


class Command(BaseCommand):
    help = "Run the character reduce pipeline (tier -> batch notes -> merge) with the real model."

    def add_arguments(self, parser):
        parser.add_argument("user_id", type=int)
        parser.add_argument("index_json")
        parser.add_argument("target", nargs="?", default="圣亚")
        parser.add_argument("output_dir", nargs="?", default="reduce_out")

    def handle(self, *args, **options):
        from scripts.character_reducer import (
            BATCH_NOTE_SYSTEM,
            MERGE_SYSTEM,
            main as _unused_main,  # noqa: F401 (keep module importable)
            produce_batch_notes,
            tier_files,
        )

        user_id = options["user_id"]
        index_path = Path(options["index_json"])
        target = options["target"]
        out_dir = Path(options["output_dir"])

        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            self.stderr.write(f"User {user_id} does not exist.")
            sys.exit(1)
        model_config = ModelRoleAssignment.get_role_config(user, ModelRole.TEXT)
        if not model_config:
            self.stderr.write(f"User {user_id} has no TEXT model configuration.")
            sys.exit(1)
        runtime_config = _model_config_to_runtime(model_config)
        self.stdout.write(f"Using model: {runtime_config['provider']} / {runtime_config['model_name']}")

        data = json.loads(index_path.read_text(encoding="utf-8"))
        records = data["records"]

        tiers = tier_files(records, target)
        for tier, recs in tiers.items():
            self.stdout.write(f"  tier {tier}: {len(recs)} files")

        # 分批精读 → 笔记（真实 LLM）
        batch_notes = []
        for tier in ("main", "mid", "cameo"):
            recs = tiers[tier]
            for i in range(0, len(recs), 6):
                batch = recs[i:i + 6]
                notes = produce_batch_notes(
                    batch, target, tier,
                    llm_call=_make_llm_call(runtime_config),
                )
                notes["_tier"] = tier
                notes["_files"] = [r["file"] for r in batch]
                batch_notes.append(notes)
                self.stdout.write(f"  [batch] {tier} #{i // 6 + 1}: {len(batch)} files done")

        # 合并（真实 LLM）
        self.stdout.write("Merging batch notes...")
        merge_messages = [
            {"role": "system", "content": MERGE_SYSTEM},
            {"role": "user", "content": json.dumps({"batches": batch_notes}, ensure_ascii=False)},
        ]
        raw = _generate_text(runtime_config, merge_messages)
        merged = _extract_json_object(raw)
        if not merged:
            self.stderr.write("Merge step did not return JSON.")
            sys.exit(1)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "character_profile.json"
        out_path.write_text(json.dumps({
            "target": target,
            "source_index": str(index_path),
            "tier_counts": {t: len(r) for t, r in tiers.items()},
            "batch_count": len(batch_notes),
            "result": merged,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        self.stdout.write(f"Final profile: {out_path}")
