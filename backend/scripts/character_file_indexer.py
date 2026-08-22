#!/usr/bin/env python3
"""character_file_indexer.py — 纯规则「map 阶段」原型（统一平铺版）。

用户把所有相关文件一股脑拖进上传框，系统不做任何精细分类：
每个文件统一产出 {file, content} 一条记录，文件名即引用键。

用法:
    python character_file_indexer.py <输入目录> [输出目录]

只依赖标准库，可直接用系统 python3 运行。
"""
import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="纯规则角色基础文件 map 阶段原型（统一平铺）")
    parser.add_argument("input_dir", help="角色基础文件目录（递归扫描 .txt）")
    parser.add_argument("output_dir", nargs="?", default="profile_out")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        print(f"输入目录不存在: {in_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(p for p in in_dir.rglob("*.txt") if p.is_file())
    records = []

    for path in txt_files:
        rel = str(path.relative_to(in_dir)).replace("\\", "/")
        try:
            text = read_text(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {rel}: {exc}")
            continue

        content = text.strip()

        record = {
            "file": rel,
            "chars": len(content),
            "content": content,
        }
        records.append(record)

        out_path = out_dir / (rel.replace("/", "__").replace("\\", "__") + ".json")
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    total_chars = sum(r["chars"] for r in records)
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps({
        "source_dir": str(in_dir),
        "total_files": len(records),
        "total_chars": total_chars,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done: {len(records)} files -> {out_dir}")
    print(f"  total_files={len(records)}  total_chars={total_chars} (全文保留)")
    print(f"  index: {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
