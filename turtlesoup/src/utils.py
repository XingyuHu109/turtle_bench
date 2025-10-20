import os
import json
import csv
import logging
from typing import List, Dict, Any

from pathlib import Path
from datetime import datetime


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_puzzles(filepath: str) -> List[Dict[str, Any]]:
    """Load puzzles from JSONL file."""
    puzzles: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            puzzles.append(json.loads(line))
    return puzzles


def save_conversation(conversation: Dict[str, Any], output_dir: str) -> None:
    """Save single conversation as JSONL (one record per file)."""
    ensure_dir(output_dir)
    pid = conversation.get("puzzle_id", f"unknown_{int(datetime.now().timestamp())}")
    out_path = Path(output_dir) / f"{pid}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(conversation, ensure_ascii=False) + "\n")


def load_conversations(conversations_dir: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    p = Path(conversations_dir)
    if not p.exists():
        return items
    for file in p.glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
    return items


def get_puzzle(puzzles: List[Dict[str, Any]], puzzle_id: str) -> Dict[str, Any] | None:
    for p in puzzles:
        if p.get("id") == puzzle_id:
            return p
    return None


def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def setup_logging(output_dir: str) -> None:
    ensure_dir(output_dir)
    log_path = Path(output_dir) / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def convert_csv_to_jsonl(csv_path: str, jsonl_path: str, source_name: str = "sample") -> None:
    """Convert the provided CSV (Chinese columns) to JSONL puzzle format.

    CSV columns expected:
      - 题目 (title)
      - 汤面 (scenario)
      - 汤底 (truth)
    """
    ensure_dir(Path(jsonl_path).parent)
    with open(csv_path, "r", encoding="utf-8") as f_in, open(jsonl_path, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for i, row in enumerate(reader, start=1):
            scenario = (row.get("汤面") or "").strip()
            truth = (row.get("汤底") or "").strip()
            title = (row.get("题目") or f"puzzle_{i}").strip()
            if not scenario or not truth:
                continue
            record = {
                "id": f"puzzle_{i:03d}",
                "title": title,
                "scenario": scenario,
                "truth": truth,
                "category": "unknown",
                "source": source_name,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")


def simple_deduplicate(puzzles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A simple deduplication by scenario text normalization."""
    seen = set()
    out = []
    for p in puzzles:
        key = (p.get("scenario") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

