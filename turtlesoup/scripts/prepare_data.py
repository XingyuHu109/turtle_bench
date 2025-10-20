import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path when running as a script
sys.path.append(str(Path(__file__).resolve().parents[2]))

from turtlesoup.src.utils import (
    load_puzzles,
    simple_deduplicate,
    convert_csv_to_jsonl,
)
from turtlesoup.src.ui import get_console


def deduplicate(puzzles: list) -> list:
    return simple_deduplicate(puzzles)


def filter_quality(puzzles: list, model: str | None = None) -> list:
    # Placeholder: trivial length filter. Extend with LLM-based scoring if desired.
    out = []
    for p in puzzles:
        if len((p.get("scenario") or "").strip()) >= 10 and len((p.get("truth") or "").strip()) >= 10:
            out.append(p)
    return out


def split_data(puzzles: list, train: float, dev: float, test: float):
    n = len(puzzles)
    n_train = int(n * train)
    n_dev = int(n * dev)
    train_set = puzzles[:n_train]
    dev_set = puzzles[n_train:n_train + n_dev]
    test_set = puzzles[n_train + n_dev:]
    return train_set, dev_set, test_set


def write_jsonl(items: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            import json
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Prepare TurtleSoup data")
    ap.add_argument("--input", required=True, help="Input CSV or JSONL file")
    ap.add_argument("--output", required=True, help="Output directory for filtered data")
    ap.add_argument("--splits", default="data/splits", help="Directory to save splits")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--dev", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    args = ap.parse_args()

    ui = get_console()
    ui.rule("TurtleSoup Data Prep")
    ui.box("Config", f"input={args.input}\noutput={args.output}\nsplits={args.splits}\ntrain={args.train} dev={args.dev} test={args.test}")

    inp = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize to JSONL first if CSV
    if inp.suffix.lower() == ".csv":
        raw_path = out_dir / "puzzles.jsonl"
        ui.info("Converting CSV to JSONL...")
        convert_csv_to_jsonl(str(inp), str(raw_path), source_name=inp.name)
    else:
        raw_path = inp

    ui.info("Loading puzzles...")
    puzzles = load_puzzles(str(raw_path))
    ui.info(f"Loaded {len(puzzles)}. Deduplicating...")
    puzzles = deduplicate(puzzles)
    ui.info(f"After dedup: {len(puzzles)}. Quality filtering...")
    puzzles = filter_quality(puzzles)
    ui.info(f"After filter: {len(puzzles)}")

    # renumber ids after filtering for simplicity
    for i, p in enumerate(puzzles, start=1):
        p["id"] = f"puzzle_{i:03d}"

    filtered_path = out_dir / "filtered.jsonl"
    write_jsonl(puzzles, str(filtered_path))

    train_set, dev_set, test_set = split_data(puzzles, args.train, args.dev, args.test)
    write_jsonl(train_set, str(Path(args.splits) / "train.jsonl"))
    write_jsonl(dev_set, str(Path(args.splits) / "dev.jsonl"))
    write_jsonl(test_set, str(Path(args.splits) / "test.jsonl"))
    ui.box("Data Written", f"filtered={filtered_path}\ntrain/dev/test -> {args.splits}")


if __name__ == "__main__":
    main()
