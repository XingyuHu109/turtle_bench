import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path when running as a script
sys.path.append(str(Path(__file__).resolve().parents[2]))

from turtlesoup.src.utils import load_conversations, load_puzzles, get_puzzle, save_metrics
from turtlesoup.src.evaluator import Evaluator
from turtlesoup.src.ui import get_console


def main():
    ap = argparse.ArgumentParser(description="Evaluate TurtleSoup conversations")
    ap.add_argument("--conversations", required=True, help="Directory with conversation JSONL files")
    ap.add_argument("--puzzles", required=True, help="Path to puzzles JSONL used for the run")
    ap.add_argument("--output", required=True, help="Path to save metrics JSON")
    ap.add_argument("--judge_model", default="deepseek-chat")
    args = ap.parse_args()

    ui = get_console()
    ui.rule("TurtleSoup Evaluation")
    ui.box("Config", f"conversations={args.conversations}\npuzzles={args.puzzles}\njudge_model={args.judge_model}")

    conversations = load_conversations(args.conversations)
    puzzles = load_puzzles(args.puzzles)
    evaluator = Evaluator(args.judge_model)

    results = []
    ui.info(f"Loaded {len(conversations)} conversations. Judging...")
    ui.progress_start(len(conversations), description="Conversations")
    for conv in conversations:
        puzzle = get_puzzle(puzzles, conv.get("puzzle_id"))
        if not puzzle:
            continue
        score, reasoning = evaluator.judge_solution(puzzle.get("truth", ""), conv.get("final_solution", ""))
        results.append({
            "puzzle_id": conv.get("puzzle_id"),
            "score": score,
            "num_questions": int(conv.get("num_questions", 0)),
            "judge_reasoning": reasoning,
        })
        ui.progress_advance(1, description=f"{conv.get('puzzle_id')}")

    ui.progress_stop()
    metrics = evaluator.compute_metrics(results)
    save_metrics(metrics, args.output)
    ui.box("Metrics", f"num_samples={metrics.get('num_samples')}\nsuccess_rate={metrics.get('success_rate')}\navg_questions={metrics.get('avg_questions')}")
    ui.rule("Evaluation Complete")


if __name__ == "__main__":
    main()
