import argparse
import os
from pathlib import Path
from datetime import datetime
import sys

# Ensure repository root is on sys.path when running as a script
sys.path.append(str(Path(__file__).resolve().parents[2]))

from turtlesoup.src.utils import load_puzzles, save_conversation, setup_logging
from turtlesoup.src.oracle import Oracle
from turtlesoup.src.solvers import create_solver
from turtlesoup.src.ui import get_console


def main():
    ap = argparse.ArgumentParser(description="Run a TurtleSoup solver on puzzles")
    ap.add_argument("--solver", required=True, choices=["naive", "hypothesis", "infogain"], help="Solver type")
    ap.add_argument("--model", default="deepseek-chat", help="LLM model name (mapped to DeepSeek)")
    ap.add_argument("--oracle_model", default="deepseek-chat", help="Oracle model (DeepSeek)")
    ap.add_argument("--puzzles", required=True, help="Path to puzzles JSONL (e.g., data/splits/dev.jsonl)")
    ap.add_argument("--output", required=True, help="Output directory for conversations")
    ap.add_argument("--max_questions", type=int, default=30)
    ap.add_argument("--num_candidates", type=int, default=5, help="InfoGain candidate questions")
    args = ap.parse_args()

    setup_logging(args.output)
    ui = get_console()

    ui.rule("TurtleSoup Run")
    ui.box("Config", f"solver={args.solver}\nmodel={args.model}\noracle={args.oracle_model}\nmax_questions={args.max_questions}\npuzzles={args.puzzles}\noutput={args.output}")

    puzzles = load_puzzles(args.puzzles)
    oracle = Oracle(args.oracle_model)

    solver_kwargs = {"num_candidates": args.num_candidates} if args.solver == "infogain" else {}
    solver = create_solver(args.solver, args.model, **solver_kwargs)

    total_puzzles = len(puzzles)
    ui.info(f"Loaded {total_puzzles} puzzles. Starting...")

    for i, puzzle in enumerate(puzzles, start=1):
        scenario = puzzle.get("scenario", "").strip()
        truth = puzzle.get("truth", "").strip()
        pid = puzzle.get("id", f"puzzle_{i:03d}")
        if not scenario or not truth:
            continue
        ui.box("Solving", f"{pid}: {puzzle.get('title', '')}")

        solver.reset()

        # Per-puzzle progress
        def on_turn(turn_no, question, answer):
            ui.progress_advance(1, description=f"{pid} Q{turn_no}")
            import os as _os
            if _os.getenv("TS_PRINT_QA", "0") not in {"0", ""}:
                ui.info(f"Q{turn_no}: {question[:100].replace('\n',' ')}")
                ui.info(f"A{turn_no}: {answer}")

        ui.progress_start(args.max_questions, description=f"{pid} Qs")
        final_solution = solver.solve(scenario, truth, oracle, max_questions=args.max_questions, on_turn=on_turn)
        ui.progress_stop()
        conv = {
            "puzzle_id": pid,
            "model": args.model,
            "solver": args.solver,
            "conversation": solver.conversation,
            "final_solution": final_solution,
            "num_questions": len([t for t in solver.conversation if t.get("type") == "question"]),
            "timestamp": datetime.utcnow().isoformat(),
        }
        save_conversation(conv, args.output)
        ui.info(f"Saved {args.output}/{pid}.jsonl  (completed {i}/{total_puzzles})")
    ui.rule("Run Complete")


if __name__ == "__main__":
    main()
