import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from .models import create_model


class Evaluator:
    """Computes evaluation metrics and uses LLM judge for solution quality."""

    def __init__(self, judge_model: str = "deepseek-chat", prompts_dir: Optional[str] = None):
        self.judge = create_model(judge_model)
        self.prompts_dir = Path(prompts_dir or Path(__file__).resolve().parent.parent / "prompts")
        self._judge_prompt = (self.prompts_dir / "judge.txt").read_text(encoding="utf-8")

    def judge_solution(self, truth: str, predicted: str) -> Tuple[int, str]:
        prompt = (
            self._judge_prompt
            .replace("{{TRUTH}}", truth.strip())
            .replace("{{PRED}}", predicted.strip())
        )
        out = self.judge.generate(prompt=prompt, temperature=0.0)
        try:
            obj = json.loads(out)
            score = 1 if int(obj.get("score", 0)) == 1 else 0
            reason = str(obj.get("reason", "")).strip()
            return score, reason
        except Exception:
            # Fallback heuristic: simple containment check
            truth_l = truth.lower()
            pred_l = predicted.lower()
            score = 1 if any(tok in pred_l for tok in truth_l.split()[:4]) else 0
            return score, "fallback"

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"success_rate": 0.0, "avg_questions": 0.0, "num_samples": 0}
        n = len(results)
        successes = sum(1 for r in results if int(r.get("score", 0)) == 1)
        avg_q = sum(int(r.get("num_questions", 0)) for r in results) / max(1, n)
        return {
            "num_samples": n,
            "success_rate": successes / n,
            "avg_questions": avg_q,
        }

    def compute_info_gain(self, conversation: List[Dict[str, Any]], hypotheses: List[str]) -> float:
        # Placeholder: compute diversity of answers and hypothesis count change
        if not conversation:
            return 0.0
        answers = [t.get("content") for t in conversation if t.get("type") == "answer"]
        unique = len(set(answers))
        return min(1.0, 0.1 * unique)

