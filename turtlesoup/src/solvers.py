import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import create_model


def _format_conversation(conv: List[Dict[str, str]]) -> str:
    lines = []
    for turn in conv:
        t = turn.get("type")
        content = turn.get("content", "").strip()
        turn_no = turn.get("turn")
        lines.append(f"[{turn_no}:{t}] {content}")
    return "\n".join(lines)


class BaseSolver:
    """Base class for all solvers."""

    def __init__(self, model_name: str = "deepseek-chat", prompts_dir: Optional[str] = None):
        self.model = create_model(model_name)
        self.prompts_dir = Path(prompts_dir or Path(__file__).resolve().parent.parent / "prompts")
        self.conversation: List[Dict[str, Any]] = []
        self.finalize_turn: Optional[int] = None
        self.finalized: bool = False

        # Prompts
        self._baseline_prompt = (self.prompts_dir / "solver_baseline.txt").read_text(encoding="utf-8")
        # Decide whether to finalize now
        stop_path = self.prompts_dir / "solver_stop.txt"
        if stop_path.exists():
            self._stop_prompt = stop_path.read_text(encoding="utf-8")
        else:
            self._stop_prompt = (
                "You judge if we should stop asking yes/no questions and write the final solution now.\n"
                "Return exactly one token: yes or no.\n\n"
                "SCENARIO:\n{{SCENARIO}}\n\nCONVERSATION:\n{{CONV}}\n\nANSWER:"
            )

    def reset(self) -> None:
        self.conversation = []
        self.finalize_turn = None
        self.finalized = False

    def generate_question(self, scenario: str) -> str:
        """Generate next question. Override in subclasses for advanced behavior."""
        conv_text = _format_conversation(self.conversation)
        prompt = (
            self._baseline_prompt
            .replace("{{SCENARIO}}", scenario.strip())
            .replace("{{CONV}}", conv_text)
        )
        question = self.model.generate(prompt=prompt, temperature=0.7).strip()
        return question

    def propose_solution(self, scenario: str) -> str:
        # Leverage the same model: ask for final solution summary based on conversation
        conv_text = _format_conversation(self.conversation)
        prompt = (
            "You are the solver. Given the scenario and the full Q/A conversation, "
            "propose your best final solution in 2-4 sentences.\n\n"
            f"SCENARIO:\n{scenario}\n\nCONVERSATION (latest last):\n{conv_text}\n\nFINAL SOLUTION:" 
        )
        return self.model.generate(prompt=prompt, temperature=0.2).strip()

    def should_finalize(self, scenario: str) -> bool:
        conv_text = _format_conversation(self.conversation)
        prompt = (
            self._stop_prompt
            .replace("{{SCENARIO}}", scenario.strip())
            .replace("{{CONV}}", conv_text)
        )
        out = self.model.generate(prompt=prompt, temperature=0.0).strip().lower()
        return out.startswith("yes")

    def solve(self, scenario: str, truth: str, oracle, max_questions: int = 30, on_turn=None) -> str:
        """Main solving loop. Returns final solution."""
        self.reset()
        for i in range(1, max_questions + 1):
            q = self.generate_question(scenario)
            self.conversation.append({"type": "question", "content": q, "turn": i})
            a = oracle.answer(q, truth)
            self.conversation.append({"type": "answer", "content": a, "turn": i})
            if on_turn:
                try:
                    on_turn(i, q, a)
                except Exception:
                    pass
            # Optional early stop if question indicates finalization
            if q.strip().lower() in {"is that enough to solve?", "is the solution clear now?"}:
                break
            # Model-driven early stopping decision
            if self.should_finalize(scenario):
                self.finalized = True
                self.finalize_turn = i
                break
        return self.propose_solution(scenario)


class NaiveSolver(BaseSolver):
    """Baseline: conversation history only."""
    pass


class HypothesisSolver(BaseSolver):
    """Maintains explicit hypotheses."""

    def __init__(self, model_name: str = "deepseek-chat", num_hypotheses: int = 5, prompts_dir: Optional[str] = None):
        super().__init__(model_name=model_name, prompts_dir=prompts_dir)
        self.num_hypotheses = num_hypotheses
        self._hypo_prompt = (self.prompts_dir / "solver_hypothesis.txt").read_text(encoding="utf-8")
        self.hypotheses: List[str] = []

    def _call_hypothesis_planner(self, scenario: str) -> Dict[str, Any]:
        conv_text = _format_conversation(self.conversation)
        hypos_text = json.dumps(self.hypotheses, ensure_ascii=False)
        prompt = (
            self._hypo_prompt
            .replace("{{SCENARIO}}", scenario.strip())
            .replace("{{CONV}}", conv_text)
            .replace("{{HYPOS}}", hypos_text)
        )
        out = self.model.generate(prompt=prompt, temperature=0.3)
        try:
            parsed = json.loads(out)
        except Exception:
            parsed = {"hypotheses": self.hypotheses, "question": out.strip()}
        return parsed

    def generate_hypotheses(self, scenario: str) -> List[str]:
        parsed = self._call_hypothesis_planner(scenario)
        hypos = parsed.get("hypotheses") or []
        # keep only top N non-empty
        clean = []
        for h in hypos:
            hs = (h or "").strip()
            if hs and hs not in clean:
                clean.append(hs)
            if len(clean) >= self.num_hypotheses:
                break
        self.hypotheses = clean
        return self.hypotheses

    def update_hypotheses(self, question: str, answer: str) -> None:
        if not self.hypotheses:
            return
        # Simple pruning rule via LLM check could be expensive; use heuristic placeholder:
        # If answer is 'irrelevant', deprioritize hypos referencing the question's specific nouns.
        if answer == "irrelevant":
            key = question.lower().split(" ")[0:1]
            self.hypotheses = [h for h in self.hypotheses if all(k not in h.lower() for k in key)]

    def generate_question(self, scenario: str) -> str:
        if not self.hypotheses:
            self.generate_hypotheses(scenario)
        parsed = self._call_hypothesis_planner(scenario)
        # Update internal hypotheses from the tool output
        hypos = parsed.get("hypotheses") or []
        if hypos:
            self.hypotheses = hypos
        q = (parsed.get("question") or "").strip()
        return q or super().generate_question(scenario)


class InfoGainSolver(HypothesisSolver):
    """Information-theoretic question selection (LLM-prioritized)."""

    def __init__(self, model_name: str = "deepseek-chat", num_candidates: int = 5, prompts_dir: Optional[str] = None):
        super().__init__(model_name=model_name, prompts_dir=prompts_dir)
        self.num_candidates = num_candidates
        self._ig_prompt = (self.prompts_dir / "solver_infogain.txt").read_text(encoding="utf-8")

    def generate_candidates(self, scenario: str, n: int = 5) -> List[Dict[str, Any]]:
        conv_text = _format_conversation(self.conversation)
        hypos_text = json.dumps(self.hypotheses, ensure_ascii=False)
        prompt = (
            self._ig_prompt
            .replace("{{SCENARIO}}", scenario.strip())
            .replace("{{CONV}}", conv_text)
            .replace("{{HYPOS}}", hypos_text)
            .replace("{{N}}", str(n))
        )
        out = self.model.generate(prompt=prompt, temperature=0.4)
        try:
            arr = json.loads(out)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
        # Fallback: single question
        return [{"question": out.strip(), "info_gain": 0.5}]

    def estimate_info_gain(self, question: str, hypotheses: List[str]) -> float:
        # Simple heuristic fallback: prefer questions that mention different keywords
        # In practice, the LLM already rated candidates; this is a backup.
        uniq = len(set(question.lower().split()))
        return min(1.0, 0.1 * max(1, uniq // 3))

    def generate_question(self, scenario: str) -> str:
        if not self.hypotheses:
            self.generate_hypotheses(scenario)
        candidates = self.generate_candidates(scenario, n=self.num_candidates)
        # Choose by provided info_gain if available
        best_q = None
        best_ig = -1.0
        for c in candidates:
            q = (c.get("question") or "").strip()
            ig = c.get("info_gain")
            if ig is None:
                ig = self.estimate_info_gain(q, self.hypotheses)
            try:
                ig = float(ig)
            except Exception:
                ig = 0.0
            if q and ig > best_ig:
                best_q, best_ig = q, ig
        return best_q or super().generate_question(scenario)


def create_solver(name: str, model_name: str = "deepseek-chat", **kwargs) -> BaseSolver:
    name = (name or "").strip().lower()
    if name in {"naive", "baseline"}:
        return NaiveSolver(model_name=model_name)
    if name in {"hypothesis", "hypos"}:
        return HypothesisSolver(model_name=model_name, **kwargs)
    if name in {"infogain", "ig"}:
        num_candidates = kwargs.get("num_candidates", 5)
        return InfoGainSolver(model_name=model_name, num_candidates=num_candidates)
    # default
    return NaiveSolver(model_name=model_name)
