from pathlib import Path
from typing import Optional

from .models import create_model


class Oracle:
    """Answers yes/no/irrelevant to questions based on ground truth."""

    def __init__(self, model_name: str = "deepseek-chat", prompts_dir: Optional[str] = None):
        self.model = create_model(model_name)
        self.prompts_dir = Path(prompts_dir or Path(__file__).resolve().parent.parent / "prompts")
        self._oracle_prompt = (self.prompts_dir / "oracle.txt").read_text(encoding="utf-8")

    @staticmethod
    def _normalize(ans: str) -> str:
        s = (ans or "").strip().lower()
        if "yes" in s or s == "y":
            return "yes"
        if "no" in s or s == "n":
            return "no"
        if "irrelevant" in s or "unknown" in s or "not" in s and "determin" in s:
            return "irrelevant"
        # default fallback when uncertain
        # keep it conservative to avoid false positives
        return "irrelevant"

    def answer(self, question: str, truth: str, temperature: float = 0.0) -> str:
        prompt = (
            self._oracle_prompt
            .replace("{{TRUTH}}", truth.strip())
            .replace("{{QUESTION}}", question.strip())
        )
        raw = self.model.generate(prompt=prompt, temperature=temperature)
        return self._normalize(raw)

