# TurtleSoup Benchmark - Design Document

## Overview

Research benchmark for evaluating LLMs' strategic question-asking and lateral thinking abilities through 海龟汤 (lateral thinking puzzles).

## Project Structure

```
turtlesoup/
├── data/
│   ├── raw/              # Raw scraped puzzles
│   ├── filtered/         # After deduplication and quality filter
│   └── splits/           # train/dev/test splits
├── src/
│   ├── oracle.py         # Oracle implementation
│   ├── solvers.py        # All solver implementations
│   ├── evaluator.py      # Metrics and LLM judge
│   ├── models.py         # LLM API wrappers
│   └── utils.py          # Data loading, logging
├── scripts/
│   ├── prepare_data.py   # Data acquisition and filtering
│   ├── run_solver.py     # Run single solver on puzzles
│   └── evaluate.py       # Compute all metrics
├── prompts/
│   ├── oracle.txt        # Oracle system prompt
│   ├── solver_baseline.txt
│   ├── solver_hypothesis.txt
│   ├── solver_infogain.txt
│   └── judge.txt         # LLM judge prompt
├── configs/
│   └── config.yaml       # Experiment configurations
├── results/              # Experiment outputs
└── requirements.txt
```

## Data Format

**Puzzle Format** (JSONL):
```json
{
  "id": "puzzle_001",
  "scenario": "A man walks into a restaurant...",
  "truth": "He was a sailor who survived...",
  "category": "logic",
  "source": "website_name"
}
```

**Conversation Log** (JSONL):
```json
{
  "puzzle_id": "puzzle_001",
  "model": "gpt-4-turbo",
  "solver": "hypothesis",
  "conversation": [
    {"type": "question", "content": "Was this recent?", "turn": 1},
    {"type": "answer", "content": "irrelevant", "turn": 1},
    ...
  ],
  "final_solution": "The man was...",
  "num_questions": 23,
  "timestamp": "2025-01-20T10:30:00"
}
```

**Evaluation Results** (JSONL):
```json
{
  "puzzle_id": "puzzle_001",
  "model": "gpt-4-turbo",
  "solver": "hypothesis",
  "score": 1,
  "num_questions": 23,
  "judge_reasoning": "Captures core twist..."
}
```

## Core Components

### 1. Oracle (oracle.py)

```python
class Oracle:
    """
    Answers yes/no/irrelevant to questions based on ground truth.
    """
    
    def __init__(self, model_name: str):
        self.model = create_model(model_name)
    
    def answer(self, question: str, truth: str) -> str:
        """
        Returns: "yes", "no", or "irrelevant"
        """
        pass
```

**Design Notes:**
- Single class, stateless
- Uses LLM (GPT-4 or Claude) with ground truth as context
- Temperature = 0 for deterministic answers
- Prompt loaded from prompts/oracle.txt

### 2. Solvers (solvers.py)

```python
class BaseSolver:
    """Base class for all solvers."""
    
    def __init__(self, model_name: str):
        self.model = create_model(model_name)
        self.conversation = []
    
    def solve(self, scenario: str, oracle: Oracle, max_questions: int = 50) -> str:
        """
        Main solving loop. Returns final solution.
        """
        pass
    
    def generate_question(self, scenario: str) -> str:
        """Generate next question. Override in subclasses."""
        pass

class NaiveSolver(BaseSolver):
    """Baseline: conversation history only."""
    pass

class HypothesisSolver(BaseSolver):
    """Maintains explicit hypotheses."""
    
    def __init__(self, model_name: str, num_hypotheses: int = 5):
        super().__init__(model_name)
        self.hypotheses = []
    
    def generate_hypotheses(self, scenario: str) -> list:
        pass
    
    def update_hypotheses(self, question: str, answer: str):
        pass

class InfoGainSolver(BaseSolver):
    """Information-theoretic question selection."""
    
    def generate_candidates(self, scenario: str, n: int = 5) -> list:
        pass
    
    def estimate_info_gain(self, question: str, hypotheses: list) -> float:
        pass
```

**Design Notes:**
- Inheritance hierarchy: BaseSolver → specific solvers
- Each solver is self-contained
- Conversation history stored in solver instance
- Prompts loaded from prompts/ directory

### 3. Evaluator (evaluator.py)

```python
class Evaluator:
    """Computes all evaluation metrics."""
    
    def __init__(self, judge_model: str = "gpt-4-turbo"):
        self.judge = create_model(judge_model)
    
    def judge_solution(self, truth: str, predicted: str) -> tuple[int, str]:
        """
        Returns: (score, reasoning)
        score: 0 or 1
        """
        pass
    
    def compute_metrics(self, results: list) -> dict:
        """
        Computes SR, QTS, SE, etc.
        Returns dict of metrics.
        """
        pass
    
    def compute_info_gain(self, conversation: list, hypotheses: list) -> float:
        """For hypothesis-driven solver only."""
        pass
```

**Design Notes:**
- Single class for all metrics
- LLM judge for solution evaluation
- Statistical metrics computed from results files

### 4. Model Wrappers (models.py)

```python
def create_model(model_name: str):
    """Factory function to create model instances."""
    if "gpt" in model_name:
        return OpenAIModel(model_name)
    elif "claude" in model_name:
        return AnthropicModel(model_name)
    # etc.

class BaseModel:
    """Abstract base for all models."""
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        pass

class OpenAIModel(BaseModel):
    pass

class AnthropicModel(BaseModel):
    pass
```

**Design Notes:**
- Simple wrapper around API calls
- Unified interface for all models
- Handle retries and rate limits
- API keys from environment variables

### 5. Utilities (utils.py)

```python
def load_puzzles(filepath: str) -> list:
    """Load puzzles from JSONL."""
    pass

def save_conversation(conversation: dict, output_dir: str):
    """Save conversation log."""
    pass

def setup_logging(output_dir: str):
    """Configure logging."""
    pass
```

## Scripts

### prepare_data.py

```python
# Usage: python scripts/prepare_data.py --input data/raw --output data/filtered

def deduplicate(puzzles: list) -> list:
    """Remove duplicates using fuzzy matching."""
    pass

def filter_quality(puzzles: list, model: str) -> list:
    """Use LLM to rate puzzle quality."""
    pass

def split_data(puzzles: list, train: float, dev: float, test: float):
    """Create train/dev/test splits."""
    pass
```

### run_solver.py

```python
# Usage: python scripts/run_solver.py --solver hypothesis --model gpt-4-turbo --puzzles data/splits/dev.jsonl

def main(args):
    puzzles = load_puzzles(args.puzzles)
    oracle = Oracle(args.oracle_model)
    solver = create_solver(args.solver, args.model)
    
    for puzzle in puzzles:
        solution = solver.solve(puzzle['scenario'], oracle)
        save_conversation({
            'puzzle_id': puzzle['id'],
            'model': args.model,
            'solver': args.solver,
            'conversation': solver.conversation,
            'final_solution': solution,
            'num_questions': len(solver.conversation) // 2
        }, args.output_dir)
```

### evaluate.py

```python
# Usage: python scripts/evaluate.py --conversations results/hypothesis_gpt4/ --output results/metrics.json

def main(args):
    conversations = load_conversations(args.conversations)
    puzzles = load_puzzles(args.puzzles)
    evaluator = Evaluator(args.judge_model)
    
    results = []
    for conv in conversations:
        puzzle = get_puzzle(puzzles, conv['puzzle_id'])
        score, reasoning = evaluator.judge_solution(
            puzzle['truth'], 
            conv['final_solution']
        )
        results.append({
            'puzzle_id': conv['puzzle_id'],
            'score': score,
            'num_questions': conv['num_questions'],
            'judge_reasoning': reasoning
        })
    
    metrics = evaluator.compute_metrics(results)
    save_metrics(metrics, args.output)
```

## Configuration

**config.yaml:**
```yaml
models:
  oracle: "gpt-4-turbo"
  judge: "gpt-4-turbo"
  solvers:
    - "gpt-4-turbo"
    - "claude-3-5-sonnet-20241022"
    - "llama-3-70b"

solvers:
  - name: "naive"
  - name: "hypothesis"
    params:
      num_hypotheses: 5
  - name: "infogain"
    params:
      num_candidates: 5

experiment:
  max_questions: 50
  temperature: 0.7
  puzzles_per_run: 50
```

## Dependencies (requirements.txt)

```
openai>=1.0.0
anthropic>=0.18.0
python-dotenv>=1.0.0
pyyaml>=6.0
scikit-learn>=1.3.0  # for deduplication
sentence-transformers>=2.2.0  # for semantic similarity
numpy>=1.24.0
tqdm>=4.65.0
```

## Execution Flow

### Phase 1: Data Preparation
```bash
python scripts/prepare_data.py \
  --input data/raw/puzzles.jsonl \
  --output data/filtered/
```

### Phase 2: Run Experiments
```bash
# Run single experiment
python scripts/run_solver.py \
  --solver hypothesis \
  --model gpt-4-turbo \
  --puzzles data/splits/dev.jsonl \
  --output results/hypothesis_gpt4/

# Run batch experiments (shell script or loop)
for solver in naive hypothesis infogain; do
  for model in gpt-4-turbo claude-3-5-sonnet; do
    python scripts/run_solver.py \
      --solver $solver --model $model \
      --puzzles data/splits/test.jsonl \
      --output results/${solver}_${model}/
  done
done
```

### Phase 3: Evaluation
```bash
python scripts/evaluate.py \
  --conversations results/hypothesis_gpt4/ \
  --puzzles data/splits/test.jsonl \
  --output results/hypothesis_gpt4_metrics.json
```

## Design Principles

1. **Flat Structure**: Minimal nesting, easy navigation
2. **Single Responsibility**: Each file has one clear purpose
3. **Stateless Where Possible**: Oracle is stateless, solvers hold minimal state
4. **Plain Text Configs**: YAML for configs, no complex frameworks
5. **JSONL for Data**: Line-delimited JSON, easy to stream and debug
6. **No Databases**: File-based storage, simple and portable
7. **Explicit Over Implicit**: No magic, clear function signatures
8. **Reproducible**: Seed setting, logging, save all intermediate outputs

## Testing Strategy

Create `tests/` directory with:
- `test_oracle.py`: Test oracle responses on known puzzles
- `test_solvers.py`: Test each solver on 2-3 example puzzles
- `test_evaluator.py`: Test metric calculations with mock data

Use pytest, keep tests simple and fast.

## Logging

```python
# In each script
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
```

Log:
- Each question asked and answer received
- Hypothesis updates (for hypothesis solver)
- API errors and retries
- Timing information

## Error Handling

- Retry API calls up to 3 times with exponential backoff
- Log all errors, don't crash on single puzzle failure
- Save partial results regularly (every 10 puzzles)
- Graceful degradation: if solver fails, save what you have

## Next Steps for Implementation

1. Start with `models.py` and `utils.py` (foundation)
2. Implement `oracle.py` (test manually with 5 examples)
3. Implement `NaiveSolver` only (test end-to-end)
4. Build `evaluator.py` (test on mock data)
5. Create `run_solver.py` and test full pipeline
6. Add `HypothesisSolver` and `InfoGainSolver`
7. Build `prepare_data.py` last (once pipeline works)

This design prioritizes getting a working end-to-end system quickly, then iterating.