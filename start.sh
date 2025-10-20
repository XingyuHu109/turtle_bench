#!/usr/bin/env bash
set -euo pipefail

# TurtleSoup quickstart runner with env setup
# Supports DeepSeek and LM Studio local endpoint.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

PROVIDER="lmstudio"   # default to local LM Studio for testing
SOLVER="hypothesis"
SPLIT="dev"            # dev|test|train
MAX_Q=30

# 0) Source environment files early so keys are available to checks below
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi
if [[ -f "$ROOT_DIR/.sample.env" ]]; then
  # Load sample env as a fallback; real .env overrides it
  set -a; source "$ROOT_DIR/.sample.env"; set +a
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider)
      PROVIDER="$2"; shift 2;;
    --solver)
      SOLVER="$2"; shift 2;;
    --split)
      SPLIT="$2"; shift 2;;
    --max_questions)
      MAX_Q="$2"; shift 2;;
    --help|-h)
      cat << USAGE
Usage: ./start.sh [--provider lmstudio|deepseek] [--solver naive|hypothesis|infogain] [--split dev|test|train] [--max_questions N]

Behavior:
  - Creates a Python venv in .venv and installs requirements.
  - Prepares sample data (CSV -> JSONL + splits) if needed.
  - Configures LLM endpoint:
      lmstudio: http://127.0.0.1:1234 (OpenAI-compatible, no key required)
      deepseek: https://api.deepseek.com (requires DEEPSEEK_API_KEY)
  - Runs the solver for the selected split and evaluates results.

Examples:
  ./start.sh --provider lmstudio --solver hypothesis --split dev
  ./start.sh --provider deepseek --solver infogain --split test
USAGE
      exit 0;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

echo "[TurtleSoup] Using provider=$PROVIDER, solver=$SOLVER, split=$SPLIT, max_questions=$MAX_Q"

# 1) Python venv + deps
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[TurtleSoup] Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q -r "$ROOT_DIR/turtlesoup/requirements.txt"

# 2) Data prep (CSV -> JSONL + splits)
if [[ ! -f "$ROOT_DIR/turtlesoup/data/splits/dev.jsonl" ]]; then
  echo "[TurtleSoup] Preparing data from sample_data.csv"
  python "$ROOT_DIR/turtlesoup/scripts/prepare_data.py" \
    --input "$ROOT_DIR/sample_data.csv" \
    --output "$ROOT_DIR/turtlesoup/data/raw" \
    --splits "$ROOT_DIR/turtlesoup/data/splits" \
    --train 0.8 --dev 0.1 --test 0.1
fi

# 3) Endpoint/env setup
case "$PROVIDER" in
  lmstudio)
    # Remember LM Studio endpoint for future use
    if [[ -f "$ROOT_DIR/turtlesoup/configs/lmstudio.env" ]]; then
      set -a; source "$ROOT_DIR/turtlesoup/configs/lmstudio.env"; set +a
    fi
    export LLM_API_BASE=${LLM_API_BASE:-"http://127.0.0.1:1234"}
    export LLM_API_KEY=${LLM_API_KEY:-""}  # Usually not required by LM Studio
    export LM_STUDIO_MODEL=${LM_STUDIO_MODEL:-"google/gemma-3-4b"}
    export MODEL_NAME="$LM_STUDIO_MODEL"
    echo "[TurtleSoup] LM Studio endpoint: $LLM_API_BASE (model=$MODEL_NAME)"
    ;;
  deepseek)
    export LLM_API_BASE=${LLM_API_BASE:-"https://api.deepseek.com"}
    # Requires DEEPSEEK_API_KEY in env or .env
    export LLM_API_KEY=${LLM_API_KEY:-${DEEPSEEK_API_KEY:-""}}
    if [[ -z "$LLM_API_KEY" ]]; then
      echo "[TurtleSoup] ERROR: DEEPSEEK_API_KEY not set in environment or .env" >&2
      exit 1
    fi
    export MODEL_NAME=${MODEL_NAME:-"deepseek-chat"}
    echo "[TurtleSoup] DeepSeek endpoint: $LLM_API_BASE (model=$MODEL_NAME)"
    ;;
  *)
    echo "[TurtleSoup] Unknown provider: $PROVIDER" >&2; exit 1;;
esac

SPLIT_PATH="$ROOT_DIR/turtlesoup/data/splits/${SPLIT}.jsonl"
if [[ ! -f "$SPLIT_PATH" ]]; then
  echo "[TurtleSoup] ERROR: split file not found: $SPLIT_PATH" >&2
  exit 1
fi

OUT_DIR="$ROOT_DIR/results/${SOLVER}_${PROVIDER}"
mkdir -p "$OUT_DIR"

echo "[TurtleSoup] Running solver=$SOLVER on $SPLIT_PATH -> $OUT_DIR"
python "$ROOT_DIR/turtlesoup/scripts/run_solver.py" \
  --solver "$SOLVER" \
  --model "$MODEL_NAME" \
  --oracle_model "$MODEL_NAME" \
  --puzzles "$SPLIT_PATH" \
  --output "$OUT_DIR" \
  --max_questions "$MAX_Q"

echo "[TurtleSoup] Evaluating results in $OUT_DIR"
python "$ROOT_DIR/turtlesoup/scripts/evaluate.py" \
  --conversations "$OUT_DIR" \
  --puzzles "$SPLIT_PATH" \
  --output "$OUT_DIR/metrics.json" \
  --judge_model "$MODEL_NAME"

echo "[TurtleSoup] Done. Metrics: $OUT_DIR/metrics.json"
