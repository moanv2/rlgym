# Common dev tasks. Run `make help` to see all.
PY      ?= python
EXP     ?= exp_001_baseline
CONFIG  ?= configs/experiments/$(EXP).yaml

.PHONY: help install test test-fast lint format type clean train eval visualize tb wandb-login

help:
	@echo "Targets:"
	@echo "  install        Install package + dev deps (editable)"
	@echo "  test           Run pytest (full suite)"
	@echo "  test-fast      Run pytest excluding slow/rocketsim/gpu tests"
	@echo "  lint           Run ruff lint + mypy"
	@echo "  format         Run ruff format"
	@echo "  type           Run mypy"
	@echo "  train EXP=...  Train using configs/experiments/<EXP>.yaml"
	@echo "  eval BLUE=... ORANGE=...  Bot-vs-bot evaluation"
	@echo "  visualize CKPT=...        Watch a checkpoint play"
	@echo "  clean          Remove caches and build artifacts"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow and not rocketsim and not gpu"

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m mypy

format:
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

type:
	$(PY) -m mypy

train:
	$(PY) -m rlbot.training.train --config $(CONFIG)

eval:
	$(PY) -m rlbot.evaluation.evaluate --blue $(BLUE) --orange $(ORANGE) --episodes $(or $(EPISODES),100)

visualize:
	$(PY) scripts/visualize.py --checkpoint $(CKPT)

wandb-login:
	$(PY) -m wandb login

clean:
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .ruff_cache -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .mypy_cache -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -prune -exec rm -rf {} + 2>/dev/null || true
