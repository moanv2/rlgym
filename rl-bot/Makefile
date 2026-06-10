# Convenience targets for the RLBot v5 bot.
# On Windows, `make` works from Git Bash / WSL, or just call the scripts directly
# (e.g. `.\scripts\setup.ps1`). PY points at the project venv interpreter.

ifeq ($(OS),Windows_NT)
    PY := venv/Scripts/python.exe
else
    PY := venv/bin/python
endif

.PHONY: help setup validate run run-only freeze docker-build docker-validate clean

help:
	@echo "Targets:"
	@echo "  setup           Create venv + install deps + validate (Windows: scripts/setup.ps1)"
	@echo "  validate        Run offline checks (no v4 symbols, valid TOML, agent_id, imports)"
	@echo "  run             Start RLBotServer + a match (Windows + Rocket League required)"
	@echo "  run-only        Start only the match (server already running)"
	@echo "  freeze          Pin exact installed versions to requirements.lock"
	@echo "  docker-build    Build the validation image"
	@echo "  docker-validate Run validate.py inside Docker"
	@echo "  clean           Remove venv and caches"

setup:
	bash scripts/setup.sh

validate:
	$(PY) validate.py

run:
	$(PY) run.py

run-only:
	$(PY) run_only.py

freeze:
	$(PY) -m pip freeze > requirements.lock
	@echo "Wrote requirements.lock — commit it for byte-for-byte reproducibility."

docker-build:
	docker build -t rlbotv5-validate .

docker-validate:
	docker run --rm rlbotv5-validate

clean:
	rm -rf venv __pycache__ src/__pycache__ src/util/__pycache__ src/policy/__pycache__ .mypy_cache .ruff_cache
