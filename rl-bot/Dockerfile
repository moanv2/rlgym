# ============================================================================
# Reproducible VALIDATION / CI image for the RLBot v5 bot.
#
#   IMPORTANT: this container does NOT play Rocket League. RLBot v5 needs the
#   actual game + native RLBotServer on Windows, which cannot run in a Linux
#   container. This image exists to give EVERY machine an identical environment
#   for: installing deps, importing the bot, parsing configs, and running
#   validate.py (no v4 symbols, valid TOML, consistent agent_id).
#
#   Real matches: use scripts/setup.ps1 + `python run.py` on Windows.
#
# Build + run:
#     docker build -t rlbotv5-validate .
#     docker run --rm rlbotv5-validate
#   (or: docker compose run --rm validate)
# ============================================================================
FROM python:3.12-slim

WORKDIR /app

# git is needed to pip-install rlgym-compat from its GitHub URL (the RLBot v5
# compatible 2.x line is not published to PyPI).
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install deps first for layer caching. Prefer the lockfile when present.
COPY requirements.txt ./
COPY requirements.lock* ./
RUN python -m pip install --upgrade pip && \
    if [ -f requirements.lock ]; then \
        pip install --no-cache-dir -r requirements.lock; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy the project.
COPY . .

ENV PYTHONPATH=/app/src

# Default action: run the offline validation suite. Exits non-zero on failure,
# so this doubles as a CI gate.
CMD ["python", "validate.py"]
