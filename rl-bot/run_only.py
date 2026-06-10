"""Start ONLY the match (RLBotServer must already be running).

Usage (from the rl-bot/ project root, venv active):
    python run_only.py

Use this during development together with dev.toml's auto_start_agents = false:
start the server once, then iterate on the bot process without restarting the
whole match.
"""

from pathlib import Path

from rlbot.managers import MatchManager

MATCH_CONFIG_PATH = "rlbot.toml"

if __name__ == "__main__":
    root_dir = Path(__file__).parent

    # RLBotServer MUST BE STARTED MANUALLY first.
    match_manager = MatchManager()
    match_manager.start_match(root_dir / MATCH_CONFIG_PATH, False)

    _ = input("\nPress enter to end the match: ")

    # End the match and disconnect, but leave RLBotServer running.
    match_manager.stop_match()
    match_manager.disconnect()
