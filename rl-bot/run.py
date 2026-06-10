"""Start RLBotServer AND a match defined by a match-config TOML.

Usage (from the rl-bot/ project root, venv active):
    python run.py              # default: rlbot.toml (my bot vs a Psyonix AllStar)
    python run.py human.toml   # my bot (Blue) vs YOU, a human (Orange)

Requires the RLBot v5 launcher/GUI installed on Windows and Rocket League owned
on Steam/Epic. This does NOT bundle the server.
"""

import sys
from pathlib import Path
from time import sleep

from rlbot import flat
from rlbot.managers import MatchManager

DEFAULT_MATCH_CONFIG_FILE = "rlbot.toml"

if __name__ == "__main__":
    root_dir = Path(__file__).parent
    match_config_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MATCH_CONFIG_FILE

    # Start RLBotServer and the match.
    match_manager = MatchManager()
    match_manager.start_match(root_dir / match_config_file)

    sleep(5)

    # Wait for the match to end (or press Ctrl+C to kill it).
    while (
        match_manager.packet is None
        or match_manager.packet.match_info.match_phase != flat.MatchPhase.Ended
    ):
        sleep(0.1)

    # Ensure RLBotServer shuts down.
    match_manager.shut_down()
