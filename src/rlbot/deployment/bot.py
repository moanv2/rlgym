"""Stub: wire the trained policy into the RLBot framework.

Final-week task. The general shape:
    1. Load policy weights from checkpoints/<exp>/latest.
    2. In the RLBot `get_output()` callback, build the same obs as training, run the
       policy forward pass, decode the action, return SimpleControllerState.
    3. Package as a `.bot.toml`-described bot for the local RLBot match runner.

References:
    - https://github.com/RLBot/RLBot
    - https://github.com/RLGym/rlgym-tools (PPO -> RLBot adapters exist)
"""
