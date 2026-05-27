"""
Watch a trained simple_bot checkpoint play in the rlviser visualizer.

Prerequisites in order:

1. simple_bot.py has already finished a training run and produced checkpoints
   under diego-bots/checkpoints/simple_bot-<timestamp>/

2. rlviser-py is installed in the rlbot310 env. It is. Verify with:
       python -c "import rlviser_py; print(rlviser_py.__version__)"

3. The rlviser visualizer binary is running. Download the latest release from
       https://github.com/VirxEC/rlviser/releases
   Extract somewhere, then double click rlviser.exe (or run it from a terminal).
   A window will open showing an empty Rocket League arena. Leave it open.

4. Activate the rlbot310 env, then run this script:
       conda activate rlbot310
       python diego-bots/simple_bot_play.py

The rlviser window will start showing two cars (blue + orange) playing a 1v1
using the latest saved policy. Press Ctrl+C in this terminal to stop.

Important caveat for this 500k step bot:
    Entropy was still ~4.48 at the end (near maximum for 90 actions ~= 4.50),
    so the policy is mostly random. Expect twitchy chaotic behavior with
    occasional ball touches. That is the baseline you compare future, longer
    trained bots against.
"""

# --------------------------------------------------------------------------
# Imports (same as simple_bot.py; duplicated because diego-bots is not a
# Python module due to the hyphen in its name, and we want the script to
# remain self contained)
# --------------------------------------------------------------------------
from pathlib import Path

import rlgym_sim
from rlgym_ppo import Learner
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    EventReward,
    VelocityBallToGoalReward,
    VelocityPlayerToBallReward,
)
from rlgym_sim.utils.state_setters import RandomState
from rlgym_sim.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition,
    NoTouchTimeoutCondition,
)

from rlbot.actions.lookup_action import LookupAction


# --------------------------------------------------------------------------
# Same env factory as simple_bot.py. The Learner needs an env_builder so it
# can spin up workers; with n_proc=1 and render=True we only get one worker
# and it streams state to rlviser at each step.
# --------------------------------------------------------------------------
def build_env():
    tick_skip = 8
    no_touch_ticks = int(10 * 120 / tick_skip)

    reward_fn = CombinedReward(
        reward_functions=(
            VelocityPlayerToBallReward(),
            VelocityBallToGoalReward(),
            EventReward(goal=1.0, concede=-1.0, shot=0.1, demo=0.1),
        ),
        reward_weights=(0.05, 0.5, 10.0),
    )

    return rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=1,
        spawn_opponents=True,
        reward_fn=reward_fn,
        obs_builder=DefaultObs(),
        state_setter=RandomState(
            ball_rand_speed=True,
            cars_rand_speed=True,
            cars_on_ground=False,
        ),
        terminal_conditions=[
            GoalScoredCondition(),
            NoTouchTimeoutCondition(no_touch_ticks),
        ],
        action_parser=LookupAction(),
    )


# --------------------------------------------------------------------------
# Find the latest checkpoint timestep folder.
#
# Folder structure produced by rlgym_ppo:
#   diego-bots/checkpoints/
#     simple_bot-<timestamp>/        <- one folder per training run
#       100000/                       <- one folder per save_every_ts hit
#         PPO_POLICY.pt + others
#       200008/
#       ...
#       500028/                       <- latest, what we want
#
# rlgym_ppo's Learner.load_from() expects the path to point at a specific
# timestep folder (it reads PPO_POLICY.pt directly), not the parent run
# folder. So we walk two levels: pick the most recent run, then the
# largest numeric timestep folder inside it.
# --------------------------------------------------------------------------
def find_latest_checkpoint() -> str:
    ckpt_root = Path("diego-bots/checkpoints")
    run_folders = sorted(
        [p for p in ckpt_root.glob("simple_bot*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_folders:
        raise SystemExit(
            "No checkpoint run folder found under diego-bots/checkpoints/. "
            "Train first with: python diego-bots/simple_bot.py"
        )

    latest_run = run_folders[0]

    # Pick the timestep subfolder with the highest numeric name
    timestep_folders = [p for p in latest_run.iterdir() if p.is_dir() and p.name.isdigit()]
    if not timestep_folders:
        raise SystemExit(
            f"Run folder {latest_run} has no timestep subfolders yet. "
            "Did training save at least one checkpoint?"
        )
    latest_ckpt = max(timestep_folders, key=lambda p: int(p.name))

    print(f"Loading checkpoint from: {latest_ckpt}")
    return str(latest_ckpt)


# --------------------------------------------------------------------------
# Main: build a Learner pointed at the saved checkpoint with render=True.
# The Learner auto resumes from the latest checkpoint inside the load folder
# and starts streaming gameplay state to rlviser.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    load_folder = find_latest_checkpoint()

    learner = Learner(
        build_env,

        # One worker is enough for visualization. Rendering benefits nothing
        # from parallelism; you only watch one car anyway.
        n_proc=1,
        min_inference_size=1,

        metrics_logger=None,

        # PPO knobs must match what the saved policy was trained with so
        # the loaded weights fit the network shape. Keep these identical
        # to simple_bot.py.
        ppo_batch_size=50_000,
        ts_per_iteration=50_000,
        exp_buffer_size=150_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=0.01,
        ppo_epochs=2,
        standardize_returns=True,
        standardize_obs=False,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),

        # The Learner exits when cumulative timesteps hit this number.
        # Our saved bot already has 500k cumulative steps, so we bump this
        # high to keep the visualization running until you hit Ctrl+C.
        timestep_limit=1_000_000_000,

        # Don't write new checkpoints while watching.
        save_every_ts=1_000_000_000,

        # Where to find the trained weights. checkpoint_load_folder points
        # at the timestamped run folder; the Learner picks the highest
        # numbered subfolder (latest cumulative timestep) automatically.
        checkpoint_load_folder=load_folder,

        # Separate save folder so this play session never overwrites the
        # training run's checkpoints. We will not actually save anything
        # because save_every_ts above is set huge, but this is a safety net.
        checkpoints_save_folder="diego-bots/checkpoints/_play_session_unused",

        log_to_wandb=False,
        load_wandb=False,

        # The big switch. With render=True, worker 0 calls into rlviser_py
        # at every step to push the current game state to the visualizer.
        render=True,

        # render_delay throttles the visualization to roughly real time.
        # Without this, the sim runs at hundreds of FPS and you would see
        # a blur. 6 ms per step ~= simulator real time at tick_skip=8.
        render_delay=0.006,
    )

    print("Press Ctrl+C in this terminal to stop the visualization.")
    learner.learn()
