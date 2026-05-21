"""
Simple Rocket League 1v1 PPO bot. Educational scaffold.

This script bypasses the YAML config system on purpose. Every knob you can turn
when training a bot lives directly in this file with a comment explaining what
it does, why it matters, and what happens if you change it.

Run from the project root with the rlbot310 conda env activated:

    conda activate rlbot310
    python diego-bots/simple_bot.py

Expected runtime on the RTX 4070 Laptop: about 10 to 20 minutes for the
500k timestep limit set below. The bot will not be good at the end of this,
but you will SEE policy reward trending upward in the console reports, which
means the PPO update loop is working and the bot is learning to move toward
the ball.

Output:
    diego-bots/checkpoints/simple_bot/<timestep>/   trained policy snapshots
"""

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------

from pathlib import Path

# rlgym_sim is the Python wrapper around the C++ RocketSim physics engine.
# It gives us the make() factory that builds a fresh environment, which is
# basically a black box that simulates one match of Rocket League and lets
# our agent send controller inputs to its car.
import rlgym_sim

# The Learner from rlgym_ppo is the brain of the operation. It owns the
# policy network (the actor), the value network (the critic), the rollout
# workers that play games in parallel, and the PPO update loop that improves
# the network from collected gameplay.
from rlgym_ppo import Learner

# Reward functions are the signals we hand the agent to nudge it toward
# useful behavior. Each is a class that, on every timestep, returns a scalar
# number for each player. CombinedReward stacks several rewards into one,
# multiplying each by a weight so we can balance their influence.
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    VelocityPlayerToBallReward,   # positive while the car is moving toward the ball
    VelocityBallToGoalReward,     # positive while the ball is moving toward the opponent goal
    EventReward,                  # one off bonus on specific events (goal, save, demo, etc.)
)

# DefaultObs converts the raw game state into the fixed length numeric
# vector that we feed into the policy network. Positions, velocities, ball
# state, boost amount, and so on. The shape of this vector is fixed at
# training time; changing the obs builder means starting a fresh bot.
from rlgym_sim.utils.obs_builders import DefaultObs

# RandomState decides where everything spawns at the start of each episode.
# Random positions plus random velocities expose the bot to far more
# situations than a fixed kickoff would, which speeds up early learning.
from rlgym_sim.utils.state_setters import RandomState

# Terminal conditions decide when an episode is over. We end on a goal
# (otherwise the ball would sit in the net forever) and on a no touch
# timeout (otherwise we collect minutes of useless data on two cars that
# are stuck upside down doing nothing).
from rlgym_sim.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition,
    NoTouchTimeoutCondition,
)

# LookupAction is our vendored fully discrete action parser. The policy
# network outputs a single integer in 0..89 and LookupAction translates it
# into the 8 dimensional controller vector that rlgym_sim expects:
# [throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
# Fully discrete heads are much easier to train than continuous ones for
# a first bot, which is why we are using it.
from rlbot.actions.lookup_action import LookupAction


# --------------------------------------------------------------------------
# Environment factory
# --------------------------------------------------------------------------

# This function must return a brand new environment every time it is called.
# rlgym_ppo's Learner calls it once inside each rollout worker process, so
# every parallel worker has its own private copy of the simulator. Keep it
# self contained: no global state, no closures over big objects, so it can
# be pickled across process boundaries.
def build_env():
    # Rocket League physics runs at 120 ticks per second. tick_skip controls
    # how many physics ticks happen between two consecutive agent decisions.
    # tick_skip = 8 means the policy acts at 120 / 8 = 15 Hz, which is the
    # community default. Lower tick_skip means more agent decisions per
    # second (finer control but more compute). Higher means the opposite.
    tick_skip = 8

    # End the episode if no player touches the ball for this many seconds.
    # 10 seconds is enough time to chase the ball from a random spawn, and
    # short enough to discard pure idle states quickly.
    no_touch_seconds = 10
    no_touch_ticks = int(no_touch_seconds * 120 / tick_skip)

    # Build the reward function as a weighted sum of three components.
    # The art of training RL bots is mostly in choosing rewards and weights.
    #
    # Some rules of thumb:
    #   Continuous rewards fire on every step, so they accumulate fast and
    #     should have small weights to avoid dominating the signal.
    #   Event rewards fire rarely (a goal might happen every few minutes),
    #     so they need large weights to actually register during learning.
    #   Negative weights punish behavior. EventReward(concede=-1.0) here
    #     teaches the bot that getting scored on is bad.
    reward_fn = CombinedReward(
        reward_functions=(
            VelocityPlayerToBallReward(),
            VelocityBallToGoalReward(),
            EventReward(
                goal=1.0,       # scoring is the only thing that ultimately matters
                concede=-1.0,   # getting scored on is the opposite of scoring
                shot=0.1,       # small bonus for shooting on net
                demo=0.1,       # small bonus for demolishing the opponent
            ),
        ),
        reward_weights=(
            0.05,   # weak: this reward is easy to game and fires every step
            0.5,    # medium: this is the offense oriented continuous signal
            10.0,   # strong: events are rare, so multiply them up
        ),
    )

    # Build and return the environment. Each named argument plugs in one
    # of the modular components we just set up.
    return rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=1,                 # 1v1 final, so one car per side
        spawn_opponents=True,        # spawn a second car for self play
        reward_fn=reward_fn,
        obs_builder=DefaultObs(),
        state_setter=RandomState(
            ball_rand_speed=True,    # ball starts with a random velocity
            cars_rand_speed=True,    # cars start with random velocities
            cars_on_ground=False,    # half the time, cars spawn airborne
        ),
        terminal_conditions=[
            GoalScoredCondition(),
            NoTouchTimeoutCondition(no_touch_ticks),
        ],
        action_parser=LookupAction(),
    )


# --------------------------------------------------------------------------
# Training entrypoint
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Auto resume from the latest saved checkpoint.
#
# PPO does not store gameplay episodes anywhere. What it stores is the policy
# weights, the critic weights, the Adam optimizer state, and the cumulative
# timestep counter. To "keep getting better" we point the Learner at the
# latest checkpoint folder; it reloads all of that state and continues from
# the same cumulative timestep, generating fresh on-policy data with the
# already-trained network.
#
# Folder layout produced by rlgym_ppo:
#   diego-bots/checkpoints/
#     simple_bot-<unix_ts>/        one folder per training session
#       100000/                     one folder per save_every_ts hit
#         PPO_POLICY.pt + others
#       200008/
#       ...
#
# We pick the most recent session, then the highest numbered timestep inside.
# Returns None if no checkpoint exists yet, in which case training starts fresh.
# --------------------------------------------------------------------------
def find_latest_checkpoint(base: str = "diego-bots/checkpoints") -> str | None:
    base_path = Path(base)
    if not base_path.exists():
        return None
    runs = sorted(
        [p for p in base_path.glob("simple_bot*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        timesteps = [p for p in run.iterdir() if p.is_dir() and p.name.isdigit()]
        if timesteps:
            latest = max(timesteps, key=lambda p: int(p.name))
            return str(latest)
    return None


if __name__ == "__main__":
    # The Learner runs Proximal Policy Optimization (PPO), the algorithm
    # used by basically every modern RL bot in this domain. The general
    # loop is:
    #   1. Run the policy in many parallel envs to collect a batch of
    #      gameplay data (this is the rollout phase).
    #   2. For each step, compute the advantage (how much better or worse
    #      the action turned out to be than the value function predicted).
    #   3. Update the policy network to make good actions more likely and
    #      bad ones less likely, while clipping the update size so the
    #      policy never moves too far in one step (this is the PPO trick).
    #   4. Update the value network to better predict future rewards.
    #   5. Repeat.
    #
    # PPO is on-policy: there is no replay buffer. Each batch of gameplay
    # is collected by the current policy, used for one round of updates,
    # then discarded. "Improvement" comes from collecting fresh data with
    # ever smarter policies, not from re-watching old episodes.
    #
    # Knobs explained:

    resume_from = find_latest_checkpoint()
    if resume_from:
        print(f"[resume] loading checkpoint: {resume_from}")
    else:
        print("[fresh] no checkpoint found, training from scratch")

    learner = Learner(
        build_env,

        # How many copies of the env to run in parallel. Each one lives in
        # its own process. More workers = more samples per second, but
        # more RAM and CPU contention. Rule of thumb: (CPU cores - 1).
        n_proc=8,

        # The Learner batches inference requests across workers for GPU
        # efficiency. min_inference_size says: do not run the policy
        # forward pass until at least this many env steps are queued up.
        # 80 is a reasonable middle ground for 8 workers.
        min_inference_size=80,

        # Optional callback for per iteration metrics. None means default
        # console reporting only.
        metrics_logger=None,

        # PPO algorithm hyperparameters. These are the actual learning levers.
        #
        # ppo_batch_size: total number of timesteps the gradient sees per
        # PPO update. Bigger batches give more stable gradients but slower
        # iteration cadence.
        ppo_batch_size=50_000,

        # ts_per_iteration: how many timesteps to collect before each PPO
        # update. We keep this equal to the batch size for simplicity.
        ts_per_iteration=50_000,

        # exp_buffer_size: rolling buffer of recent experience that the
        # Learner may sample from. Slightly larger than batch size gives
        # the algorithm a small amount of off policy slack.
        exp_buffer_size=150_000,

        # ppo_minibatch_size: gradient descent batch size inside the PPO
        # update. Keep equal to ppo_batch_size for the simplest behavior
        # (full batch updates). Smaller values mean SGD with more steps.
        ppo_minibatch_size=50_000,

        # ppo_ent_coef: entropy bonus, encourages exploration. Higher
        # values keep the policy stochastic for longer; lower values let
        # it converge faster but risk premature exploitation of a bad
        # local optimum. 0.01 is the conventional starting point.
        ppo_ent_coef=0.01,

        # ppo_epochs: how many passes over each collected batch the PPO
        # update makes. More passes = faster learning per timestep, but
        # the policy can drift too far from the data and the KL divergence
        # blows up. 2 to 4 is the sweet spot in most rocket league setups.
        ppo_epochs=2,

        # standardize_returns: normalize advantage estimates by their
        # running standard deviation. Almost always helps stability. Turn
        # off only if you have a very specific reason to.
        standardize_returns=True,

        # standardize_obs: same idea but for the input observations.
        # DefaultObs is already roughly in a reasonable scale, so we leave
        # this off. Custom observation builders may want it on.
        standardize_obs=False,

        # Neural network architecture. policy_layer_sizes is the actor
        # (the network that picks actions), critic_layer_sizes is the
        # critic (the network that estimates value). Both are vanilla MLPs.
        # (256, 256, 256) is a small but solid first try. Doubling these
        # would make the bot stronger eventually but multiply training time.
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),

        # Total CUMULATIVE environment steps allowed before the Learner exits.
        # This is the sum across all training sessions, not just this one.
        # Set very high so resumed runs keep going until you Ctrl+C. The
        # Learner saves the running counter in BOOK_KEEPING_VARS.json so
        # resuming preserves it (your already trained 500k is still counted).
        # For a real strong bot you want tens of millions; 1 billion is just
        # "effectively unlimited until I stop it manually."
        timestep_limit=1_000_000_000,

        # Save a checkpoint every this many timesteps. Frequent saves mean
        # you can Ctrl+C any time without losing more than a few minutes of
        # progress. Saved checkpoints accumulate; older ones beyond
        # n_checkpoints_to_keep (default 5) are auto-deleted by the Learner.
        save_every_ts=100_000,

        # Where to write checkpoints. The Learner appends a unix timestamp
        # to make each session's folder unique, so resuming creates a fresh
        # simple_bot-<new_timestamp>/ folder containing only the new saves.
        # The cumulative timestep counter in BOOK_KEEPING_VARS.json continues
        # from where the loaded checkpoint left off.
        checkpoints_save_folder="diego-bots/checkpoints/simple_bot",

        # Where to LOAD the starting weights from. None = train from scratch.
        # find_latest_checkpoint() returns the most recent saved checkpoint
        # path or None if there is no prior run.
        checkpoint_load_folder=resume_from,

        # Enable wandb logging. Requires `wandb login` to have been run once
        # so the API key is cached in your ~/.netrc. All runs land in the
        # 'rlgym-finalproject' project under whichever wandb entity your
        # account defaults to (for Diego that is 'diego08-ie-university').
        # wandb_group_name lets you cluster runs of the same experiment in
        # the dashboard, useful when running multiple seeds.
        log_to_wandb=True,
        wandb_project_name="rlgym-finalproject",
        wandb_group_name="simple_bot",
        wandb_run_name=None,    # auto-generated run name; set to a string to force one

        # Real time rendering is off; we have no visualizer hooked up here.
        # Use scripts/visualize.py once you have a trained checkpoint.
        render=False,
    )

    # learn() is blocking. It runs until timestep_limit is reached or you
    # press Ctrl+C. Each iteration prints a report to the console:
    #
    #   Policy Reward: average reward per episode this iteration.
    #     Trending up over time means the bot is improving.
    #   Policy Entropy: how random the policy still is. Drops over time
    #     as the bot commits to specific behavior. A sudden crash to zero
    #     is bad and usually means you need more entropy bonus.
    #   Mean KL Divergence: how much the policy moved this update.
    #     Should be small and stable. Spikes mean lower ppo_epochs or
    #     learning rate.
    #   SB3 Clip Fraction: fraction of updates clipped by PPO. 0.05 to
    #     0.20 is healthy. Near zero means nothing is updating; near 0.5
    #     means updates are too aggressive.
    #   Collected Steps per Second: raw rollout throughput. The bigger
    #     the better and is the main thing your hardware controls.
    #
    # Watch Policy Reward most of all. That is the bot getting smarter.
    learner.learn()
