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
# Experiment identity — change THIS string when you change rewards, obs,
# architecture, or anything else that fundamentally changes the bot. Each
# experiment has its own checkpoint folder so different experiments do not
# pollute each other's checkpoints. Each wandb run name is also prefixed
# with this, so you can overlay experiments cleanly in the dashboard.
#
# Examples to copy:
#     "baseline"           - the original 3 reward components, plateaued at 11M
#     "richer_rewards"     - 5 reward components (current)
#     "nexto_rewards"      - ported Nexto reward function
#     "advanced_obs"       - richer observation builder
#
# When you change this string and rerun, the bot starts training from
# scratch in a fresh folder. The previous experiment's checkpoints stay
# untouched as comparison baselines.
# --------------------------------------------------------------------------
# Fresh experiment: Nexto-style 10-component reward + 4 custom hardcoded
# rewards (Supersonic, AerialBall, BigBoostProximity, BackboardDefense)
# + 30/70 kickoff-vs-random state mix + medium 512x3 architecture.
# Previous experiments are preserved in their own folders for comparison:
#   - nexto_rewards/        : 5→10 component reward, 256x3, reached 130M
#   - nexto_plus_kickoff/    : 14-component reward, 256x3, reached 17.6M
EXPERIMENT_NAME = "nexto_plus_kickoff_512"

# --------------------------------------------------------------------------
# wandb display-name convention.
#
# Every run shows up in wandb as f"{WANDB_NAME_PREFIX}_{end_timesteps}_r{N}",
# e.g. "baseline_70M_r3" — the prefix is fixed (Diego likes "baseline"), the
# middle is the cumulative timesteps at the end of THIS session, and N is the
# sequential session number.
#
# STARTING_RUN_NUMBER seeds the sequence so it lines up with what's already
# in your wandb dashboard. Diego already has r1 (baseline_11M) and r2
# (baseline_26M_r2) from before this auto-saving system existed, so the next
# auto-saved session should be r3. Bump this once if you ever need to.
# --------------------------------------------------------------------------
WANDB_NAME_PREFIX = "baseline"
STARTING_RUN_NUMBER = 3

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------

import atexit
import json
from datetime import datetime, timezone
from pathlib import Path

# Where per-session summary JSONs are saved on Ctrl+C or normal exit. One
# file per run, globally numbered (run_001_*.json, run_002_*.json, ...) so
# future analysis scripts can ingest the whole history at once.
HISTORY_DIR = Path("history_and_summary")

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

# Reward function is now constructed via build_nexto_style_reward() inside
# build_env(). The previously-inlined CombinedReward + individual reward
# imports moved into src/rlbot/rewards/nexto_style.py.

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

    # Reward function: Nexto-style 10-component base + 4 custom RL-physics
    # rewards stacked on top, all wrapped in ZeroSumReward.
    #
    # The Nexto base (built unwrapped so we can compose more components into
    # the same CombinedReward) covers general offense, defense, positioning,
    # and event-based scoring. The 4 custom rewards on top exploit hardcoded
    # game knowledge (Octane physics, big boost pad positions, ball/car
    # height thresholds, defensive geometry) to surface skills that emerge
    # slowly from the base alone:
    #
    #   - SupersonicReward       → use boost meaningfully, play at pace
    #   - AerialBallReward       → go up when ball is up
    #   - BigBoostProximityReward → grab 100-pads when empty
    #   - BackboardDefenseReward → shadow defense positioning
    #
    # See src/rlbot/rewards/custom_rl.py and src/rlbot/utils/rl_constants.py
    # for the physics values these rewards reference.
    from rlgym_sim.utils.reward_functions import CombinedReward

    from rlbot.rewards.custom_rl import (
        AerialBallReward,
        AerialTouchReward,
        BackboardDefenseReward,
        BallAwayFromOwnGoalReward,
        BigBoostProximityReward,
        SupersonicReward,
    )
    from rlbot.rewards.nexto_style import build_nexto_style_reward
    from rlbot.rewards.zero_sum import ZeroSumReward

    # Build the Nexto base WITHOUT its own zero-sum wrapper so we can stack
    # additional components alongside it and wrap the whole thing once at the end.
    nexto_base = build_nexto_style_reward(zero_sum=False)

    # Tuned reward stack (v2). Changes from the original nexto_plus_kickoff_512
    # reward set, based on watching the 202M bot play:
    #   - AerialTouchReward (NEW, weight 1.5): rewards ACTUAL aerial ball
    #     contact, not just being airborne. Diego wanted stronger aerial play.
    #   - BigBoostProximityReward (now ball-distance aware, weight bumped
    #     0.3 → 0.5): only rewards grabbing a big pad when the ball is far,
    #     so the bot does not abandon plays to chase boost. Diego wanted more
    #     boost reward but conditioned on ball distance.
    #   - BallAwayFromOwnGoalReward (NEW, weight 0.6): penalizes the ball
    #     heading toward the bot's own net in its defensive half — directly
    #     attacks the own-goal-under-pressure behavior seen vs Marian's bot.
    combined = CombinedReward(
        reward_functions=(
            nexto_base,                   # whole 10-component Nexto stack as one unit
            SupersonicReward(),
            AerialBallReward(),
            AerialTouchReward(),          # NEW — real aerial contact
            BigBoostProximityReward(),    # MODIFIED — ball-distance aware
            BackboardDefenseReward(),
            BallAwayFromOwnGoalReward(),  # NEW — anti-own-goal
        ),
        reward_weights=(
            1.0,    # nexto base counts as 1x (already weighted internally to ~12 max)
            0.05,   # supersonic: cheap, fires often when boosting hard
            0.5,    # aerial position: sparse but valuable when it fires
            1.5,    # aerial TOUCH: strong reward for actual aerial hits
            0.5,    # big-boost proximity: ball-distance aware, bumped from 0.3
            0.4,    # backboard defense: only fires in defensive scenarios
            0.6,    # ball-away-from-own-goal: anti own-goal under pressure
        ),
    )
    reward_fn = ZeroSumReward(combined, team_spirit=0.0, opp_scale=1.0)

    # State setter: 70% wild RandomState (broad exploration of states the
    # bot might face mid-game) + 30% RandomKickoffSetter (forces practice
    # of the 5 canonical kickoff positions using hardcoded coordinates from
    # rl_constants.py). This is the curriculum-learning side of the new
    # experiment — instead of only seeing chaos, the bot also does dedicated
    # kickoff drills, which is the single highest-leverage situation in 1v1.
    from rlbot.state_setters.kickoff_scenarios import RandomKickoffSetter
    from rlbot.state_setters.weighted_sample_setter import WeightedSampleSetter

    state_setter = WeightedSampleSetter(
        state_setters=[
            RandomState(
                ball_rand_speed=True,
                cars_rand_speed=True,
                cars_on_ground=False,
            ),
            RandomKickoffSetter(),
        ],
        weights=[0.7, 0.3],
    )

    # Build and return the environment.
    return rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=1,                 # 1v1 final, so one car per side
        spawn_opponents=True,        # spawn a second car for self play
        reward_fn=reward_fn,
        obs_builder=DefaultObs(),
        state_setter=state_setter,
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
# Auto resume from the latest saved checkpoint WITHIN the current experiment.
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
#     <EXPERIMENT_NAME>/                  base folder for this experiment
#       <EXPERIMENT_NAME>-<unix_ts>/      one folder per training session
#         100000/                          one folder per save_every_ts hit
#           PPO_POLICY.pt + others
#         200008/
#         ...
#     <OTHER_EXPERIMENT_NAME>/            other experiments live alongside
#       ...
#
# We scan only the current experiment's folder so different experiments
# never resume from each other's weights. Returns None if no checkpoint
# exists yet for this experiment, in which case training starts fresh.
# --------------------------------------------------------------------------
def find_latest_checkpoint(experiment: str = EXPERIMENT_NAME) -> str | None:
    base_path = Path("diego-bots/checkpoints") / experiment
    if not base_path.exists():
        return None
    # Scan any direct subdirectory of the experiment folder. We accept folder
    # names that don't start with the experiment string so that archived
    # checkpoints with their original naming (e.g. simple_bot-<ts>/) still
    # resolve when you switch EXPERIMENT_NAME back to compare.
    runs = sorted(
        [p for p in base_path.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        timesteps = [p for p in run.iterdir() if p.is_dir() and p.name.isdigit()]
        if timesteps:
            latest = max(timesteps, key=lambda p: int(p.name))
            return str(latest)
    return None


# --------------------------------------------------------------------------
# Per-session run summary — saved to history_and_summary/ on every exit.
#
# Each Ctrl+C (or normal end) writes a JSON snapshot of the session: which
# experiment, which wandb run, final metrics, cumulative timesteps, the
# reason for stopping, and a snapshot of the important config knobs.
#
# Files are globally numbered so later analysis can ingest the whole history
# in order:  history_and_summary/run_001_<experiment>.json
#            history_and_summary/run_002_<experiment>.json  etc.
# --------------------------------------------------------------------------
def _format_steps(n: int) -> str:
    """Format a cumulative timestep count as 'M' up to 999M and 'B' from 1B.
    Used by both the at-start wandb run name and the at-end rename."""
    if n >= 1_000_000_000:
        if n % 1_000_000_000 == 0:
            return f"{n // 1_000_000_000}B"
        return f"{n / 1_000_000_000:.1f}B"
    return f"{round(n / 1_000_000)}M"


def _next_run_number() -> int:
    """Next sequential run number for both the JSON filename and the
    wandb `_rN` suffix. Both share the same number so files and dashboard
    entries always align (run_003_*.json on disk == r3 in wandb)."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for f in HISTORY_DIR.glob("run_*.json"):
        try:
            numbers.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    if not numbers:
        return STARTING_RUN_NUMBER
    return max(numbers) + 1


def _capture_config() -> dict:
    """Snapshot of the levers worth A/B comparing across runs.

    Update this whenever you change a knob you want to track in the history.
    Not exhaustive on purpose — captures the important stuff for analysis,
    not every internal default of rlgym_ppo.
    """
    return {
        "experiment_name": EXPERIMENT_NAME,
        # we changing to 16 as we have 24 logical processors used to be 14
        "n_proc": 16,
        "ppo_batch_size": 100_000,
        "ts_per_iteration": 100_000,
        "ppo_minibatch_size": 100_000,
        "exp_buffer_size": 300_000,
        # changing here to 180 used to be 140 
        "min_inference_size": 180,
        "ppo_epochs": 2,
        "ppo_ent_coef": 0.01,
        "policy_layer_sizes": [512, 512, 512],
        "critic_layer_sizes": [512, 512, 512],
        "standardize_returns": True,
        "standardize_obs": False,
        "save_every_ts": 100_000,
        "reward_function": "ZeroSumReward(CombinedReward(nexto_base [unwrapped] + custom_rl rewards))",
        "reward_components": [
            "nexto_base (10-component Nexto-style, weight 1.0)",
            "SupersonicReward (weight 0.05)",
            "AerialBallReward (weight 0.5)",
            "AerialTouchReward (weight 1.5) [v2: real aerial contact]",
            "BigBoostProximityReward (weight 0.5) [v2: ball-distance aware]",
            "BackboardDefenseReward (weight 0.4)",
            "BallAwayFromOwnGoalReward (weight 0.6) [v2: anti own-goal]",
            "ZeroSumReward wrapping (team_spirit=0, opp_scale=1)",
        ],
        "reward_version": "v2 (tuned after 202M: +aerial touch, ball-distance boost, anti-own-goal)",
        "nexto_base_components": [
            "VelocityPlayerToBallReward (weight 0.6)",
            "LiuDistancePlayerToBallReward (weight 0.7)",
            "VelocityBallToGoalReward (weight 2.0)",
            "LiuDistanceBallToGoalReward (weight 1.0)",
            "AlignBallGoal(defense=1, offense=1) (weight 0.4)",
            "BallYCoordinateReward (weight 0.5)",
            "FaceBallReward (weight 0.3)",
            "TouchBallReward (weight 5.0)",
            "SaveBoostReward (weight 0.05)",
            "EventReward(goal=10, concede=-10, shot=1.5, save=3, touch=0.05, demo=0.5, boost_pickup=0.3) (weight 12.0)",
        ],
        "obs_builder": "DefaultObs",
        "action_parser": "LookupAction (90 discrete actions)",
        "state_setter": "WeightedSampleSetter(RandomState 0.7 + RandomKickoffSetter 0.3)",
    }


def _serializable_wandb_summary() -> dict:
    """Best-effort capture of wandb.run.summary as plain JSON-safe values."""
    try:
        import wandb
    except ImportError:
        return {}
    if wandb.run is None:
        return {}
    out: dict = {}
    try:
        items = list(wandb.run.summary.items())
    except Exception:
        return {}
    for key, value in items:
        if str(key).startswith("_"):
            continue
        try:
            json.dumps(value)
            out[key] = value
        except (TypeError, ValueError):
            try:
                out[key] = float(value)
            except Exception:
                out[key] = str(value)
    return out


def _rename_wandb_run(entity: str, project: str, run_id: str, new_name: str) -> None:
    """Rename a wandb run to its final story-friendly name via the public API.

    We need to use wandb's REST API (wandb.Api) rather than wandb.run.name
    because by the time this runs, the rlgym_ppo Learner has already called
    wandb.finish() and the in-process run handle is closed.
    """
    try:
        import wandb

        api = wandb.Api()
        wb_run = api.run(f"{entity}/{project}/{run_id}")
        wb_run.name = new_name
        wb_run.save()
        print(f"[wandb] renamed run to {new_name}")
    except Exception as exc:
        print(f"[wandb] rename skipped: {exc}")


# Module-level flag so save_run_summary is idempotent. Multiple paths can
# trigger it (try/except, finally, atexit) — but it should only write once.
_summary_saved: bool = False


def save_run_summary(run_name: str, started_at: datetime, stop_reason: str) -> None:
    """Write a single JSON file describing this training session, and rename
    the wandb run to the final story-friendly name based on cumulative end
    timesteps.

    Idempotent — safe to call multiple times. Subsequent calls are no-ops.
    """
    global _summary_saved
    if _summary_saved:
        return
    _summary_saved = True

    ended_at = datetime.now(timezone.utc)
    run_number = _next_run_number()

    summary = {
        "run_number": run_number,
        "experiment_name": EXPERIMENT_NAME,
        "wandb_run_name": run_name,
        "wandb_final_name": None,  # filled in after we rename
        "stop_reason": stop_reason,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 1),
        "wandb_run_id": None,
        "wandb_run_url": None,
        "wandb_entity": None,
        "wandb_project": None,
        "cumulative_timesteps_end": None,
        "final_metrics": _serializable_wandb_summary(),
        "config": _capture_config(),
    }

    # wandb run id + url + entity + project, if still accessible after Ctrl+C
    try:
        import wandb
        if wandb.run is not None:
            summary["wandb_run_id"] = wandb.run.id
            summary["wandb_run_url"] = wandb.run.url
            summary["wandb_entity"] = wandb.run.entity
            summary["wandb_project"] = wandb.run.project
    except Exception:
        pass

    # Pull cumulative_timesteps from the last on-disk checkpoint, which is
    # the most authoritative source even if wandb is misbehaving.
    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt:
        bk_path = Path(latest_ckpt) / "BOOK_KEEPING_VARS.json"
        if bk_path.exists():
            try:
                bk = json.loads(bk_path.read_text())
                summary["cumulative_timesteps_end"] = bk.get("cumulative_timesteps")
            except Exception:
                pass

    # Rename the wandb run to f"{prefix}_{end_ts_M_or_B}_r{N}", e.g.
    # baseline_70M_r3. Only when we have all three required pieces: run_id,
    # entity/project, and cumulative_timesteps_end.
    end_ts = summary["cumulative_timesteps_end"]
    if (
        summary["wandb_run_id"]
        and summary["wandb_entity"]
        and summary["wandb_project"]
        and end_ts is not None
    ):
        final_name = f"{WANDB_NAME_PREFIX}_{_format_steps(int(end_ts))}_r{run_number}"
        _rename_wandb_run(
            summary["wandb_entity"],
            summary["wandb_project"],
            summary["wandb_run_id"],
            final_name,
        )
        summary["wandb_final_name"] = final_name

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HISTORY_DIR / f"run_{run_number:03d}_{EXPERIMENT_NAME}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[summary] saved run #{run_number} to {out_path}")


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

    print(f"[experiment] {EXPERIMENT_NAME}")
    resume_from = find_latest_checkpoint()
    if resume_from:
        print(f"[resume] loading checkpoint: {resume_from}")
    else:
        print(f"[fresh] no checkpoint found for experiment '{EXPERIMENT_NAME}', training from scratch")

    # Build the wandb run name as <experiment>_<XM_or_B>. Each experiment's
    # runs cluster together in the dashboard and overlay cleanly on charts.
    # Format examples:
    #     richer_rewards_0M       (fresh start of the richer_rewards experiment)
    #     richer_rewards_11M      (resumed at ~11M)
    #     baseline_11M            (the previous baseline plateau)
    #     nexto_rewards_500M      (a future experiment)
    # Add a suffix in the wandb UI after the run finishes if you want to tag
    # it (_PLATEAU, _BREAKTHROUGH, _REWARD_SHAPED, etc.) for the final report.
    _start_ts = 0
    if resume_from:
        try:
            _start_ts = int(Path(resume_from).name)
        except ValueError:
            _start_ts = 0

    _run_name = f"{EXPERIMENT_NAME}_{_format_steps(_start_ts)}"
    print(f"[wandb] run name: {_run_name}")

    learner = Learner(
        build_env,

        # How many copies of the env to run in parallel. Each one lives in
        # its own process. More workers = more samples per second, but
        # more RAM and CPU contention. Rule of thumb: ~1.5x physical cores.
        # Diego's machine has 12 physical / 24 logical cores, so 14 leaves
        # plenty of headroom for OS / browser / training side-tools.
        # Was 8 — bumped to 14 to push Overall Steps/sec from ~7k toward 12k+.
        n_proc=16,

        # The Learner batches inference requests across workers for GPU
        # efficiency. min_inference_size says: do not run the policy
        # forward pass until at least this many env steps are queued up.
        # Bumped from 80 → 140 to match the higher n_proc — larger GPU
        # batches per forward pass = better GPU utilization.

        # now bumped to 180 due to my GPU being better (rtx 4070)
        min_inference_size=180,

        # Optional callback for per iteration metrics. None means default
        # console reporting only.
        metrics_logger=None,

        # PPO algorithm hyperparameters. These are the actual learning levers.
        #
        # Batches bumped from 50k → 100k. Rationale: at ~12k Steps/sec with
        # n_proc=14, a 50k batch fills in ~4s — too short, the PPO update
        # overhead becomes a larger fraction of iteration time. 100k batches
        # let each rollout phase last ~8s, smoothing the iteration cadence
        # and giving the gradient a more stable signal per update. Combined
        # with the higher throughput, you get bigger learning steps per
        # iteration (was ~7k effective, now ~11-15k).
        ppo_batch_size=100_000,

        # ts_per_iteration: keep equal to batch size for simplicity (full
        # on-policy batch per update with no slack).
        ts_per_iteration=100_000,

        # exp_buffer_size: rolling buffer of recent experience. Stay at 3x
        # batch size so a small amount of slightly off-policy data is
        # available without overwhelming the on-policy assumption.
        exp_buffer_size=300_000,

        # ppo_minibatch_size: gradient descent batch size inside the PPO
        # update. Keep equal to ppo_batch_size for full-batch updates (the
        # simplest and most stable choice for PPO).
        ppo_minibatch_size=100_000,

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

        # Neural network architecture: medium (512x3), matching Marian's 1.35B
        # bot. Used because we now have a 14-component reward signal that
        # gives the bigger network meaningful gradient. ~15-20% slower per
        # timestep vs 256x3 but higher skill ceiling. The previous
        # nexto_plus_kickoff experiment (256x3, 17.6M) stays preserved in its
        # own folder as a comparison baseline.
        policy_layer_sizes=(512, 512, 512),
        critic_layer_sizes=(512, 512, 512),

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

        # Where to write checkpoints. Each experiment gets its own subfolder
        # so different experiments do not pollute each other. The Learner
        # appends a unix timestamp inside that folder for per-session uniqueness:
        #     diego-bots/checkpoints/<EXPERIMENT_NAME>/<EXPERIMENT_NAME>-<unix_ts>/
        # The cumulative timestep counter in BOOK_KEEPING_VARS.json continues
        # from where the loaded checkpoint left off, even across sessions.
        checkpoints_save_folder=f"diego-bots/checkpoints/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}",

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
        wandb_run_name=_run_name,    # self-describing per-session name

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
    #
    # learner.learn() is wrapped in BOTH try/finally AND atexit so the session
    # summary lands no matter how the run ends:
    #   - "completed"        : timestep_limit reached
    #   - "keyboard_interrupt": Ctrl+C
    #   - "exception: ..."    : something blew up (rare, mostly rocketsim crashes)
    #   - "atexit_fallback"   : process exited bypassing the try/finally
    #                           (rlgym_ppo sometimes does this via os._exit on
    #                           multiprocess teardown — atexit catches it)
    #
    # The summary is idempotent (see save_run_summary's `_summary_saved` flag)
    # so the multiple registration paths never produce duplicate entries.
    _session_started_at = datetime.now(timezone.utc)
    _session_stop_reason = "completed"

    # Register atexit FIRST so it fires even if try/finally is bypassed by
    # process-exit paths the Learner takes during multiprocess teardown.
    atexit.register(
        save_run_summary,
        _run_name,
        _session_started_at,
        "atexit_fallback",
    )

    try:
        learner.learn()
    except KeyboardInterrupt:
        _session_stop_reason = "keyboard_interrupt"
        print("\n[interrupted] training stopped by user — saving session summary...")
        save_run_summary(_run_name, _session_started_at, _session_stop_reason)
    except Exception as _exc:
        _session_stop_reason = f"exception: {type(_exc).__name__}: {_exc}"
        save_run_summary(_run_name, _session_started_at, _session_stop_reason)
        raise
    finally:
        save_run_summary(_run_name, _session_started_at, _session_stop_reason)
