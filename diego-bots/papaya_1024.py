"""
papaya_1024 — 1v1 PPO bot, LARGE 1024x3 architecture variant. Educational scaffold.

Sibling of simple_bot.py, but long since diverged: papaya runs AdvancedObs
(107-dim), the v5 reward stack (boost conservation + anti-overcommit +
recovery/flick mechanics), aerial/dribble drill state setters, and the v6
optimizer tune (ppo_epochs=3, ent_coef=0.005, explicit LRs — see
docs/papaya_v6_overnight_tune.md). simple_bot.py remains the 512x3
DefaultObs baseline for comparison.

This is a FRESH-FROM-SCRATCH run, not a warm-start. It does NOT inherit the
512x3 bot's learned strategy: that strategy lives entirely in the trained
weights, and a 512x3 checkpoint cannot be loaded into a 1024x3 net
(rlgym_ppo's PPOLearner.load_from() calls load_state_dict() with strict=True,
ppo_learner.py:260, so mismatched layer shapes raise a RuntimeError). What it
DOES share with the 512 run is the training recipe — the reward function, obs,
actions, and state setter — i.e. the same incentives, so it will rediscover a
similar strategy on its own, but from random init. It auto-resumes only from
its OWN checkpoints once it has saved some.

Run from the project root with the rlbot310 conda env activated:

    conda activate rlbot310
    python diego-bots/papaya_1024.py

Output:
    diego-bots/checkpoints/papaya_1024/<run>/<timestep>/   policy snapshots
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
# papaya_1024: same v3 reward stack as nexto_plus_kickoff_512, but LARGE
# 1024x3 architecture (4x the params of 512x3). Architecture is the only
# changed variable vs the 512 run, so the two overlay as a clean A/B in wandb.
# Previous experiments are preserved in their own folders for comparison:
#   - nexto_rewards/         : 5→10 component reward, 256x3, reached 130M
#   - nexto_plus_kickoff/     : 14-component reward, 256x3, reached 17.6M
#   - nexto_plus_kickoff_512/ : v3 reward, 512x3, reached 1.18B (current best)
EXPERIMENT_NAME = "papaya_1024"

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
WANDB_NAME_PREFIX = "papaya1024"
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

# AdvancedObs converts the raw game state into the fixed-length numeric vector
# we feed the policy. It is the richer obs: on top of DefaultObs's absolute
# positions/velocities it adds car->ball and car->opponent RELATIVE position &
# velocity, giving the policy directly usable spatial relationships. 1v1 obs
# size is 107 (vs DefaultObs's 89). The shape is fixed at training time;
# changing the obs builder means a fresh bot (papaya is fresh, so that's fine).
# This is a custom rlgym_sim-compatible builder — rlgym_tools 2.6.4 has no
# drop-in AdvancedObs for the rlgym_sim API. See src/rlbot/obs/advanced_obs.py.
from rlbot.obs.advanced_obs import AdvancedObs

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

# v7: ends episodes where the kickoff never resolves (ball still in the center
# zone after ~4s), so kickoff drills recycle faster. Arms only until the ball
# first leaves the center radius -- can never terminate normal mid-game play.
from rlbot.terminal.kickoff_stall import KickoffStallCondition

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
        BoostReserveReward,
        DribbleToGoalReward,
        FlickReward,
        KickoffReward,
        RecoveryReward,
        SupersonicReward,
    )
    from rlbot.rewards.nexto_style import build_nexto_style_reward
    from rlbot.rewards.zero_sum import ZeroSumReward

    # Build the Nexto base WITHOUT its own zero-sum wrapper so we can stack
    # additional components alongside it and wrap the whole thing once at the end.
    # v5: lower the ball-chase signals (curb overcommitting — bot was constantly
    # diving at the ball) and raise SaveBoost (conserve boost for recoveries).
    nexto_base = build_nexto_style_reward(
        zero_sum=False,
        velocity_player_to_ball_weight=0.3,   # was 0.6 default
        liu_distance_player_to_ball_weight=0.3,  # was 0.7 default
        save_boost_weight=0.3,                # was 0.05 default
    )

    # Tuned reward stack (v3). Changes from v2, based on watching the 635M bot
    # play in rlviser (65% win rate vs Marian's 1.35B bot):
    #   - DribbleToGoalReward (NEW, weight 0.15): the bot dribbles well but into
    #     the side walls/poles. This nudges the dribble TOWARD the enemy net by
    #     scaling possession reward by enemy-goal proximity. Small weight — a
    #     directional nudge, not a dominant signal.
    #   - KickoffReward (NEW, weight 0.3): kickoffs are already a major source of
    #     goals; this rewards committing hard to the ball off kickoff (first touch
    #     while the ball is still near center) to push that strength further.
    #   - MaintainSpeedReward (NEW, weight 0.1): continuous speed reward saturating
    #     at supersonic. The bot drove at low pace; pros keep momentum so they can
    #     rotate back on defense instantly. Fills the gap below SupersonicReward's
    #     hard 2200 threshold.
    #   - BigBoostProximityReward (weight 0.5 → 0.8, threshold 0.30 → 0.40 in
    #     custom_rl.py): bot ignored big pads. Stronger pull toward boost when
    #     low, and it tops up proactively rather than running empty.
    #   - SupersonicReward (weight 0.05 → 0.1): reinforce playing at pace.
    #   - boost_pickup event bumped 0.3 → 0.6 in nexto_style.py (each pad grab is
    #     now a clearer positive event).
    # v4 (papaya): add two mechanics the stack was missing —
    #   - RecoveryReward (weight 0.15): orient upright + toward motion while
    #     airborne OUTSIDE an aerial play, so the bot stops tumbling after
    #     bumps/clears/challenges and lands control-ready. Continuous but
    #     air-gated; small weight so it just polishes recoveries.
    #   - FlickReward (weight 1.0): launching the ball up+forward toward the net
    #     out of a dribble — the 1v1 finishing mechanic. Sparse event reward, so
    #     a meaningful weight. Paired with DribbleSetupState below so the bot is
    #     in carries often enough to learn it.
    # These are reinforced by the new aerial/dribble state-setters (see below):
    # AerialTouchReward and FlickReward barely fire under random spawns alone.
    # v5 (after watching papaya @ 828M play 1v1 + entropy stuck at ~4.0):
    # the speed rewards were making it dump all its boost and the dense
    # ball-chase made it overcommit. Fix = REDUCE the conflicting signals, not
    # add more:
    #   - MaintainSpeedReward: REMOVED (it was the "always floor it" driver).
    #   - SupersonicReward: 0.25 → 0.03 (a whisper of pace, no longer a boost sink).
    #   - SaveBoost: 0.05 → 0.3 (inside nexto base) + BoostReserveReward (NEW 0.4):
    #     keep boost in reserve when not in an active play, ready to recover/save.
    #   - ball-chase (velocity/liu player-to-ball): 0.6/0.7 → 0.3/0.3 in nexto base
    #     to curb overcommitting.
    #   - BackboardDefenseReward: 0.45 → 0.7 (reward holding goal-side instead).
    #   - AerialTouchReward: 1.5 → 2.0 (+ aerial drill state bumped below) to push
    #     it to actually go up.
    combined = CombinedReward(
        reward_functions=(
            nexto_base,                   # whole Nexto stack as one unit (v5: ball-chase down, save-boost up)
            SupersonicReward(),
            AerialBallReward(),
            AerialTouchReward(),          # real aerial contact
            BigBoostProximityReward(),    # ball-distance aware
            BackboardDefenseReward(),
            BallAwayFromOwnGoalReward(),  # anti-own-goal
            DribbleToGoalReward(),        # v3 — dribble toward the enemy net
            KickoffReward(),              # v3 — win the kickoff
            RecoveryReward(),             # v4 — clean recoveries / land wheels-down
            FlickReward(),                # v4 — flick the ball off a dribble
            BoostReserveReward(),         # NEW v5 — keep boost for recoveries/saves
        ),
        reward_weights=(
            1.0,    # nexto base counts as 1x (already weighted internally)
            0.03,   # supersonic: v5 0.25→0.03, stop draining boost for raw speed
            0.6,    # aerial position: sparse but valuable when it fires
            2.0,    # aerial TOUCH: v5 1.5→2.0, push it to actually go up
            0.8,    # big-boost proximity: stronger pull to grab pads when low
            0.7,    # backboard defense: v5 0.45→0.7, hold goal-side vs overcommit
            0.6,    # ball-away-from-own-goal: anti own-goal under pressure
            0.20,   # dribble-to-goal: small directional nudge toward enemy net
            0.5,    # kickoff: v7 0.35->0.5 -- now time-decaying first-touch (pays once, early >> late)
            0.15,   # recovery: polish signal, air-gated, kept small
            1.0,    # flick: sparse finishing-mechanic event, meaningful weight
            0.4,    # NEW v5 boost-reserve: conserve boost when not in an active play
        ),
    )
    reward_fn = ZeroSumReward(combined, team_spirit=0.0, opp_scale=1.0)

    # State setter (v5): same curriculum mix, but MORE aerial reps and LESS pure
    # chaos (the wild RandomState was spawning the "weird scenarios" you saw).
    #   - RandomState        0.35 — broad coverage (v5 0.45→0.35, less unrealistic chaos)
    #   - RandomKickoffSetter 0.25 — the 5 canonical kickoffs (paired w/ KickoffReward)
    #   - AerialSetupState    0.25 — high ball + grounded cars w/ boost (v5 0.15→0.25,
    #                                more reps so it actually learns to aerial)
    #   - DribbleSetupState   0.15 — one car carrying the ball toward the net, so it
    #                                practices dribble control + FlickReward
    # Weights sum to 1.0.
    from rlbot.state_setters.kickoff_scenarios import RandomKickoffSetter
    from rlbot.state_setters.mechanic_scenarios import (
        AerialSetupState,
        DribbleSetupState,
    )
    from rlbot.state_setters.weighted_sample_setter import WeightedSampleSetter

    state_setter = WeightedSampleSetter(
        state_setters=[
            RandomState(
                ball_rand_speed=True,
                cars_rand_speed=True,
                cars_on_ground=False,
            ),
            RandomKickoffSetter(),
            AerialSetupState(),
            DribbleSetupState(),
        ],
        weights=[0.35, 0.25, 0.25, 0.15],
    )

    # Build and return the environment.
    return rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=1,                 # 1v1 final, so one car per side
        spawn_opponents=True,        # spawn a second car for self play
        reward_fn=reward_fn,
        obs_builder=AdvancedObs(),
        state_setter=state_setter,
        terminal_conditions=[
            GoalScoredCondition(),
            NoTouchTimeoutCondition(no_touch_ticks),
            # v7: kill kickoffs that never resolve (ball still in the center
            # zone after 4s) so kickoff drills recycle ~2.5x faster. Arms only
            # while the ball has never left the center radius, so it can never
            # cut normal mid-game play. See src/rlbot/terminal/kickoff_stall.py.
            KickoffStallCondition(
                max_kickoff_seconds=4.0,
                center_radius=300.0,
                tick_skip=tick_skip,
            ),
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
        "n_proc": 18,
        "ppo_batch_size": 100_000,
        "ts_per_iteration": 100_000,
        "ppo_minibatch_size": 100_000,
        "exp_buffer_size": 300_000,
        # changing here to 180 used to be 140
        "min_inference_size": 180,
        "ppo_epochs": 3,                     # v6: 2→3 (6→9 optimizer steps/iter)
        "ppo_ent_coef": 0.005,               # v7.1: reverted 0.0075→0.005 (chart: 0.005 is the 66% regime)
        "policy_lr": 3e-4,                   # v6: now explicit (unchanged value)
        "critic_lr": 3e-4,                   # v6: now explicit (unchanged value)
        "policy_layer_sizes": [1024, 1024, 1024],
        "critic_layer_sizes": [1024, 1024, 1024],
        "standardize_returns": True,
        "standardize_obs": False,
        "save_every_ts": 1_000_000,          # v6: 100k→1M (SSD wear / iteration time)
        "n_checkpoints_to_keep": 50,         # v6: 5→50 (overnight rollback window)
        "timestep_limit": 5_000_000_000,     # v6: 2B→5B (don't self-stop mid-grind)
        "training_tune_version": "v6 optimizer pass (2026-06-11): +1 ppo epoch, ent_coef halved, LRs pinned explicit, checkpoint retention overhaul. Rewards untouched (still v5). See docs/papaya_v6_overnight_tune.md",
        "reward_function": "ZeroSumReward(CombinedReward(nexto_base [unwrapped] + custom_rl rewards))",
        "reward_components": [
            "nexto_base (Nexto-style, weight 1.0) [v5: player-to-ball 0.6/0.7→0.3/0.3, save_boost 0.05→0.3]",
            "SupersonicReward (weight 0.03) [v5: 0.25→0.03, stop draining boost]",
            "AerialBallReward (weight 0.6)",
            "AerialTouchReward (weight 2.0) [v5: 1.5→2.0]",
            "BigBoostProximityReward (weight 0.8)",
            "BackboardDefenseReward (weight 0.7) [v5: 0.45→0.7, hold goal-side]",
            "BallAwayFromOwnGoalReward (weight 0.6)",
            "DribbleToGoalReward (weight 0.20)",
            "KickoffReward v2 (weight 0.5) [v7: time-decaying first-touch, pays once, early >> late, +directional term]",
            "RecoveryReward (weight 0.15) [v4]",
            "FlickReward (weight 1.0) [v4]",
            "BoostReserveReward (weight 0.4) [v5: keep boost for recoveries/saves]",
            "MaintainSpeedReward REMOVED [v5: was the boost-dumping driver]",
            "ZeroSumReward wrapping (team_spirit=0, opp_scale=1)",
        ],
        "reward_version": "v7 (fast-kickoff: KickoffReward v2 time-decaying first-touch @0.5, +KickoffStallCondition 4s; v5 boost/overcommit stack otherwise unchanged)",
        "terminal_conditions": "GoalScored + NoTouchTimeout(10s) + KickoffStallCondition(4s, r=300, armed-until-ball-leaves-center) [v7]",
        "nexto_base_components": [
            "VelocityPlayerToBallReward (weight 0.3) [v5: 0.6→0.3, anti-overcommit]",
            "LiuDistancePlayerToBallReward (weight 0.3) [v5: 0.7→0.3, anti-overcommit]",
            "VelocityBallToGoalReward (weight 2.0)",
            "LiuDistanceBallToGoalReward (weight 1.0)",
            "AlignBallGoal(defense=1, offense=1) (weight 0.4)",
            "BallYCoordinateReward (weight 0.5)",
            "FaceBallReward (weight 0.3)",
            "TouchBallReward (weight 5.0)",
            "SaveBoostReward (weight 0.3) [v5: 0.05→0.3, conserve boost]",
            "EventReward(goal=10, concede=-10, shot=1.5, save=3, touch=0.05, demo=0.5, boost_pickup=0.6) (weight 12.0)",
        ],
        "obs_builder": "AdvancedObs (107-dim, custom rlgym_sim-compatible; rel pos/vel to ball & opponent)",
        "action_parser": "LookupAction (90 discrete actions)",
        "state_setter": "WeightedSampleSetter(RandomState 0.35 + RandomKickoffSetter 0.25 + AerialSetupState 0.25 + DribbleSetupState 0.15)",
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
        # Was 8 — bumped to 14 to push Overall Steps/sec from ~7k toward 12k+
        # 
        # bumping to 18 (june 14th)
        n_proc=18,

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
        #
        # v6: 0.01 → 0.005. Policy entropy sat PINNED at ~4.0 (max ln(90)≈4.5)
        # for the entire 1.34B-step run — the policy never sharpened. Cutting
        # the bonus let it drop to ~3.53 and play improved (66% stochastic vs
        # Martin's 2.1B champion). LOWER entropy helped THIS (heavily-shaped)
        # stack.
        #
        # v7.1: briefly tried 0.0075 (a step toward Martin's 0.01), then REVERTED
        # to 0.005. The entropy chart settled it: ent_coef cleanly sets the
        # entropy equilibrium (0.01 -> ~4.0, 0.005 -> ~3.53), and ~3.53 is the
        # regime where this bot reached 66% vs Martin's champion. Raising the
        # coef just pulls entropy back UP toward the un-committed 4.0 state.
        # Martin runs 0.01 only because his rewards are lean; for our heavily
        # shaped stack, LOWER is the proven-good direction. Staying at 0.005.
        ppo_ent_coef=0.005,

        # ppo_epochs: how many passes over the experience buffer the PPO
        # update makes. NOTE the real math (verified in rlgym_ppo source):
        # optimizer steps per iteration = ppo_epochs * floor(exp_buffer_size /
        # ppo_batch_size) = epochs * 3 with our 300k buffer / 100k batch.
        #
        # v6: 2 → 3 (6 → 9 optimizer steps per iteration). Mean KL (~0.003)
        # and clip fraction (~0.03) ran 3-5x BELOW the healthy PPO band for
        # the entire run — updates were too conservative, which is the real
        # reason progress slowed. One extra epoch is a bounded 1.5x increase,
        # triple-guarded by clip_range 0.2, grad-norm clip 0.5, and PPO's
        # ratio clipping. Expect KL ~0.004-0.007 — still conservative.
        ppo_epochs=3,

        # Learning rates, passed EXPLICITLY as of v6 (previously left to the
        # library defaults — which are these same values, so NO change in
        # behavior). Pinned for two reasons: (1) rlgym_ppo re-applies the
        # constructor LRs after loading the checkpointed Adam state
        # (learner.py load() → update_learning_rate), so whatever stands here
        # is what actually runs after a resume; (2) the run-history JSONs now
        # record them. v6 deliberately does NOT cut the LR: with only a 1.5x
        # step increase and KL far below the danger zone, reducing it would
        # cancel the ppo_epochs change (this exact mistake was caught in
        # review — see the v6 doc).
        policy_lr=3e-4,
        critic_lr=3e-4,

        # standardize_returns: normalize advantage estimates by their
        # running standard deviation. Almost always helps stability. Turn
        # off only if you have a very specific reason to.
        standardize_returns=True,

        # standardize_obs: same idea but for the input observations.
        # DefaultObs is already roughly in a reasonable scale, so we leave
        # this off. Custom observation builders may want it on.
        standardize_obs=False,

        # Neural network architecture: LARGE (1024x3), 4x the params of the
        # 512x3 nexto_plus_kickoff_512 run. Higher skill ceiling at the cost of
        # a slower GPU forward pass per inference batch (rollout throughput is
        # CPU-bound on n_proc so it's only mildly affected). Trained fresh from
        # scratch — a 512x3 checkpoint can't be loaded into this wider net
        # (strict load_state_dict shape mismatch).
        policy_layer_sizes=(1024, 1024, 1024),
        critic_layer_sizes=(1024, 1024, 1024),

        # Total CUMULATIVE environment steps allowed before the Learner exits.
        # This is the sum across all training sessions, not just this one.
        # Set very high so resumed runs keep going until you Ctrl+C. The
        # Learner saves the running counter in BOOK_KEEPING_VARS.json so
        # resuming preserves it (your already trained 500k is still counted).
        # For a real strong bot you want tens of millions; 1 billion is just
        # "effectively unlimited until I stop it manually."

        # changed from 1B to 2B
        # v6: 2B → 5B. papaya is at ~1.35B and adds ~300M+ per overnight run —
        # at 2B the Learner would SILENTLY stop itself mid-grind within a
        # night or two. 5B = "until I Ctrl+C" for the remaining project window.
        timestep_limit=5_000_000_000,

        # Save a checkpoint every this many timesteps.
        # v6: 100k → 1M. At ~10k steps/sec, 100k meant a ~54MB checkpoint
        # write every ~10 SECONDS (~200GB of SSD writes per night, plus
        # iteration time lost to disk). 1M = a save every ~100s; a Ctrl+C or
        # crash loses at most ~2 minutes of training.
        save_every_ts=1_000_000,

        # v6 (NEW): keep the last 50 checkpoints instead of the default 5.
        # With the default, the rotation (learner.py:407-409) deletes all but
        # the newest 5 saves — by morning your only rollback points would be
        # the last few minutes or the pre-v6 archive, all-or-nothing. 50 x 1M
        # = a 50M-step (~1.4h) rollback window at ~2.7GB of disk. The pre-v6
        # full checkpoint is archived at
        # diego-bots/checkpoints/_archive/papaya_1024_PRE_V6_1.346B/.
        n_checkpoints_to_keep=50,

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
