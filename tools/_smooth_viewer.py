"""Smooth rlviser viewer: steps 4 physics ticks per render and lets rlviser interpolate.

Rendering EVERY tick (tick_skip=1) is IPC/overhead-bound and caps the sim at ~0.6x real time
(the slow-mo). Instead we step 4 ticks per env.step (~30 states/sec) and rely on rlviser's
PacketSmoothing=Interpolate to fill the gaps -> smooth AND real-time. The loop is paced to
wall-clock and prints the measured playback speed so we can prove it is not slow-mo.
The policy still acts every 8 ticks (correct behaviour). Camera stays behind the blue car
(set ball cam OFF in rlviser).

Run from repo root, PYTHONPATH=src:
  python tools/_smooth_viewer.py --orange <opp> --episodes 100 --speed 1.0 --deterministic
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "tools")

import torch
from papaya_1v1_viewer import DEFAULT_BLUE, _action_to_int, load_policy, resolve_checkpoint

import ctypes
try:
    ctypes.windll.winmm.timeBeginPeriod(1)   # accurate sleep -> real-time (kills the ~0.65x slow-mo)
except Exception:
    pass

POLICY_TICK = 8          # policy acts every 8 physics ticks (correct behaviour)
ENV_TICK_SKIP = 4        # step 4 ticks per render -> ~30 states/sec, rlviser interpolates the rest
ACT_EVERY = POLICY_TICK // ENV_TICK_SKIP   # recompute the action every 2 env.steps
BACK = 5120.0


def _attack_state():
    """A state setter that sets the champion (blue, team 0) up to shoot at the orange goal:
    ball in the attacking third (often airborne for aerials), champion behind it boosted,
    Nachi defending its net. Yields frequent cool finishes."""
    from rlgym_sim.utils.state_setters import StateSetter

    class AttackState(StateSetter):
        def reset(self, sw):
            # Proven aerial setup: ball launched UP fast from low height, both cars boosted on
            # the ground facing it -> both FLY up to challenge (champion reaches ~1000uu). Placed
            # in the attacking third so the champion's aerial heads toward the orange goal.
            bx = float(np.random.uniform(-900, 900))
            by = float(np.random.uniform(1100, 2300))
            sw.ball.set_pos(bx, by, 300.0)
            sw.ball.set_lin_vel(float(np.random.uniform(-150, 150)), float(np.random.uniform(120, 520)), 1400.0)
            sw.ball.set_ang_vel(0.0, 0.0, 0.0)
            for car in sw.cars:
                if car.team_num == 0:                       # champion below the ball, facing the goal
                    car.set_pos(bx + float(np.random.uniform(-250, 250)), by - 1500.0, 17.0)
                    car.set_rot(yaw=0.5 * np.pi)
                else:                                       # Nachi on the far side, defending
                    car.set_pos(bx + float(np.random.uniform(-250, 250)), by + 1500.0, 17.0)
                    car.set_rot(yaw=-0.5 * np.pi)
                car.boost = 1.0; car.set_lin_vel(0.0, 0.0, 0.0); car.set_ang_vel(0.0, 0.0, 0.0)
    return AttackState()


def _mixed_state(attack_frac=0.5):
    from rlgym_sim.utils.state_setters import DefaultState, StateSetter
    attack, default = _attack_state(), DefaultState()

    class MixedState(StateSetter):
        def reset(self, sw):
            (attack if np.random.random() < attack_frac else default).reset(sw)
    return MixedState()


def _snap(state):
    b = state.ball
    cars = [(float(p.car_data.position[0]), float(p.car_data.position[1]), float(p.car_data.position[2]),
             int(p.team_num), bool(p.ball_touched)) for p in state.players]
    return (b.position[2], math.sqrt(float(b.linear_velocity[0]) ** 2 + float(b.linear_velocity[1]) ** 2
            + float(b.linear_velocity[2]) ** 2), cars)


def _coolness(traj):
    spd = max((s[1] for s in traj[-14:]), default=0.0)
    air = max((s[0] for s in traj[-45:]), default=0.0)
    dist = 0.0
    for s in reversed(traj):
        ch = [c for c in s[2] if c[4] and c[3] == 0]
        if ch:
            dist = math.hypot(ch[0][0], ch[0][1] - BACK); break
    return 0.45 * (spd / 6000) + 0.30 * (air / 2000) + 0.25 * (dist / 7000), spd, air, dist


def build_smooth_env():
    import rlgym_sim
    from rlgym_sim.utils.obs_builders import AdvancedObs
    from rlgym_sim.utils.reward_functions import DefaultReward
    from rlgym_sim.utils.state_setters import DefaultState
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        GoalScoredCondition, TimeoutCondition,
    )
    from rlbot.actions.lookup_action import LookupAction
    # NO GoalScoredCondition: we DON'T want the episode to end the instant the ball crosses the
    # line. We detect the goal ourselves and let the sim KEEP PLAYING a couple seconds of natural
    # follow-through (ball settling in the net) before resetting -> smoother clip endings, no hard
    # cut. TimeoutCondition counts env.steps: 480 * 4 = 1920 ticks ~= 16s hard cap for plays that
    # never score, so games still cycle fast.
    del GoalScoredCondition
    return rlgym_sim.make(
        tick_skip=ENV_TICK_SKIP, team_size=1, spawn_opponents=True,
        obs_builder=AdvancedObs(), action_parser=LookupAction(), reward_fn=DefaultReward(),
        state_setter=_mixed_state(0.6),     # 60% aerial-attack, 40% real kickoff
        terminal_conditions=[TimeoutCondition(480)],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orange", required=True)
    ap.add_argument("--blue", default=DEFAULT_BLUE)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 real time, 0.6 = slow-mo")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--goal-hold", type=float, default=2.5,
                    help="keep PLAYING this many seconds after a goal (natural follow-through, not a freeze)")
    ap.add_argument("--goal-log", default="highlights_3d/_goals.jsonl",
                    help="append each goal (either bot, wall-clock + coolness) here for auto-clipping")
    a = ap.parse_args()

    blue, _ = load_policy(resolve_checkpoint(a.blue))
    orange, _ = load_policy(resolve_checkpoint(a.orange))
    env = build_smooth_env()
    # wall-clock seconds one env.step (ENV_TICK_SKIP ticks of game time) SHOULD take at this speed
    frame_dt = (ENV_TICK_SKIP / 120.0) / max(0.1, a.speed)
    game_per_step = ENV_TICK_SKIP / 120.0          # game seconds advanced per env.step
    det = a.deterministic
    glog = Path(a.goal_log); glog.parent.mkdir(exist_ok=True)
    glog.write_text("")  # fresh log per run
    print(f"Smooth viewer up (tick_skip={ENV_TICK_SKIP}, target {a.speed:.2f}x). "
          "In rlviser: press Esc to close the menu, keep BALL CAM OFF (behind-car).", flush=True)
    next_t = time.perf_counter()                   # frame-pacing clock
    pace_t0 = time.perf_counter(); pace_steps = 0; pace_last = pace_t0
    for ep in range(1, a.episodes + 1):
        obs = env.reset(); held = [0, 0]; step = 0; done = False; info = {}
        peak_blue = 0.0; peak_orange = 0.0             # peak car height per team this episode
        shot_spd = 0.0; goal = None                    # goal record (set when the ball crosses the line)
        shake = 0; prev_sign = 0                        # shake = blue (followed car) yaw-rate reversals
        while not done:
            if step % ACT_EVERY == 0:
                with torch.no_grad():
                    ba, _ = blue.get_action(obs[0], deterministic=det)
                    oa, _ = orange.get_action(obs[1], deterministic=det)
                held = [_action_to_int(ba), _action_to_int(oa)]
            obs, _r, done, info = env.step(held)
            st = env._prev_state
            for p in st.players:                       # track how high each side flew (aerial-ness)
                z = float(p.car_data.position[2])
                if p.team_num == 0:
                    peak_blue = z if z > peak_blue else peak_blue
                    avz = float(p.car_data.angular_velocity[2])     # yaw rate of the camera-followed car
                    s = 1 if avz > 1.5 else (-1 if avz < -1.5 else 0)
                    if s != 0 and prev_sign != 0 and s != prev_sign:
                        shake += 1                     # a hard left<->right reversal = camera whip
                    if s != 0:
                        prev_sign = s
                else:
                    peak_orange = z if z > peak_orange else peak_orange
            # detect the goal the instant the ball crosses the line (robust via ball-in-net y)
            if goal is None:
                ball_y = float(st.ball.position[1]); res = int(info.get("result", 0))
                sb = res > 0 or ball_y > 5050          # champion scored into the orange net (+y)
                so = res < 0 or ball_y < -5050         # nachi scored into the blue net (-y)
                if sb or so:
                    bv = st.ball.linear_velocity
                    shot_spd = math.sqrt(float(bv[0]) ** 2 + float(bv[1]) ** 2 + float(bv[2]) ** 2)
                    scorer = "MARTIN 10B" if sb else "NACHI"
                    speak = peak_blue if sb else peak_orange
                    goal = {"t": time.time(), "type": "goal", "scorer": scorer,
                            "champ_z": round(speak), "ball_spd": round(shot_spd),
                            "aerial": speak > 350, "goal": True,
                            "score": round(0.5 + speak / 2000.0 + shot_spd / 12000.0, 3)}
                    print(f"Ep {ep}/{a.episodes}  GOAL by {scorer}  height={speak:.0f} "
                          f"spd={shot_spd:.0f} score={goal['score']}", flush=True)
                    # the sim auto-resets the ball to kickoff the instant a goal scores, so we CAN'T
                    # keep playing the follow-through. Instead linger on the goal moment (ball in the
                    # net) by re-rendering this state -> no instant cut, no kickoff flash, smooth.
                    hold_until = time.perf_counter() + a.goal_hold
                    while time.perf_counter() < hold_until:
                        env.render(); time.sleep(frame_dt)
                    next_t = time.perf_counter()
                    break
            env.render(); step += 1
            # pace to wall-clock: sleep the remaining budget, or if we're behind don't spiral
            next_t += frame_dt
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.perf_counter()
            # measure + report the ACTUAL playback speed every ~2s (proves it is not slow-mo)
            pace_steps += 1
            now = time.perf_counter()
            if now - pace_last >= 2.0:
                actual = (pace_steps * game_per_step) / (now - pace_t0)
                print(f"[pace] actual={actual:.2f}x  target={a.speed:.2f}x  "
                      f"steps/s={pace_steps / (now - pace_t0):.0f}", flush=True)
                pace_last = now
        if goal is not None:                           # log the goal (shake recorded for smoother selection)
            goal["shake"] = shake
            with glog.open("a") as fh:
                fh.write(json.dumps(goal) + "\n")
        elif peak_blue > 700:                          # notable champion aerial (no goal) = fallback clip
            with glog.open("a") as fh:
                fh.write(json.dumps({"t": time.time(), "type": "aerial", "scorer": "MARTIN 10B",
                                     "champ_z": round(peak_blue), "ball_spd": round(shot_spd),
                                     "aerial": True, "goal": False, "shake": shake,
                                     "score": round(0.2 + peak_blue / 2000.0, 3)}) + "\n")
            print(f"Ep {ep}/{a.episodes}  aerial MARTIN 10B height={peak_blue:.0f} (no goal)", flush=True)
        else:
            print(f"Ep {ep}/{a.episodes}  -", flush=True)
    env.close()


if __name__ == "__main__":
    main()
