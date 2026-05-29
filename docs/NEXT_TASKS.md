# NEXT TASKS

Concrete work items for the next session, in priority order. Read `docs/HANDOFF.md`
first for project state and `docs/LESSONS_LEARNED.md` for gotchas. Each task below has
enough spec to implement without re-deriving anything.

Diego's stated goal has escalated: **the final is now "billions vs billions"** — both his
bot and the opponent will be trained to billions of timesteps. So tasks 1-2 improve bot
quality; task 3 is about reaching that compute scale.

---

## FIELD OBSERVATIONS — 635M champion (2026-05-29)

Diego watched the 635M bot (65% win rate vs Marian's 1.35B) in rlviser and noted these
behaviors. They drive Tasks 4-6 below. Recording them verbatim with Diego's gamesense so
the next session understands the *why*, not just the *what*.

1. **Not collecting boost — especially the big 100-boost pads.** Diego's gamesense:
   Rocket League is a high-paced game. Pros maintain high speed constantly so they can
   rotate back to their own net for defense the instant the ball is cleared. A car that
   reaches high speed *keeps* that momentum without boost (RL physics: momentum carries;
   natural no-boost cap is 1410 uu/s, but flips/momentum preserve speed up toward
   supersonic 2200). The bot currently drives around at low pace and ignores big pads.
   **Want: higher reward for grabbing pads (esp. big 100 pads) AND higher reward for
   maintaining speed.** The car's velocity is available at `player.car_data.linear_velocity`
   (a 3-vector); its magnitude is current speed. → **Task 4.**

2. **Kickoffs are AMAZING — a major source of goals.** Keep this up; do not regress it.
   The specific remaining want is to implement **"fast kickoff" (speedflip) logic** so the
   bot reaches the ball even faster off kickoff. REMIND DIEGO to implement fast-kickoff
   logic — it was the source of many goals and is a clear strength to push further. → **Task 2 (expanded).**

3. **Shooting is good but sometimes excessive** — the bot shoots when it could wait, set
   up, and take a better shot. Diego doesn't know how to tune this. It's a subtle shot-
   selection / patience problem. Noted as an open problem with candidate approaches. → **Task 6.**

---

## Task 1 — Dribbling-toward-enemy-goal reward (HIGH priority, easy)

**Problem Diego observed (watching 416M bot in rlviser):** dribbling has improved
dramatically, BUT the bot dribbles the ball into the side walls/poles and loses it. It
has ball control but no sense of *where* to take the ball.

**Fix:** add a reward that scales ball-control/possession by **proximity to the enemy
goal** — small weight, so it nudges the dribble toward the net rather than into a wall,
without overwhelming the existing stack.

**Implementation spec** — new reward class in `src/rlbot/rewards/custom_rl.py`:

```python
class DribbleToGoalReward(RewardFunction):
    """Reward keeping the ball near/on the car (dribble pose) AND close to the
    enemy goal. Scales possession reward by enemy-goal proximity so the bot
    learns to dribble TOWARD the net, not into the side walls.

    Keep the weight small (~0.1-0.2) — this is a directional nudge, not a
    dominant signal.
    """
    # ball-on-car detection thresholds (uu)
    MAX_HORIZONTAL_DIST = 170.0   # ball within this horizontal dist of car
    MIN_BALL_HEIGHT = 110.0       # ball above the car roof (dribble, not just near)
    MAX_BALL_HEIGHT = 400.0

    def reset(self, initial_state): ...
    def get_reward(self, player, state, prev):
        car = player.car_data.position
        ball = state.ball.position
        horiz = ((ball[0]-car[0])**2 + (ball[1]-car[1])**2) ** 0.5
        height = ball[2] - car[2]
        # Is the bot dribbling (ball balanced above/near the car)?
        dribbling = (horiz < self.MAX_HORIZONTAL_DIST
                     and self.MIN_BALL_HEIGHT < height < self.MAX_BALL_HEIGHT)
        if not dribbling:
            return 0.0
        # Enemy goal Y: blue attacks +BACK_WALL_Y, orange attacks -BACK_WALL_Y
        enemy_goal_y = BACK_WALL_Y if player.team_num == 0 else -BACK_WALL_Y
        # Closeness to enemy goal along Y, normalized 0..1 (1 = at the goal line)
        # ball_y ranges -BACK_WALL_Y..+BACK_WALL_Y. Distance to enemy goal:
        dist_to_goal = abs(enemy_goal_y - ball[1])
        proximity = 1.0 - min(1.0, dist_to_goal / (2 * BACK_WALL_Y))
        return proximity   # in [0,1], higher the closer the dribble is to enemy net
    def get_final_reward(self, player, state, prev): return 0.0
```

Add `BACK_WALL_Y` to the imports if not present (it is, used by other rewards). Then in
`simple_bot.py` build_env, add to the CombinedReward with **weight ~0.15**:
```python
DribbleToGoalReward(),   # weight 0.15 — small directional nudge
```

**Caveat:** this is a reward change → value-function recalibration on resume (~5-10M
timesteps wobble). Resume the 416M checkpoint (same EXPERIMENT_NAME) so the strong
policy is preserved; it's a fine-tune. Validate with the progression eval after ~30M.

---

## Task 2 — Kickoff logic + FAST KICKOFF (MEDIUM-HIGH — kickoffs are a proven strength)

**IMPORTANT REMINDER FOR DIEGO: implement the FAST KICKOFF (speedflip) logic.** Per the
635M field observations, kickoffs are already AMAZING and a major source of goals — this
is a strength to push, not fix. The specific want is a faster kickoff approach.

Two ways to get a faster kickoff, pick based on effort budget:

- **(A) Learned via reward (easier, lower ceiling):** the `KickoffReward` sketch below +
  more kickoff scenario weight already pushes the bot to commit hard off kickoff. The bot
  learned good kickoffs this way. Bumping kickoff reward/scenario weight pushes speed further.
- **(B) Scripted speedflip (harder, higher ceiling):** the "fast kickoff" pros use is a
  precise input sequence (diagonal flip → cancel → boost) that reaches the ball in the
  theoretical minimum time. This is HARDCODED, not learned — you'd intercept the action at
  kickoff and replay a fixed input sequence for the first ~0.7s, then hand control back to
  the policy. This is the genuinely "fast kickoff" Diego means. It's complex (frame-perfect
  input timing at tick_skip=8) and is a separate sub-project. Flag it, scope it, get Diego's
  go-ahead before building — it may not be worth it given the learned kickoffs are already
  winning games.

Diego also wants kickoff-specific reward behavior. Two complementary pieces:

**2a. KickoffReward** — reward winning the kickoff (getting first touch / pushing the ball
toward enemy half right after a kickoff). New class in `custom_rl.py`. Sketch:

```python
class KickoffReward(RewardFunction):
    """Active only in the first ~3 seconds after a kickoff (ball near center,
    low time elapsed). Rewards approaching the ball fast + first touch. Teaches
    the bot to commit hard off kickoff instead of hesitating."""
    KICKOFF_BALL_DIST = 300.0   # ball still near center → likely a kickoff
    def get_reward(self, player, state, prev):
        ball = state.ball.position
        ball_near_center = (ball[0]**2 + ball[1]**2) ** 0.5 < self.KICKOFF_BALL_DIST
        if not ball_near_center:
            return 0.0
        # reward speed toward ball during kickoff
        # (reuse VelocityPlayerToBall logic, or +1 on touch during kickoff)
        return 1.0 if player.ball_touched else 0.0
```
Weight ~0.3. Note: detecting "kickoff" purely from ball-near-center is approximate but
works in practice — at kickoff the ball is exactly at (0,0,93).

**2b. Already have kickoff state setters** — `RandomKickoffSetter` (in
`src/rlbot/state_setters/kickoff_scenarios.py`) is already mixed at 30% in the state
setter. If Diego wants MORE kickoff practice, bump that weight to 0.4-0.5, OR use
`FixedKickoffSetter(idx)` to drill a specific kickoff position the bot is weak at.

The 5 kickoff positions + coordinates are in `rl_constants.py` (`KICKOFF_POSITIONS_BLUE`/
`_ORANGE`). All sourced from the RLBot wiki.

---

## Task 3 — Billions-scale training (STRATEGIC, the big one)

**The reality:** the RTX 4070 Laptop does ~10-12k steps/sec on 512×3. To reach 1B from
416M is ~13-16 hours; to reach several billion is days-to-weeks of 24/7 laptop training.
If the opponent is a multi-billion-step bot, the laptop alone won't keep pace within the
class timeline.

**Recommended path: cloud GPU (NOT the GigaLearn C++ rewrite).**

Diego found a leaked C++ framework (GigaLearn, ~10x faster). We decided AGAINST it
because: (a) it requires rewriting all rewards in C++ (Diego is Python-only, contradicts
his professor pitch), (b) leaked/cheater-associated provenance is an academic-integrity
risk, (c) the time cost. **Do not pursue GigaLearn unless Diego explicitly overrides this.**

Instead, the clean path to billions:
1. **Rent a cloud GPU** (RunPod / Lambda / Vast.ai), Linux + A100 or H100.
2. The EXISTING Python stack runs ~3-5x faster on an A100 vs the 4070 Laptop — **no code
   rewrite**, just a faster machine. H100 is ~5-7x.
3. On an A100 running 24/7: ~30-50k steps/sec → ~3-4B steps/day. Reaching billions becomes
   a multi-day cloud run, ~$1.50-3/hr.

**Implementation when Diego is ready:**
- Write a `docs/cloud_gpu_setup.md` runbook: spin up RunPod with the rlbot310 env recreated
  (conda create + pip install -r requirements.txt, dump or upload collision_meshes/),
  rsync/scp the latest checkpoint up, run `simple_bot.py`, sync wandb, pull checkpoints down.
- collision_meshes/ can be uploaded once (it's ~10MB, gitignored). On Linux the rocketsim
  wheel installs cleanly (better than Windows in some respects).
- Keep wandb logging so Diego monitors from his laptop while the cloud GPU grinds.

**Do NOT start this without Diego confirming the cloud spend.** It's a money decision.

---

## Task 4 — Boost collection + speed maintenance (HIGH priority — from 635M observations)

**Problem:** the bot ignores boost pads (especially the big 100 pads) and drives at low
pace. Diego's gamesense: pros maintain high speed so they can rotate back to defend the
instant the ball is cleared. Speed = the ability to be everywhere in time. The bot needs
to value (a) grabbing boost and (b) keeping its velocity high.

Two reward changes in `simple_bot.py` build_env + `custom_rl.py`:

**4a. Bump boost-seeking.** `BigBoostProximityReward` is already ball-distance aware. Two
moves:
- Bump its weight 0.5 → ~0.8 in the CombinedReward.
- Consider raising its `LOW_BOOST_THRESHOLD` from 0.30 → ~0.40 so the bot tops up *before*
  it's nearly empty (pros refill proactively, not reactively).
- Also `EventReward` already has `boost_pickup` (currently 0.3 inside nexto_style) — bump it
  to ~0.5-0.8 so each pickup is a clear positive event. (Edit `nexto_style.py`'s EventReward.)

**4b. Add a continuous speed-maintenance reward.** rlgym_sim ships `VelocityReward` which
rewards raw car speed proportionally (not the binary supersonic threshold `SupersonicReward`
uses). Add it so the bot is rewarded for *keeping* pace, not just briefly hitting supersonic:

```python
from rlgym_sim.utils.reward_functions.common_rewards import VelocityReward
# in the CombinedReward:
VelocityReward(negative=False),   # weight ~0.1 — continuous "keep moving fast" nudge
```

The car's speed is `np.linalg.norm(player.car_data.linear_velocity)`. If you want a custom
version (e.g. only reward speed above the no-boost cap of 1410, or only when not near the
ball), write a `MaintainSpeedReward` in custom_rl.py referencing `CAR_MAX_SPEED_NO_BOOST`
and `CAR_SUPERSONIC_THRESHOLD` from `rl_constants.py`. But start with the built-in
`VelocityReward` at low weight — simplest, and tune from there.

Also consider bumping `SupersonicReward` weight 0.05 → 0.1.

**Watch for over-tuning:** too much speed reward → the bot zooms around pointlessly
ignoring the ball. Keep speed/boost weights modest relative to the ball/goal rewards.
Validate with the progression eval and by watching in rlviser (is it rotating faster and
grabbing pads, without abandoning plays?).

---

## Task 5 — Shot-selection / patience (OPEN PROBLEM — from 635M observations)

**Problem:** the bot shoots too eagerly — it shoots when it could wait, set up, and take a
better shot. Diego doesn't know how to tune this, and honestly it's a genuinely hard,
subtle behavioral problem. Candidate approaches, none guaranteed — experiment:

1. **Reward shot QUALITY, not just shots.** Right now `EventReward(shot=...)` rewards any
   shot on target equally. Consider a custom reward that scales shot reward by shot quality
   — e.g. ball speed toward goal at the moment of the shot, or proximity to goal, or angle.
   A weak pot-shot from midfield gets less than a close, fast, well-angled shot. This
   discourages spamming low-quality shots.
2. **Reward possession/setup.** A small reward for keeping the ball on the bot's side of a
   contest (control) before shooting could encourage patience. Risk: rewards passivity.
3. **Reduce the `shot` weight.** Crude, but if shots are over-incentivized, lowering
   `EventReward(shot=)` makes the bot shoot only when it really pays off (goals still
   rewarded heavily). Try halving it first as a cheap experiment.
4. **Accept it.** Honestly, "shoots a bit too much" is a minor flaw for a bot that wins 65%
   vs a 1.35B opponent. May not be worth the tuning risk before the final. Low priority
   relative to Tasks 1 and 4.

No clean solution — flag to Diego that this is exploratory. Start with option 3 (cheap)
or 1 (most principled) and measure via win rate + rlviser observation.

---

## Task ordering recommendation

1. **Task 1 (dribbling)** — quick, high-impact, fixes dribbling-into-walls.
2. **Task 4 (boost + speed)** — high-impact, directly from Diego's strongest gamesense note.
3. **Task 2 (kickoff reward bump; fast-kickoff is a separate scoped sub-project)** — kickoffs
   already great, light touch to push further. Get go-ahead before the scripted speedflip.
4. Train the combined reward set to ~800M-1B on the laptop, eval vs Marian + the 635M champion.
5. **Task 5 (shot patience)** — only if there's time; exploratory, low priority.
6. **Task 3 (cloud billions)** — only once Diego commits to the spend.

## Always, after any reward/code change

```powershell
# Compile + smoke test before training
C:/Users/Diego/miniconda3/envs/rlbot310/python.exe -m py_compile src/rlbot/rewards/custom_rl.py diego-bots/simple_bot.py
# Then a build_env() smoke test (see how HANDOFF / prior edits did it)
```
And preserve the current champion checkpoint before any fine-tune:
```powershell
$src = "diego-bots/checkpoints/nexto_plus_kickoff_512/<latest_session>/<latest_ts>"
Copy-Item -Path "$src/*" -Destination "diego-bots/checkpoints/MILESTONE_<desc>" -Recurse
```
