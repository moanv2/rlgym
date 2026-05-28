# NEXT TASKS

Concrete work items for the next session, in priority order. Read `docs/HANDOFF.md`
first for project state and `docs/LESSONS_LEARNED.md` for gotchas. Each task below has
enough spec to implement without re-deriving anything.

Diego's stated goal has escalated: **the final is now "billions vs billions"** — both his
bot and the opponent will be trained to billions of timesteps. So tasks 1-2 improve bot
quality; task 3 is about reaching that compute scale.

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

## Task 2 — Kickoff logic (MEDIUM priority)

Diego wants kickoff-specific behavior. Two complementary pieces:

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

## Task ordering recommendation

1. **Task 1 (dribbling)** first — quick, high-impact, fixes a behavior Diego saw failing.
2. **Task 2 (kickoff)** — quick, complements task 1.
3. Train the combined task-1+2 reward set to ~600M on the laptop, eval vs Marian + vs the
   416M champion to confirm improvement.
4. **Task 3 (cloud)** only once Diego commits to the billions-scale push and the spend.

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
