# Pairing Diego's bot with Marian's bot for a 1v1 in rlviser

This doc maps out where each bot lives in the repo, why they can (or can't) play each other directly, and a copy-paste recipe to watch them play 1v1 inside the rlviser window so you can show it to the team.

---

## 1. Where each bot lives

### Your bot (Diego) — branch `diego`

| Item                 | Path                                                                                        |
|----------------------|---------------------------------------------------------------------------------------------|
| Training script      | `diego-bots/simple_bot.py`                                                                  |
| Visualisation script | `diego-bots/simple_bot_play.py` (self-play, single policy, render via rlviser)              |
| Trained checkpoints  | `diego-bots/checkpoints/<group>/<run>/<timestep>/` — e.g. `baseline/`, `nexto_rewards/`     |
| Obs / action heads   | `DefaultObs()` (89-dim @ team_size=1) + `LookupAction()` (90 discrete actions)              |
| Net shape            | `policy_layer_sizes=(256, 256, 256)` (saved in `BOOK_KEEPING_VARS.json/wandb_config`)       |

### Marian's bot (teammate) — branch `marian/setup-fixes`

Marian's contribution is the structured config-driven pipeline under `src/rlbot/`. Their latest commit (`02b3d69`, *"feat: Mac compatibility fixes, real eval script, and stage 3 reward tuning"*) ships:

| Item                       | Path                                                                                  |
|----------------------------|---------------------------------------------------------------------------------------|
| Training entrypoint        | `src/rlbot/training/train.py` — driven by YAML configs                                |
| Env factory                | `src/rlbot/env/builder.py` (`_EnvBuilder` class, picklable for multi-proc)            |
| **Real bot-vs-bot eval**   | `src/rlbot/evaluation/evaluate.py` — loads two `DiscreteFF` policies, plays N episodes |
| Experiment configs         | `configs/experiments/exp_001_baseline.yaml`, `exp_002_advanced_obs.yaml`, `exp_003_long_run.yaml` |
| Action parser (vendored)   | `src/rlbot/actions/lookup_act.py` *(note: this branch renames `lookup_action.py` → `lookup_act.py`)* |
| Where its checkpoints land | `checkpoints/<experiment_name>/<timestep>/`                                           |

Run training:

```powershell
conda activate rlbot310
python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml
```

Run headless head-to-head eval (Marian's CLI, no rlviser yet):

```powershell
python -m rlbot.evaluation.evaluate `
    --blue   checkpoints/exp_001_baseline/<timestep> `
    --orange latest:exp_001_baseline `
    --episodes 20
```

The eval prints `BLUE wins / ORANGE wins / draws` and a final `blue_win_rate`.

---

## 2. Compatibility — what has to match before they can play

The two policies are independent neural nets, so they only play each other correctly if **the observation vector and action space are identical** on both sides. Otherwise the policy receives noise or its output indexes the wrong control.

| Spec               | Diego's `simple_bot.py` | Marian's `exp_001_baseline` | Marian's `exp_002` / `exp_003` |
|--------------------|-------------------------|-----------------------------|--------------------------------|
| `team_size`        | 1                       | 1                           | 1                              |
| `tick_skip`        | 8                       | 8                           | 8                              |
| Obs builder        | `DefaultObs` (89-dim)   | `DefaultObs` (89-dim) ✅    | `AdvancedObs` ❌ (different shape) |
| Action parser      | `LookupAction` (90)     | `LookupAction` (90) ✅      | `LookupAction` (90)            |
| Net shape          | (256, 256, 256)         | from `arch: small`          | from `arch: medium`            |

**Direct 1v1 will only work between Diego's bot and an `exp_001_baseline` checkpoint** (matching `DefaultObs`). For `exp_002` / `exp_003` you'd need either (a) retrain Diego's bot with `AdvancedObs` or (b) retrain Marian's experiment on `DefaultObs`.

The net shape mismatch (256³ vs `arch: small`) is fine — Marian's `_load_policy` reads `policy_layer_sizes` from each checkpoint's `BOOK_KEEPING_VARS.json` and builds the network to match.

---

## 3. Bringing both code paths into one working tree

Right now Marian's code lives only on `marian/setup-fixes`; you're on `diego`. To run them together you need both in the same checkout. The cleanest way:

```powershell
git checkout diego
git merge marian/setup-fixes
# Resolve the lookup_action.py vs lookup_act.py rename if it conflicts —
# keep both, or pick the marian/setup-fixes version and update simple_bot.py's
# import from `rlbot.actions.lookup_action` → `rlbot.actions.lookup_act`.
```

If you'd rather not merge yet, you can also `git worktree add ../rlgym-marian marian/setup-fixes` and operate on two folders, but that splits checkpoints across trees, which is annoying for eval.

---

## 4. Watching them play 1v1 in rlviser

Marian's `evaluate.py` runs the env headlessly. rlgym_sim's env exposes `env.render()` (it forwards to `rlviser_py.render`), so the minimal change is: call it once per step and throttle to roughly real time. Save the following as **`diego-bots/eval_render.py`** on the merged branch — it's a thin wrapper over Marian's eval that adds the rlviser hooks.

```python
"""Watch two trained policies play 1v1 inside rlviser.

Prereqs:
  - rlviser.exe is running in the background (double-click rlviser.exe in repo root)
  - rlviser_py is installed in rlbot310 (it is, per the existing simple_bot_play.py)
  - You've merged marian/setup-fixes into diego (or are on a branch that has both)

Run:
  conda activate rlbot310
  python diego-bots/eval_render.py \
      --blue   diego-bots/checkpoints/nexto_rewards/nexto_rewards-1779731987743259900/46802152 \
      --orange checkpoints/exp_001_baseline/<some_timestep>
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

# Re-use Marian's helpers so we don't drift from his loader logic.
from rlbot.evaluation.evaluate import (
    _resolve_checkpoint_path,
    _load_policy,
    _build_eval_env,
    _action_to_int,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blue",   required=True, help="Blue policy checkpoint folder")
    p.add_argument("--orange", required=True, help="Orange policy checkpoint folder (or 'latest:exp_name')")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--deterministic", action="store_true",
                   help="Greedy argmax instead of sampling — looks less twitchy")
    p.add_argument("--step-delay", type=float, default=0.006,
                   help="Seconds to sleep per env.step — 0.006 ≈ real time at tick_skip=8")
    args = p.parse_args()

    blue_ckpt   = _resolve_checkpoint_path(args.blue)
    orange_ckpt = _resolve_checkpoint_path(args.orange)
    print(f"BLUE   {blue_ckpt}")
    print(f"ORANGE {orange_ckpt}")

    blue_policy   = _load_policy(blue_ckpt,   device="cpu")
    orange_policy = _load_policy(orange_ckpt, device="cpu")

    env = _build_eval_env()
    print("Make sure rlviser.exe is open. Press Ctrl+C to stop.")
    try:
        for ep in range(1, args.episodes + 1):
            obs_list = env.reset()
            blue_obs, orange_obs = obs_list[0], obs_list[1]
            done = False
            while not done:
                with torch.no_grad():
                    b_act, _ = blue_policy.get_action(blue_obs,   deterministic=args.deterministic)
                    o_act, _ = orange_policy.get_action(orange_obs, deterministic=args.deterministic)
                obs_list, _r, done, info = env.step([_action_to_int(b_act), _action_to_int(o_act)])
                blue_obs, orange_obs = obs_list[0], obs_list[1]

                env.render()              # push current state to rlviser
                time.sleep(args.step_delay)  # throttle to ~real time

            result = int(info.get("result", 0))
            outcome = "BLUE" if result > 0 else "ORANGE" if result < 0 else "DRAW"
            print(f"Ep {ep}/{args.episodes}  {outcome}  (delta={result:+d})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

### Step-by-step to show your team

1. **Open rlviser.** Double-click `rlviser.exe` in the repo root. An empty arena window appears — leave it open.
2. **Activate the env.** `conda activate rlbot310`
3. **Pick checkpoints.**
   - Blue (yours): pick any timestep folder under `diego-bots/checkpoints/...`. The `nexto_rewards/.../46802152` one is your most-trained policy (~46M timesteps).
   - Orange (Marian's): once they've trained at least one `exp_001_baseline` run on this machine, point at `checkpoints/exp_001_baseline/<timestep>` (or use `latest:exp_001_baseline`). If they don't have one yet, kick off `python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml` and let it save a checkpoint (default `save_every_ts: 500_000` for exp_001).
4. **Run the viewer.**
   ```powershell
   python diego-bots/eval_render.py `
       --blue   diego-bots/checkpoints/nexto_rewards/nexto_rewards-1779731987743259900/46802152 `
       --orange latest:exp_001_baseline `
       --episodes 5 --deterministic
   ```
5. Watch the rlviser window — blue (your bot) vs orange (teammate's bot) playing 1v1 kickoffs from `DefaultState`. Per-episode outcome prints in the terminal.

### If something looks wrong

- **Window stays empty / cars don't move.** `rlviser.exe` isn't running, or `rlviser_py` is in a different env. Verify with `python -c "import rlviser_py; print(rlviser_py.__version__)"` inside `rlbot310`.
- **`FileNotFoundError: BOOK_KEEPING_VARS.json not found`.** You pointed at a *run* folder (`simple_bot-<ts>/`) instead of a *timestep* folder inside it (`simple_bot-<ts>/500028/`).
- **`RuntimeError: size mismatch ... PPO_POLICY.pt`.** Obs spec mismatch. The two checkpoints were trained with different obs builders — see the compatibility table above. Pick an exp_001 checkpoint, not exp_002/exp_003.
- **Visualisation runs at 200 fps blur.** Bump `--step-delay` from 0.006 to 0.01 — that throttles the env loop, the policy still acts at 15 Hz.

---

## 5. TL;DR

- **Diego's bot:** `diego-bots/simple_bot.py` + checkpoints in `diego-bots/checkpoints/`.
- **Marian's bot:** `src/rlbot/` pipeline on branch `marian/setup-fixes`, trained via `python -m rlbot.training.train --config configs/experiments/exp_001_baseline.yaml`, checkpoints in `checkpoints/exp_001_baseline/`.
- **They can play 1v1** only when both use `DefaultObs` + `LookupAction` (so: Diego vs `exp_001_baseline`, not vs `exp_002` / `exp_003` until obs is unified).
- **To visualise in rlviser**, merge `marian/setup-fixes` into your branch, save `diego-bots/eval_render.py` (above), open `rlviser.exe`, and run the script with `--blue` / `--orange` pointing at the two checkpoint timestep folders.
