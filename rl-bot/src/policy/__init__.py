"""Learned-policy seam — drives the car with the trained PPO checkpoint.

This is the real decision layer that ``bot.py`` calls every tick. It loads the
1B-timestep ``nexto_plus_kickoff_512`` policy (DiscreteFF 89->512->512->512->90)
that was trained in ``diego-bots/`` against rlgym_sim's ``DefaultObs`` +
``LookupAction``, and reproduces that exact obs/action pipeline in real Rocket
League:

    packet --(rlgym-compat V1GameState)--> rlgym-v1-style GameState
           --(vendored DefaultObs)-------> 89-dim obs
           --(DiscreteFF forward)--------> action index 0..89
           --(vendored LookupAction)-----> 8-dim controls --> ControllerState

Design notes
------------
* **Graceful fallback.** Heavy deps (torch, numpy, rlgym_compat) are imported
  lazily and wrapped in try/except. If any are missing, or the checkpoint can't
  load, ``decide`` returns ``None`` and ``bot.py`` falls back to its
  Always-Towards-Ball baseline. This keeps ``validate.py`` (which only has
  ``rlbot`` installed) green and the skeleton importable.
* **tick_skip.** The policy was trained acting at 15 Hz (tick_skip=8 of the
  game's 120 Hz). RLBot calls ``get_output`` every physics tick, so we hold the
  last chosen controls and only run a new forward pass every 8 ticks. The
  compat ``V1GameState`` is still ``update``-d every tick for accurate
  ``has_flip`` / boost tracking.
* **previous_action.** DefaultObs feeds the last emitted 8-dim controller vector
  back into the observation, so we track it (zeros on the first decision).
* **Determinism.** Greedy argmax by default (less twitchy, plays a human
  cleaner). Set ``Policy.DETERMINISTIC = False`` for stochastic sampling, which
  matches how the policy explored in training.

Contract: ``decide(packet) -> ControllerState | None`` — fast, non-blocking,
called every tick @120Hz.
"""
from __future__ import annotations

from pathlib import Path

from rlbot.flat import ControllerState, GamePacket

_WEIGHTS_DIR = Path(__file__).parent / "weights"


class Policy:
    # Must match the trained checkpoint (see weights/BOOK_KEEPING_VARS.json).
    # papaya_1024 @ 828M: AdvancedObs (107) + LookupAction (90), 1024x3 net.
    INPUT_SIZE = 107
    N_ACTIONS = 90
    LAYER_SIZES = (1024, 1024, 1024)
    TICK_SKIP = 8
    DETERMINISTIC = True

    def __init__(self, weights_dir: Path = _WEIGHTS_DIR) -> None:
        self.ready: bool = False
        self.reason: str = ""

        self._net = None
        self._torch = None
        self._np = None
        self._lookup = None
        self._build_obs = None

        self._game_state = None
        self._index = 0
        self._prev_action = None
        self._controls = ControllerState()
        self._ticks = self.TICK_SKIP  # force a decision on the first acting tick
        self._prev_time: float | None = None
        self._warned_obs_size = False
        self._error_count = 0

        # --- load the network (needs torch + numpy + the vendored modules) ---
        try:
            import numpy as np
            import torch

            from .discrete_ff import DiscreteFF
            from .advanced_obs import build_obs
            from .lookup_action import LOOKUP_TABLE

            ckpt = weights_dir / "PPO_POLICY.pt"
            if not ckpt.exists():
                self.reason = f"checkpoint missing: {ckpt}"
                return

            # Read the trained step count from the bundled book-keeping so the
            # ready-banner always reflects whichever checkpoint is deployed.
            self._trained_steps: int | None = None
            try:
                import json
                bk = weights_dir / "BOOK_KEEPING_VARS.json"
                if bk.exists():
                    self._trained_steps = json.loads(bk.read_text()).get("cumulative_timesteps")
            except Exception:
                pass

            net = DiscreteFF(self.INPUT_SIZE, self.N_ACTIONS, self.LAYER_SIZES, "cpu")
            # weights_only=True: the file is a pure tensor state_dict, so this is
            # both safe and forward-compatible with torch's changing default.
            net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            net.eval()

            self._torch = torch
            self._np = np
            self._net = net
            self._lookup = LOOKUP_TABLE
            self._build_obs = build_obs
            self._prev_action = np.zeros(8, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - any failure -> baseline fallback
            self.reason = f"net load failed: {type(exc).__name__}: {exc}"
            self._net = None

    def setup(self, field_info, match_config, index: int) -> None:
        """Build the rlgym-compat game-state adapter. Call once from
        ``bot.initialize`` (field_info / match_config are available then)."""
        if self._net is None:
            print(f"[policy] learned controller disabled, using baseline ({self.reason})")
            return
        try:
            from rlgym_compat import V1GameState

            self._game_state = V1GameState(
                field_info,
                match_configuration=match_config,
                tick_skip=self.TICK_SKIP,
                standard_map=True,
            )
            self._index = index
            self.ready = True
            steps = (
                f"{self._trained_steps/1e9:.2f}B" if self._trained_steps else "?"
            )
            print(
                f"[policy] learned controller ready: papaya_1024 @ {steps} steps "
                f"(AdvancedObs 107, 1024x3) "
                f"({'greedy' if self.DETERMINISTIC else 'stochastic'})"
            )
        except Exception as exc:  # noqa: BLE001
            self.reason = f"compat setup failed: {type(exc).__name__}: {exc}"
            self.ready = False
            print(f"[policy] {self.reason} — using baseline")

    def decide(self, packet: GamePacket) -> ControllerState | None:
        """Return controls from the trained policy, or None to defer to the
        baseline. Never raises into the tick loop — on any error it disables
        itself and hands control back to the baseline."""
        if not self.ready:
            return None
        if len(packet.balls) == 0 or len(packet.players) <= self._index:
            return self._controls

        try:
            # Update compat state every frame (accurate has_flip / boost / demo).
            self._game_state.update(packet)

            cur = packet.match_info.seconds_elapsed
            if self._prev_time is None:
                self._prev_time = cur
            self._ticks += max(1, int(round((cur - self._prev_time) * 120)))
            self._prev_time = cur

            if self._ticks >= self.TICK_SKIP:
                self._ticks = 0
                player = self._game_state.players[self._index]
                obs = self._build_obs(player, self._game_state, self._prev_action)
                # rlgym-compat may not have every car populated on the first
                # ticks after spawn (e.g. a human opponent that just appeared),
                # producing a short obs. Hold and retry instead of feeding the
                # net a wrong-sized vector — a transient frame must NOT bench
                # the policy for the whole match.
                if obs.shape[0] != self.INPUT_SIZE:
                    if not self._warned_obs_size:
                        print(
                            f"[policy] obs not ready yet ({obs.shape[0]}/{self.INPUT_SIZE} "
                            f"dims) — holding, will retry once all cars are present"
                        )
                        self._warned_obs_size = True
                    self._ticks = self.TICK_SKIP  # re-attempt on the next tick
                    return self._controls
                with self._torch.no_grad():
                    idx = self._net.get_action(obs, deterministic=self.DETERMINISTIC)
                action_vec = self._lookup[idx]
                self._prev_action = action_vec
                self._controls = self._vec_to_controls(action_vec)
                self._error_count = 0  # a clean decision clears the error streak
            return self._controls
        except Exception as exc:  # noqa: BLE001 - never crash a live match
            # Skip this tick and keep the last controls. Only give up on the
            # policy after a sustained streak of failures (~2s @120Hz), so one
            # bad frame doesn't permanently drop us to the baseline.
            self._error_count += 1
            if self._error_count <= 3:
                print(f"[policy] runtime error (tick skipped): {type(exc).__name__}: {exc}")
            if self._error_count >= 240:
                self.ready = False
                self.reason = f"runtime error: {type(exc).__name__}: {exc}"
                print(f"[policy] too many runtime errors — falling back to baseline ({self.reason})")
                return None
            return self._controls

    @staticmethod
    def _vec_to_controls(v) -> ControllerState:
        """Map an 8-dim LookupAction row
        [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        to a v5 ControllerState."""
        c = ControllerState()
        c.throttle = float(v[0])
        c.steer = float(v[1])
        c.pitch = float(v[2])
        c.yaw = float(v[3])
        c.roll = float(v[4])
        c.jump = bool(v[5])
        c.boost = bool(v[6])
        c.handbrake = bool(v[7])
        return c
