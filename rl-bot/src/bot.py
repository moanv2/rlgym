"""RLBot v5 bot — Always-Towards-Ball baseline.

This is the DEPLOYMENT bot: the agent that plays real Rocket League against
another RLBot (e.g. the classmate's bot at the final). It is deliberately
separate from the RLGym training pipeline in ``diego-bots/`` and ``src/rlbot/``;
this process talks to RLBotServer over a socket and returns controller inputs
every tick @ 120 Hz.

Behaviour (v0 baseline, see docs/rlbot_v5_skeleton_brief.md §6.3):
    - steer toward the ball's (ground) position, leading it with ball prediction
      when far away,
    - full throttle, BOOST when roughly facing the ball and below supersonic,
    - a front flip when close to the ball and moving fast enough for it to pay
      off,
    - one guarded debug render (car -> target line) that no-ops when disabled.

Strategy seam:
    Every tick we ask ``self.policy.decide(packet)`` first. The stub in
    ``policy/__init__.py`` returns ``None`` (defer to baseline), so today the bot
    behaves exactly as the baseline. When a real decision layer / learned
    controller is dropped into ``policy/`` later, NOTHING here needs to change.

v5 API notes (verified against RLBot/python-example @ master):
    - base class: ``rlbot.managers.Bot``; flat types from ``rlbot.flat``,
    - lifecycle: ``initialize(self)`` runs once after field_info/match_config
      are available; ``get_output(self, packet) -> ControllerState`` runs per tick,
    - ball prediction via ``self.ball_prediction`` + ``find_slice_at_time``,
    - rendering via ``self.renderer`` (RenderAnchors: BallAnchor / CarAnchor),
    - the constructor arg MUST match ``agent_id`` in ``src/bot.toml``.
"""

from typing import override

from rlbot.flat import BallAnchor, ControllerState, GamePacket
from rlbot.managers import Bot
from rlbot_flatbuffers import CarAnchor

from policy import Policy
from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.orientation import Orientation
from util.sequence import ControlStep, Sequence
from util.vec import Vec3

# Rocket League physics constants (uu/s). Octane / standard match.
SUPERSONIC_SPEED = 2200.0  # car is "supersonic" at/above this; no point boosting past it
# How aligned the car must be with the target before we spend boost. dot product
# of the car's forward vector and the (ground) direction to the target; 1.0 = dead-on.
BOOST_ALIGNMENT = 0.8
# Proximity (uu) at which we commit to a front flip into the ball.
FLIP_BALL_DISTANCE = 500.0
# Don't flip from a standstill — only when we have enough speed for it to pay off.
FLIP_MIN_SPEED = 1000.0


class MyBot(Bot):
    active_sequence: Sequence | None = None
    boost_pad_tracker: BoostPadTracker = BoostPadTracker()

    # Client-side render guard. The match config's ``enable_rendering`` is the
    # real switch (server-side); this lets us also turn debug drawing off here
    # so render calls truly no-op without touching the tick logic.
    DEBUG_RENDER: bool = True

    @override
    def initialize(self):
        # Field info (boost pad layout, goals) is available now that the match
        # is active. Build the strategy delegate once; construction must be cheap.
        self.boost_pad_tracker.initialize_boosts(self.field_info)
        self.policy = Policy()
        # Wire the learned controller now that field_info / match_config / index
        # are available. If deps or weights are missing, setup leaves the policy
        # not-ready and decide() returns None, so we fall back to the baseline.
        self.policy.setup(self.field_info, self.match_config, self.index)

    @override
    def get_output(self, packet: GamePacket) -> ControllerState:
        """Called by the framework many times per second. Must be fast / non-blocking."""

        # Keep our boost pad info updated with which pads are currently active.
        self.boost_pad_tracker.update_boost_status(packet)

        if len(packet.balls) == 0:
            # No ball this tick (e.g. during a replay) — do nothing.
            return ControllerState()

        # Finish any multi-frame action (e.g. a flip) we already started.
        if self.active_sequence is not None and not self.active_sequence.done:
            return self.active_sequence.tick(packet)

        # --- Strategy seam ------------------------------------------------
        # Ask the learned policy (1B-step PPO checkpoint) for controls. It drives
        # whenever it loaded successfully; it returns None only if its deps /
        # weights are missing or it hit a runtime error, in which case we fall
        # through to the Always-Towards-Ball baseline below.
        override_controls = self.policy.decide(packet)
        if override_controls is not None:
            return override_controls

        # --- Baseline: chase the ball -------------------------------------
        my_car = packet.players[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.balls[0].physics.location)

        target_location = ball_location

        if car_location.dist(ball_location) > 1500:
            # Far away: lead the ball using prediction (it can bounce/roll).
            ball_in_future = find_slice_at_time(
                self.ball_prediction, packet.match_info.seconds_elapsed + 2
            )
            if ball_in_future is not None:
                target_location = Vec3(ball_in_future.physics.location)

        # Guarded debug render: a line from the car to the current target.
        self._debug_render(target_location, car_velocity)

        # Compute controls toward the (ground) target.
        controls = ControllerState()
        controls.steer = steer_toward_target(my_car, target_location)
        controls.throttle = 1.0

        # Boost when roughly facing the target and not already supersonic.
        speed = car_velocity.length()
        if speed < SUPERSONIC_SPEED and self._aligned_with(my_car, target_location):
            controls.boost = True

        # Flip into the ball when close and moving fast enough for it to help.
        ball_distance = car_location.dist(ball_location)
        if ball_distance < FLIP_BALL_DISTANCE and speed > FLIP_MIN_SPEED:
            return self.begin_front_flip(packet)

        return controls

    def _aligned_with(self, my_car, target_location: Vec3) -> bool:
        """True if the car's nose points roughly at the target (ground plane)."""
        to_target = (target_location - Vec3(my_car.physics.location)).flat()
        if to_target.length() == 0:
            return True
        forward = Orientation(my_car.physics.rotation).forward.flat()
        if forward.length() == 0:
            return False
        return forward.normalized().dot(to_target.normalized()) > BOOST_ALIGNMENT

    def _debug_render(self, target_location: Vec3, car_velocity: Vec3) -> None:
        """Draw a single debug line car->target. Guarded + wrapped so it can
        never block or crash the tick when rendering is disabled."""
        if not self.DEBUG_RENDER:
            return
        try:
            self.renderer.begin_rendering()
            self.renderer.draw_line_3d(
                CarAnchor(self.index), target_location, self.renderer.white
            )
            self.renderer.draw_string_3d(
                f"Speed: {car_velocity.length():.0f}",
                CarAnchor(self.index),
                1,
                self.renderer.white,
            )
            self.renderer.draw_line_3d(
                BallAnchor(0), target_location, self.renderer.cyan
            )
            self.renderer.end_rendering()
        except Exception:
            # Rendering is best-effort debug sugar; never let it affect play.
            pass

    def begin_front_flip(self, packet: GamePacket) -> ControllerState:
        # A front flip is a fixed multi-frame input sequence. We commit to it for
        # ~1s; get_output ignores other logic while active_sequence runs.
        self.active_sequence = Sequence(
            [
                ControlStep(duration=0.05, controls=ControllerState(jump=True)),
                ControlStep(duration=0.05, controls=ControllerState(jump=False)),
                ControlStep(
                    duration=0.2, controls=ControllerState(jump=True, pitch=-1)
                ),
                ControlStep(duration=0.8, controls=ControllerState()),
            ]
        )
        return self.active_sequence.tick(packet)


if __name__ == "__main__":
    # The agent id MUST match `agent_id` in src/bot.toml. The base class reads
    # RLBOT_AGENT_ID from the environment when set, and falls back to this.
    MyBot("moanv2/myrlbot").run()
