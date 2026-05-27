"""Canonical Rocket League game values.

Sourced from the RLBot wiki "Useful game values" page. Diego uses the standard
Octane car (community meta), so car-specific constants below are for Octane.

Unit conventions:
    1 uu (unreal unit) = 1 cm
    2778 uu/s = 100 km/h
    Coordinate system: X = right, Y = forward (positive Y = toward orange's net),
    Z = up. Origin at field center.

These constants exist so reward functions, state setters, and analysis tools
can reference field geometry / physics without magic numbers scattered across
the codebase.
"""
from __future__ import annotations

import math

# ============================================================================
# Field dimensions
# ============================================================================
SIDE_WALL_X = 4096.0          # distance from center to side wall along X
BACK_WALL_Y = 5120.0          # distance from center to back wall along Y
CEILING_Z = 2044.0            # height of arena ceiling
SIDE_WALL_LENGTH = 7936.0
BACK_WALL_LENGTH = 5888.0
CORNER_WALL_LENGTH = 1629.174  # 45° corner-cut panels

# Goal geometry
GOAL_HEIGHT = 642.775
GOAL_CENTER_TO_POST = 892.755   # half goal width = 892.755, full width = 1785.5
GOAL_DEPTH = 880.0

# Goal centers (used as targets for shot/save reward calculations)
BLUE_GOAL_CENTER = (0.0, -BACK_WALL_Y, GOAL_HEIGHT / 2)
ORANGE_GOAL_CENTER = (0.0, BACK_WALL_Y, GOAL_HEIGHT / 2)

# ============================================================================
# Ball
# ============================================================================
BALL_RADIUS = 91.25
BALL_HEIGHT_AT_REST = 93.15       # slightly above floor due to mesh
BALL_MAX_SPEED = 6000.0
BALL_RESTITUTION = 0.6            # loses 40% of incoming velocity on bounce
BALL_MAX_ANG_VEL = 6.0            # rad/s
BALL_MASS = 30.0                  # arbitrary units

# ============================================================================
# Car physics — Octane is the community standard, all bots train on it
# ============================================================================
OCTANE_HEIGHT_AT_REST = 17.01     # very slightly above floor

# Octane hitbox dimensions (from community measurements)
OCTANE_HITBOX_LENGTH = 118.0074
OCTANE_HITBOX_WIDTH = 84.1994
OCTANE_HITBOX_HEIGHT = 36.1591

# Speed thresholds
CAR_MAX_SPEED_BOOSTING = 2300.0   # absolute cap (boost ceiling)
CAR_SUPERSONIC_THRESHOLD = 2200.0 # car enters supersonic state at this speed
CAR_MAX_SPEED_NO_BOOST = 1410.0   # natural top speed without boost

# Acceleration / deceleration
GRAVITY_Z = -650.0                 # uu/s² (default; mutators can change this)
CAR_BOOST_ACCEL_GROUND = 991.666
CAR_BOOST_ACCEL_AIR = 1058.333
CAR_BRAKING_DECEL = 3500.0
CAR_COAST_DECEL = 525.0
CAR_AIR_THROTTLE_ACCEL = 66.667

# Mass / boost
CAR_MASS = 180.0                  # arbitrary units
BOOST_CONSUMPTION_RATE = 33.3     # boost amount per second while boosting
MAX_BOOST = 100.0                 # full tank
KICKOFF_BOOST = 33.0              # starting boost at standard kickoff

# Rotation
CAR_MAX_PITCH_RATE = 5.5          # rad/s, also yaw, also roll

# ============================================================================
# Boost pads
# ============================================================================
SMALL_BOOST_AMOUNT = 12.0
SMALL_BOOST_RESPAWN = 4.0         # seconds
SMALL_BOOST_PAD_RADIUS = 144.0
SMALL_BOOST_PAD_HEIGHT = 165.0

BIG_BOOST_AMOUNT = 100.0
BIG_BOOST_RESPAWN = 10.0
BIG_BOOST_PAD_RADIUS = 208.0
BIG_BOOST_PAD_HEIGHT = 168.0

# Big boost pad positions (the 6 high-value pads — corners + sides)
BIG_BOOST_PAD_POSITIONS = (
    (3584.0, 0.0, 73.0),
    (-3584.0, 0.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (-3072.0, 4096.0, 73.0),
    (-3072.0, -4096.0, 73.0),
)

# ============================================================================
# Kickoff positions (standard 5)
#
# rlgym_sim's DefaultState picks one of these at random. Listing them
# explicitly enables custom kickoff scenarios (e.g. "always train back center").
# Each entry: (x, y, yaw_radians).
# ============================================================================
KICKOFF_POSITIONS_BLUE = (
    (-2048.0, -2560.0,  0.25 * math.pi),  # right corner
    ( 2048.0, -2560.0,  0.75 * math.pi),  # left corner
    ( -256.0, -3840.0,  0.50 * math.pi),  # back right
    (  256.0, -3840.0,  0.50 * math.pi),  # back left
    (    0.0, -4608.0,  0.50 * math.pi),  # far back center
)
KICKOFF_POSITIONS_ORANGE = (
    ( 2048.0,  2560.0, -0.75 * math.pi),
    (-2048.0,  2560.0, -0.25 * math.pi),
    (  256.0,  3840.0, -0.50 * math.pi),
    ( -256.0,  3840.0, -0.50 * math.pi),
    (    0.0,  4608.0, -0.50 * math.pi),
)

# ============================================================================
# Convenience: derived constants useful for normalization in rewards
# ============================================================================
HALF_FIELD_Y = BACK_WALL_Y            # 5120, used for territorial normalization
DIAGONAL_FIELD_LENGTH = math.sqrt(    # max possible flat distance car-to-anything
    (2 * SIDE_WALL_X) ** 2 + (2 * BACK_WALL_Y) ** 2
)
