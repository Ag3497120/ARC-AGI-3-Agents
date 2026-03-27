"""
cross_engine — Verantyx Cross Engine for ARC-AGI-3

Modules:
  simulator  — CrossWorld: game world model + BFS simulation
  primitives — Rule primitives as Cross operations
  shape_eye  — Multi-scale shape recognition
  rule_mixer — Rule combination explorer (zero API cost)

Usage:
    from agents.cross_engine import CrossWorld, ShapeEye, RuleMixer
    from agents.cross_engine import all_primitives
"""

from .simulator import CrossWorld, CrossCell
from .shape_eye import ShapeEye
from .rule_mixer import RuleMixer
from .cross_sensor import CrossSensor, CrossSnapshot, CrossObject, CrossDescriptor, FrameDiff
from .primitives import (
    RulePrimitive,
    MazeRule,
    KeyMatchRule,
    PatternMatchRule,
    ProximityRule,
    ReversiRule,
    GravityRule,
    FillRule,
    PatternStampRule,
    ConnectRule,
    all_primitives,
)

__all__ = [
    # Core world model
    'CrossWorld',
    'CrossCell',
    # Recognition
    'ShapeEye',
    # Rule exploration
    'RuleMixer',
    # Primitives
    'RulePrimitive',
    'MazeRule',
    'KeyMatchRule',
    'PatternMatchRule',
    'ProximityRule',
    'ReversiRule',
    'GravityRule',
    'FillRule',
    'PatternStampRule',
    'ConnectRule',
    'all_primitives',
    # Cross Sensor
    'CrossSensor',
    'CrossSnapshot',
    'CrossObject',
    'CrossDescriptor',
    'FrameDiff',
]
