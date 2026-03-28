"""
rule_learner.py — The "soul" that v26 is missing.

Closes the Observe → Hypothesize → Verify → Learn loop:
  - ReactionAnalyzer: turns grid diffs into structured ReactionEvents
  - RuleLearner: builds and confirms hypotheses from ReactionEvents
  - DynamicPlanner: routes through learned trigger points before goals
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReactionEvent:
    """Structured description of a single environmental reaction."""

    frame: int
    trigger_action: int               # action index the player performed
    trigger_pos: Tuple[int, int]       # player (row, col) when reaction occurred

    # What changed
    changed_cells: List[Tuple[int, int]]
    color_transitions: Dict[Tuple[int, int], int]  # (old_color, new_color) → count

    # Spatial analysis
    change_center: Tuple[int, int]             # centroid of changed area
    change_bbox: Tuple[int, int, int, int]     # (r_min, c_min, r_max, c_max)
    distance_from_player: float

    # Classification
    change_type: str  # 'wall_opened' | 'wall_closed' | 'color_cycle' |
                      # 'pattern_transform' | 'structure_revealed' | 'unknown'
    affected_structure: Optional[str] = None   # structural role tag, if known


@dataclass
class LearnedRule:
    """A hypothesis (possibly confirmed) about what causes a reaction."""

    rule_id: int

    # Trigger condition
    trigger_type: str                              # 'proximity' | 'passage' | 'click' | 'time'
    trigger_color: Optional[int] = None            # color player stood on when rule fired
    trigger_region: Optional[Tuple[int, int, int, int]] = None  # bbox of trigger area

    # Effect
    effect_type: str = "unknown"
    effect_region: Tuple[int, int, int, int] = (0, 0, 0, 0)
    effect_transitions: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Confidence
    observations: int = 1
    confirmed: bool = False

    # Strategic value
    enables_path: bool = False        # opening this rule makes new cells walkable
    required_for_goal: bool = False   # must trigger before reaching a goal


# ---------------------------------------------------------------------------
# Class 1: ReactionAnalyzer
# ---------------------------------------------------------------------------

class ReactionAnalyzer:
    """
    Compares two consecutive grids and returns a structured ReactionEvent
    describing what changed, where, and what kind of change it was.
    """

    # Player typically occupies 1–4 cells; anything more is an environmental reaction
    PLAYER_FOOTPRINT_RADIUS = 2

    def __init__(self, grid_rows: int = 64, grid_cols: int = 64) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.timer_row_start = 60  # rows ≥ 60 are timer/UI — ignore them

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        prev_grid,
        curr_grid,
        player_pos: Tuple[int, int],
        action_idx: int,
        frame: int = 0,
        smap=None,
        corridor_colors: Optional[Set[int]] = None,
    ) -> Optional[ReactionEvent]:
        """
        Analyze what changed between prev_grid and curr_grid.

        Returns a ReactionEvent if a meaningful environmental reaction is found,
        or None if nothing significant changed (e.g. only player footprint moved).
        """
        corridor_colors = corridor_colors or set()

        # 1. Find all changed cells (exclude timer/UI rows)
        rows = min(self.timer_row_start, len(prev_grid))
        import numpy as _np
        if isinstance(prev_grid, _np.ndarray):
            rows = min(self.timer_row_start, prev_grid.shape[0])
            cols = prev_grid.shape[1] if prev_grid.ndim >= 2 else 0
        else:
            cols = len(prev_grid[0]) if prev_grid else 0

        all_changed: List[Tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                if int(prev_grid[r][c]) != int(curr_grid[r][c]):
                    all_changed.append((r, c))

        if not all_changed:
            return None

        # 2. Remove player-footprint cells (cells immediately around the player)
        pr, pc = player_pos
        env_changed = [
            (r, c) for (r, c) in all_changed
            if abs(r - pr) > self.PLAYER_FOOTPRINT_RADIUS
            or abs(c - pc) > self.PLAYER_FOOTPRINT_RADIUS
        ]

        # If nothing changed beyond the player's footprint, it's just movement
        if not env_changed:
            return None

        # 3. Compute color transitions
        color_transitions: Dict[Tuple[int, int], int] = {}
        for r, c in env_changed:
            key = (int(prev_grid[r][c]), int(curr_grid[r][c]))
            color_transitions[key] = color_transitions.get(key, 0) + 1

        # 4. Spatial metrics
        change_center = self._centroid(env_changed)
        change_bbox = self._bbox(env_changed)
        distance = math.hypot(change_center[0] - pr, change_center[1] - pc)

        # 5. Classify
        change_type = self._classify(
            color_transitions, env_changed, player_pos, corridor_colors, smap, distance
        )

        return ReactionEvent(
            frame=frame,
            trigger_action=action_idx,
            trigger_pos=player_pos,
            changed_cells=env_changed,
            color_transitions=color_transitions,
            change_center=change_center,
            change_bbox=change_bbox,
            distance_from_player=distance,
            change_type=change_type,
            affected_structure=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(
        self,
        color_transitions: Dict[Tuple[int, int], int],
        changed: List[Tuple[int, int]],
        player_pos: Tuple[int, int],
        corridor_colors: Set[int],
        smap,
        distance: float,
    ) -> str:
        """
        Classify what happened:

        wall_opened       — impassable color → passable/corridor color
        wall_closed       — corridor color → impassable color
        color_cycle       — multiple colors rotating (no net gain/loss)
        pattern_transform — large area changed shape/color
        structure_revealed — new structure appeared (was background before)
        unknown           — can't determine
        """
        opened = 0
        closed = 0

        old_colors: Set[int] = set()
        new_colors: Set[int] = set()

        for (old, new), count in color_transitions.items():
            old_colors.add(old)
            new_colors.add(new)

            if corridor_colors:
                if old not in corridor_colors and new in corridor_colors:
                    opened += count
                elif old in corridor_colors and new not in corridor_colors:
                    closed += count

        # Prefer explicit corridor-based classification
        if opened > 0 and opened >= closed:
            return "wall_opened"
        if closed > 0 and closed > opened:
            return "wall_closed"

        # Color cycle: old and new colors are the same set (just shuffled)
        if old_colors and old_colors == new_colors:
            return "color_cycle"

        # Pattern transform: large diffuse change far from player
        if len(changed) > 20 and distance > 10:
            return "pattern_transform"

        # Small change far from player with new colors appearing → structure revealed
        if distance > 10 and new_colors - old_colors:
            return "structure_revealed"

        return "unknown"

    @staticmethod
    def _centroid(cells: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not cells:
            return (0, 0)
        r_sum = sum(r for r, _ in cells)
        c_sum = sum(c for _, c in cells)
        n = len(cells)
        return (r_sum // n, c_sum // n)

    @staticmethod
    def _bbox(cells: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        if not cells:
            return (0, 0, 0, 0)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return (min(rs), min(cs), max(rs), max(cs))


# ---------------------------------------------------------------------------
# Class 2: RuleLearner
# ---------------------------------------------------------------------------

class RuleLearner:
    """
    Maintains a set of hypotheses (LearnedRule objects) derived from
    observed ReactionEvents. Confirms rules when the same pattern repeats.
    """

    # Minimum observations before a rule is considered confirmed
    CONFIRM_THRESHOLD = 2

    def __init__(self) -> None:
        self.events: List[ReactionEvent] = []
        self.rules: List[LearnedRule] = []
        self.next_rule_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        event: ReactionEvent,
        player_grid_color: Optional[int] = None,
    ) -> Optional[LearnedRule]:
        """
        Process a new reaction event.

        player_grid_color — the color value at the player's position in the
                            previous grid (likely trigger color).

        Returns the matched or newly created LearnedRule, or None if the
        event couldn't be hypothesized.
        """
        self.events.append(event)

        # Try to match with an existing rule
        for rule in self.rules:
            if self._matches(rule, event):
                rule.observations += 1
                if rule.observations >= self.CONFIRM_THRESHOLD:
                    rule.confirmed = True
                # Update trigger color if we have better evidence
                if player_grid_color is not None and rule.trigger_color is None:
                    rule.trigger_color = player_grid_color
                return rule

        # No match — form a new hypothesis
        new_rule = self._hypothesize(event, player_grid_color)
        if new_rule:
            self.rules.append(new_rule)
            return new_rule

        return None

    def get_required_triggers(self) -> List[LearnedRule]:
        """Return rules that enable paths and have a known trigger region."""
        return [r for r in self.rules if r.enables_path and r.trigger_region is not None]

    def get_waypoints_from_rules(self) -> List[Tuple[int, int]]:
        """Convert rule trigger regions into centre-point waypoints."""
        waypoints: List[Tuple[int, int]] = []
        for rule in self.get_required_triggers():
            if rule.trigger_region:
                r_min, c_min, r_max, c_max = rule.trigger_region
                center = ((r_min + r_max) // 2, (c_min + c_max) // 2)
                waypoints.append(center)
        return waypoints

    def get_summary(self) -> str:
        """Human-readable summary for logging."""
        lines = [f"RuleLearner: {len(self.rules)} rules, {len(self.events)} events"]
        for rule in self.rules:
            status = "CONFIRMED" if rule.confirmed else f"hyp({rule.observations})"
            lines.append(
                f"  Rule#{rule.rule_id} [{status}] "
                f"{rule.trigger_type}→{rule.effect_type} "
                f"enables_path={rule.enables_path}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hypothesize(
        self,
        event: ReactionEvent,
        player_grid_color: Optional[int],
    ) -> Optional[LearnedRule]:
        """Form an initial hypothesis from a single ReactionEvent."""

        # Determine trigger type based on proximity
        trigger_type = (
            "proximity" if event.distance_from_player < 15 else "passage"
        )

        rule = LearnedRule(
            rule_id=self.next_rule_id,
            trigger_type=trigger_type,
            trigger_color=player_grid_color,
            trigger_region=None,  # caller can set this with more context
            effect_type=event.change_type,
            effect_region=event.change_bbox,
            effect_transitions=dict(event.color_transitions),
        )
        self.next_rule_id += 1

        # Mark as path-enabling if it opened a wall
        if event.change_type == "wall_opened":
            rule.enables_path = True

        # Set trigger region — always use player position as trigger area
        # For proximity: tight radius around player
        # For passage: wider radius (player recently passed through this area)
        pr, pc = event.trigger_pos
        if trigger_type == "proximity":
            r = 3
        else:
            r = 8  # wider search area for passage triggers
        rule.trigger_region = (max(0, pr - r), max(0, pc - r),
                               min(59, pr + r), min(63, pc + r))

        return rule

    def _matches(self, rule: LearnedRule, event: ReactionEvent) -> bool:
        """Does this event match an existing rule?"""
        if rule.effect_type != event.change_type:
            return False
        # Same effect region (within margin)
        if not self._bbox_overlaps(rule.effect_region, event.change_bbox):
            return False
        return True

    @staticmethod
    def _bbox_overlaps(
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
        margin: int = 5,
    ) -> bool:
        """Return True if the two bounding boxes overlap when expanded by margin."""
        r1_min, c1_min, r1_max, c1_max = bbox1
        r2_min, c2_min, r2_max, c2_max = bbox2

        # Expand by margin
        r1_min -= margin; c1_min -= margin
        r1_max += margin; c1_max += margin

        # Standard rectangle overlap test
        if r1_max < r2_min or r2_max < r1_min:
            return False
        if c1_max < c2_min or c2_max < c1_min:
            return False
        return True


# ---------------------------------------------------------------------------
# Class 3: DynamicPlanner
# ---------------------------------------------------------------------------

class DynamicPlanner:
    """
    Plans routes that incorporate learned rules as prerequisites.

    Typical flow:
      1. Collect trigger waypoints from confirmed (or high-confidence) rules.
      2. Order waypoints greedily (nearest-neighbour).
      3. BFS through each waypoint segment.
      4. Return the concatenated action sequence.
    """

    def plan_with_rules(
        self,
        player_pos: Tuple[int, int],
        goals: List[Tuple[int, int]],
        detours: List[Tuple[int, int]],
        rules: List[LearnedRule],
        smap,
        mv_actions: Dict[int, Tuple[int, int]],
        offsets,
        budget: int,
    ) -> Optional[List[int]]:
        """
        Plan a route visiting required trigger points before goals.

        Parameters
        ----------
        player_pos  : current (row, col) of the player
        goals       : list of goal positions to reach
        detours     : additional intermediate waypoints (from existing logic)
        rules       : learned rules from RuleLearner
        smap        : spatial map with can_occupy(r, c, offsets) method
        mv_actions  : {action_idx: (delta_row, delta_col)}
        offsets     : passed through to smap.can_occupy
        budget      : maximum number of actions to return

        Returns
        -------
        List of action indices, or None if no rules apply (caller falls back).
        """
        # Collect waypoints from path-enabling rules
        trigger_waypoints: List[Tuple[int, int]] = []
        for rule in rules:
            if rule.enables_path and rule.trigger_region is not None:
                r_min, c_min, r_max, c_max = rule.trigger_region
                wp = ((r_min + r_max) // 2, (c_min + c_max) // 2)
                trigger_waypoints.append(wp)

        if not trigger_waypoints:
            return None  # Nothing to add — caller uses its own planner

        # Build ordered waypoint list: trigger points + existing detours + goals
        all_waypoints = trigger_waypoints + detours + goals

        # Greedy nearest-neighbour ordering
        ordered: List[Tuple[int, int]] = []
        current = player_pos
        remaining = list(all_waypoints)
        while remaining:
            nearest = min(
                remaining,
                key=lambda p: abs(p[0] - current[0]) + abs(p[1] - current[1]),
            )
            ordered.append(nearest)
            current = nearest
            remaining.remove(nearest)

        # BFS through each segment
        full_route: List[int] = []
        current = player_pos
        for waypoint in ordered:
            if len(full_route) >= budget:
                break
            segment = self._bfs_path(current, waypoint, smap, mv_actions, offsets)
            if segment:
                remaining_budget = budget - len(full_route)
                full_route.extend(segment[:remaining_budget])
                current = waypoint

        return full_route[:budget] if full_route else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bfs_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        smap,
        mv_actions: Dict[int, Tuple[int, int]],
        offsets,
        max_nodes: int = 8000,
        max_depth: int = 100,
    ) -> Optional[List[int]]:
        """
        BFS from start to goal.

        Returns a list of action indices, or the closest reachable path if
        the exact goal is unreachable.
        """
        if start == goal:
            return []

        distances: Dict[Tuple[int, int], int] = {start: 0}
        paths: Dict[Tuple[int, int], List[int]] = {start: []}
        queue: deque = deque([(start, [])])

        while queue and len(distances) < max_nodes:
            (cr, cc), path = queue.popleft()

            if len(path) >= max_depth:
                continue

            if (cr, cc) == goal:
                return path

            for aidx, (dr, dc) in mv_actions.items():
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in distances:
                    continue
                if not smap.can_occupy(nr, nc, offsets):
                    continue
                new_path = path + [aidx]
                distances[(nr, nc)] = len(new_path)
                paths[(nr, nc)] = new_path
                queue.append(((nr, nc), new_path))

        # Exact goal found (possibly hit inside loop before exit condition)
        if goal in paths:
            return paths[goal]

        # Return the path to the closest reachable position
        best_dist = float("inf")
        best_path: List[int] = []
        for pos, path in paths.items():
            d = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            if d < best_dist:
                best_dist = d
                best_path = path

        return best_path if best_path else None
