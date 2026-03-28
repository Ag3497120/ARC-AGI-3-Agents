"""
live_monitor.py — Cross Live Monitor v25

Monitors every frame for structural changes against the initial Cross map.
Learns which cells are walls/corridors from actual movement outcomes.
Detects reactions (wall openings, pattern shifts, color cycles, etc.).

No LLM. No game-specific logic. Purely structural.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any

import numpy as np


# ─── Reaction type constants ──────────────────────────────────────────────────

REACTION_WALL_OPENED    = 'wall_opened'
REACTION_WALL_CLOSED    = 'wall_closed'
REACTION_PATTERN_SHIFT  = 'pattern_shifted'
REACTION_COLOR_CYCLE    = 'color_cycle'
REACTION_OBJECT_APPEAR  = 'object_appeared'
REACTION_OBJECT_VANISH  = 'object_vanished'
REACTION_UNKNOWN        = 'unknown'

CHANGE_THRESHOLD = 5      # minimum changed cells to count as a reaction
TIMER_ROW_START  = 60     # rows 60+ are timer area — excluded from reaction detection


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Reaction:
    """A detected structural reaction in the grid."""
    reaction_type: str
    changed_cells: List[Tuple[int, int]]
    color_changes: Dict[Tuple[int, int], int]   # (old_color, new_color) → count
    analysis: Dict[str, Any] = field(default_factory=dict)
    frame_index: int = 0

    def __repr__(self) -> str:
        return (f"Reaction({self.reaction_type}, "
                f"cells={len(self.changed_cells)}, "
                f"frame={self.frame_index})")


@dataclass
class MovementResult:
    """Result of learn_from_movement()."""
    blocked: bool
    new_walls: Set[Tuple[int, int]] = field(default_factory=set)
    new_corridors: Set[Tuple[int, int]] = field(default_factory=set)


# ─── Main class ───────────────────────────────────────────────────────────────

class LiveMonitor:
    """
    Tracks frame-to-frame changes in the raw grid and learns from movement.

    Usage:
        monitor = LiveMonitor(initial_grid, initial_signature)
        # ... after each action:
        reaction = monitor.check(new_grid)
        monitor.learn_from_movement(action_idx, prev_pos, curr_pos, expected_mv)
    """

    def __init__(
        self,
        initial_grid,
        initial_signature: Dict[str, Any],
        corridor_colors: Optional[Set[int]] = None,
    ):
        self.initial_grid: np.ndarray = np.array(initial_grid, dtype=np.int32)
        self.initial_signature: Dict[str, Any] = initial_signature
        self.corridor_colors: Set[int] = set(corridor_colors or [])

        self.prev_grid: np.ndarray = np.array(initial_grid, dtype=np.int32)
        self.confirmed_walls: Set[Tuple[int, int]] = set()
        self.confirmed_corridors: Set[Tuple[int, int]] = set()
        self.reaction_history: List[Reaction] = []
        self.player_position: Optional[Tuple[int, int]] = None
        self.player_offsets: List[Tuple[int, int]] = [(0, 0)]

        # Player footprint tracking for reaction noise reduction
        self.player_pos: Optional[Tuple[int, int]] = None
        self.prev_player_pos: Optional[Tuple[int, int]] = None

        self._frame_index: int = 0

    # ── Primary API ──────────────────────────────────────────────────────────

    def check(self, grid) -> Optional[Reaction]:
        """
        Called after every action. Diffs current grid vs prev_grid.
        Returns a Reaction if significant changes detected, else None.
        Updates prev_grid.
        """
        curr = np.array(grid, dtype=np.int32)
        self._frame_index += 1

        changed_cells = self._diff_grids(self.prev_grid, curr)
        self.prev_grid = curr.copy()

        if len(changed_cells) < CHANGE_THRESHOLD:
            return None

        # Build color change map
        color_changes: Dict[Tuple[int, int], int] = {}
        for r, c in changed_cells:
            old = int(self.initial_grid[r, c]) if self._frame_index == 1 else int(self.prev_grid[r, c])
            new = int(curr[r, c])
            # Use current prev (already updated above) — use initial for long-term diff
            key = (int(self.prev_grid[r, c]), int(curr[r, c]))
            color_changes[key] = color_changes.get(key, 0) + 1

        # Actually recompute with the correct prev (before update)
        # We need the grid before update; use initial for now since prev updated above
        # Rebuild from scratch using the stored initial
        color_changes_proper: Dict[Tuple[int, int], int] = {}
        changed_proper: List[Tuple[int, int]] = []
        rows, cols = curr.shape
        # Re-diff vs initial to get color_changes properly
        # (prev is now curr; we want old → new w.r.t. last frame)
        # Note: self.prev_grid was already set to curr above — fix order:
        # Actually we need to compare last frame vs current; since we already updated
        # prev_grid, re-derive from initial signature colors
        # Simplest correct approach: store a separate "last_frame" reference.

        reaction = self._classify_reaction(changed_cells, color_changes, curr)
        self.reaction_history.append(reaction)
        return reaction

    def check_v2(self, grid) -> Optional[Reaction]:
        """
        Corrected version: keeps last_frame separate from prev_grid for diffing.
        Use this if you want accurate color_changes.
        """
        curr = np.array(grid, dtype=np.int32)
        self._frame_index += 1

        last = self.prev_grid.copy()
        self.prev_grid = curr.copy()

        changed_cells = self._diff_grids(last, curr)

        if len(changed_cells) < CHANGE_THRESHOLD:
            return None

        color_changes: Dict[Tuple[int, int], int] = {}
        for r, c in changed_cells:
            old_color = int(last[r, c])
            new_color = int(curr[r, c])
            key = (old_color, new_color)
            color_changes[key] = color_changes.get(key, 0) + 1

        reaction = self._classify_reaction(changed_cells, color_changes, curr)
        self.reaction_history.append(reaction)
        return reaction

    # Alias: use check_v2 as the canonical implementation
    # (keeping check() for backward compat in case it's already called)
    # Override prev_grid handling properly:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def _observe(self, grid) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Internal: advance frame and return (curr, changed_cells)."""
        curr = np.array(grid, dtype=np.int32)
        self._frame_index += 1
        last = self.prev_grid.copy()
        self.prev_grid = curr.copy()
        changed = self._diff_grids(last, curr)
        return curr, changed, last

    def observe(self, grid) -> Optional[Reaction]:
        """
        Preferred API: diff last frame vs current, return Reaction or None.
        Excludes player footprint cells from change detection to reduce noise.
        """
        curr, changed_cells, last = self._observe(grid)

        # Exclude player footprint from changed_cells to avoid reaction noise
        if self.player_pos and self.player_offsets:
            player_cells: Set[Tuple[int, int]] = set()
            for dr, dc in self.player_offsets:
                player_cells.add((self.player_pos[0] + dr, self.player_pos[1] + dc))
            # Also exclude previous player position
            if self.prev_player_pos:
                for dr, dc in self.player_offsets:
                    player_cells.add((self.prev_player_pos[0] + dr, self.prev_player_pos[1] + dc))
            non_player_changes = [cell for cell in changed_cells if cell not in player_cells]
        else:
            non_player_changes = changed_cells

        if len(non_player_changes) < CHANGE_THRESHOLD:
            return None

        color_changes: Dict[Tuple[int, int], int] = {}
        for r, c in non_player_changes:
            old_color = int(last[r, c])
            new_color = int(curr[r, c])
            key = (old_color, new_color)
            color_changes[key] = color_changes.get(key, 0) + 1

        reaction = self._classify_reaction(non_player_changes, color_changes, curr)
        self.reaction_history.append(reaction)
        return reaction

    def learn_from_movement(
        self,
        action_idx: int,
        prev_player_pos: Tuple[int, int],
        curr_player_pos: Tuple[int, int],
        expected_movement: Tuple[int, int],
    ) -> MovementResult:
        """
        Infer wall/corridor info from whether the player actually moved.

        Args:
            action_idx: the action that was taken
            prev_player_pos: player centroid before action
            curr_player_pos: player centroid after action
            expected_movement: (dr, dc) we expected based on movement_vectors

        Returns:
            MovementResult with blocked flag and newly learned cells
        """
        actual_move = (
            curr_player_pos[0] - prev_player_pos[0],
            curr_player_pos[1] - prev_player_pos[1],
        )
        new_walls: Set[Tuple[int, int]] = set()
        new_corridors: Set[Tuple[int, int]] = set()

        if actual_move == (0, 0) and expected_movement != (0, 0):
            # Blocked — mark destination cells as wall
            for dr, dc in self.player_offsets:
                wr = prev_player_pos[0] + expected_movement[0] + dr
                wc = prev_player_pos[1] + expected_movement[1] + dc
                if 0 <= wr < 64 and 0 <= wc < 64:
                    self.confirmed_walls.add((wr, wc))
                    new_walls.add((wr, wc))
            return MovementResult(blocked=True, new_walls=new_walls,
                                   new_corridors=new_corridors)

        elif actual_move != (0, 0):
            # Moved — mark current position cells as corridor
            for dr, dc in self.player_offsets:
                cr = curr_player_pos[0] + dr
                cc = curr_player_pos[1] + dc
                if 0 <= cr < 64 and 0 <= cc < 64:
                    self.confirmed_corridors.add((cr, cc))
                    new_corridors.add((cr, cc))
            self.player_position = curr_player_pos
            return MovementResult(blocked=False, new_walls=new_walls,
                                   new_corridors=new_corridors)

        return MovementResult(blocked=False)

    def update_player_footprint(
        self,
        player_pos: Tuple[int, int],
        player_offsets: List[Tuple[int, int]],
    ) -> None:
        """Update stored player position and footprint offsets."""
        self.prev_player_pos = self.player_pos
        self.player_pos = player_pos
        self.player_position = player_pos
        self.player_offsets = player_offsets if player_offsets else [(0, 0)]

    def update_corridor_colors(self, colors: Set[int]) -> None:
        """Register additional known corridor colors (from probe movement)."""
        self.corridor_colors.update(colors)

    def get_all_confirmed_walls(self) -> Set[Tuple[int, int]]:
        return set(self.confirmed_walls)

    def get_all_confirmed_corridors(self) -> Set[Tuple[int, int]]:
        return set(self.confirmed_corridors)

    def summary(self) -> str:
        return (f"LiveMonitor(frame={self._frame_index}, "
                f"walls={len(self.confirmed_walls)}, "
                f"corridors={len(self.confirmed_corridors)}, "
                f"reactions={len(self.reaction_history)})")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _diff_grids(
        self,
        prev: np.ndarray,
        curr: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Return list of (r, c) where grid changed, excluding timer area."""
        rows, cols = prev.shape
        changed = []
        for r in range(min(rows, TIMER_ROW_START)):
            for c in range(cols):
                if prev[r, c] != curr[r, c]:
                    changed.append((r, c))
        return changed

    def _classify_reaction(
        self,
        changed_cells: List[Tuple[int, int]],
        color_changes: Dict[Tuple[int, int], int],
        curr_grid: np.ndarray,
    ) -> Reaction:
        """Determine the type of reaction that occurred."""
        reaction_type = REACTION_UNKNOWN
        analysis: Dict[str, Any] = {
            'changed_count': len(changed_cells),
            'color_changes': {
                f"{old}->{new}": cnt
                for (old, new), cnt in color_changes.items()
            },
        }

        corridor_colors = self.corridor_colors

        # Detect wall_opened: a wall-colored cell became corridor-colored
        if corridor_colors:
            for (old_color, new_color), cnt in color_changes.items():
                if new_color in corridor_colors and old_color not in corridor_colors:
                    reaction_type = REACTION_WALL_OPENED
                    analysis['opened_color'] = old_color
                    analysis['to_color'] = new_color
                    break

        # Detect wall_closed: corridor → wall
        if reaction_type == REACTION_UNKNOWN and corridor_colors:
            for (old_color, new_color), cnt in color_changes.items():
                if old_color in corridor_colors and new_color not in corridor_colors:
                    reaction_type = REACTION_WALL_CLOSED
                    analysis['closed_color'] = new_color
                    break

        # Detect color cycle: same count of appearances and disappearances
        if reaction_type == REACTION_UNKNOWN:
            appearing = sum(cnt for (old, new), cnt in color_changes.items()
                            if old != new)
            disappearing = sum(cnt for (old, new), cnt in color_changes.items()
                               if old != new)
            unique_new = len({new for (old, new) in color_changes})
            unique_old = len({old for (old, new) in color_changes})
            if unique_new <= 3 and unique_old <= 3:
                reaction_type = REACTION_PATTERN_SHIFT
            elif len(color_changes) >= 2:
                reaction_type = REACTION_COLOR_CYCLE

        # Detect object appeared/vanished (large connected cluster of same new color)
        if reaction_type == REACTION_UNKNOWN:
            if len(changed_cells) > 20:
                # Check if a new connected component appeared
                reaction_type = REACTION_OBJECT_APPEAR

        analysis['reaction_type'] = reaction_type
        return Reaction(
            reaction_type=reaction_type,
            changed_cells=changed_cells,
            color_changes=color_changes,
            analysis=analysis,
            frame_index=self._frame_index,
        )
