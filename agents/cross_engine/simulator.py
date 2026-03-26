"""
simulator.py — Cross World Simulator

The game world as a Cross structure. Built from a single frame, enables
unlimited internal simulation at zero API cost.
"""

from __future__ import annotations
import copy
from typing import Optional, Dict, Tuple, List
from collections import deque


class CrossCell:
    """A cell in the Cross structure. Holds color + spatial role."""
    __slots__ = ('color', 'role')

    def __init__(self, color: int, role: str = 'unknown'):
        self.color = color
        self.role = role  # 'corridor', 'wall', 'player', 'lock', 'border', 'timer', ...


class CrossWorld:
    """The game world as a Cross structure.
    Built from a single frame, enables unlimited internal simulation."""

    def __init__(self, grid: list):
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.raw = [row[:] for row in grid]  # keep raw grid

        # Classify every cell
        self.cells = [[CrossCell(grid[r][c]) for c in range(self.cols)]
                      for r in range(self.rows)]

        # Structural elements
        self.player_cells: List[Tuple[int, int]] = []
        self.player_pos: Optional[Tuple[int, int]] = None  # center
        self.player_shape: List[Tuple[int, int]] = []      # relative offsets from center
        self.lock_pos: Optional[Tuple[int, int]] = None
        self.lock_cells: List[Tuple[int, int]] = []        # absolute positions of lock border

        self._classify_cells()
        self._detect_player()
        self._detect_lock()

    # ── Classification ────────────────────────────────────────

    def _classify_cells(self):
        """Classify cells by color → structural role."""
        for r in range(self.rows):
            for c in range(self.cols):
                color = self.raw[r][c]
                if r >= 60:
                    self.cells[r][c].role = 'timer'
                elif color == 3:
                    self.cells[r][c].role = 'corridor'
                elif color == 4:
                    self.cells[r][c].role = 'wall'
                elif color == 5:
                    self.cells[r][c].role = 'border'
                elif color == 12:
                    self.cells[r][c].role = 'player'
                elif color == 9:
                    self.cells[r][c].role = 'player_or_pattern'
                elif color == 11:
                    self.cells[r][c].role = 'timer'
                elif color in (0, 1, 8):
                    self.cells[r][c].role = 'lock_detail'
                else:
                    self.cells[r][c].role = 'unknown'

    def _detect_player(self):
        """Find the player block using color 12 (unique to player)."""
        c12_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.raw[r][c] == 12 and r < 60
        ]
        if not c12_cells:
            return

        r_min = min(r for r, c in c12_cells)
        c_min = min(c for r, c in c12_cells)
        c_max = max(c for r, c in c12_cells)

        self.player_cells = []
        for r in range(r_min, r_min + 5):
            for c in range(c_min, c_max + 1):
                self.player_cells.append((r, c))

        center_r = r_min + 2
        center_c = (c_min + c_max) // 2
        self.player_pos = (center_r, center_c)
        self.player_shape = [(r - center_r, c - center_c) for r, c in self.player_cells]

        for r, c in self.player_cells:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.cells[r][c].role = 'player'

    def _detect_lock(self):
        """Find the lock/keyhole area."""
        c8_cells = [
            (r, c)
            for r in range(min(30, self.rows))
            for c in range(self.cols)
            if self.raw[r][c] == 8
        ]
        if c8_cells:
            self.lock_pos = (
                sum(r for r, c in c8_cells) // len(c8_cells),
                sum(c for r, c in c8_cells) // len(c8_cells),
            )
        else:
            lock_pattern = []
            for r in range(min(20, self.rows)):
                for c in range(self.cols):
                    if self.raw[r][c] == 9 and self._is_in_lock_area(r, c):
                        lock_pattern.append((r, c))
            if lock_pattern:
                self.lock_pos = (
                    sum(r for r, c in lock_pattern) // len(lock_pattern),
                    sum(c for r, c in lock_pattern) // len(lock_pattern),
                )

        # Find border cells (color 5) around lock
        if self.lock_pos:
            lr, lc = self.lock_pos
            for r in range(max(0, lr - 10), min(self.rows, lr + 10)):
                for c in range(max(0, lc - 10), min(self.cols, lc + 10)):
                    if self.raw[r][c] == 5:
                        self.lock_cells.append((r, c))

    def _is_in_lock_area(self, r: int, c: int) -> bool:
        """Check if (r,c) is inside a color-5 bordered region."""
        has_5 = [False, False, False, False]
        for dr in range(1, 10):
            if r - dr >= 0 and self.raw[r - dr][c] == 5:
                has_5[0] = True
            if r + dr < self.rows and self.raw[r + dr][c] == 5:
                has_5[1] = True
        for dc in range(1, 10):
            if c - dc >= 0 and self.raw[r][c - dc] == 5:
                has_5[2] = True
            if c + dc < self.cols and self.raw[r][c + dc] == 5:
                has_5[3] = True
        return all(has_5)

    # ── Pattern Extraction ────────────────────────────────────

    def get_player_pattern(self) -> list:
        """Extract the internal pattern of the player block.
        Returns list of (rel_r, rel_c) for color-9 cells relative to player center."""
        if self.player_pos is None:
            return []
        cr, cc = self.player_pos
        pattern = []
        for r, c in self.player_cells:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if self.raw[r][c] == 9:
                    pattern.append((r - cr, c - cc))
        return pattern

    def get_lock_pattern(self) -> list:
        """Extract the pattern inside the lock area.
        Returns list of (rel_r, rel_c) for color-9 cells relative to lock center."""
        if self.lock_pos is None:
            return []
        lr, lc = self.lock_pos
        pattern = []
        for r in range(max(0, lr - 8), min(self.rows, lr + 8)):
            for c in range(max(0, lc - 8), min(self.cols, lc + 8)):
                if self.raw[r][c] == 9 and self._is_in_lock_area(r, c):
                    pattern.append((r - lr, c - lc))
        return pattern

    # ── Cross Simulation Engine ───────────────────────────────

    def can_move_to(self, center_r: int, center_c: int) -> bool:
        """Check if the player block can occupy position centered at (center_r, center_c)."""
        for dr, dc in self.player_shape:
            nr, nc = center_r + dr, center_c + dc
            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                return False
            color = self.raw[nr][nc]
            role = self.cells[nr][nc].role
            if role in ('wall', 'timer'):
                return False
            if color in (4, 5):
                return False
        return True

    def simulate_move(self, center_r: int, center_c: int, action: int) -> Tuple[int, int]:
        """Simulate a move. Returns new center position (same if blocked).
        Movement = 5 cells per action."""
        deltas = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}
        dr, dc = deltas.get(action, (0, 0))
        new_r, new_c = center_r + dr, center_c + dc
        if self.can_move_to(new_r, new_c):
            return (new_r, new_c)
        return (center_r, center_c)

    def apply_rule(self, rule_fn, *args) -> 'CrossWorld':
        """Apply a rule function to modify the world state.
        Returns a new CrossWorld (immutable simulation)."""
        new_world = self.clone()
        rule_fn(new_world, *args)
        return new_world

    def clone(self) -> 'CrossWorld':
        """Deep copy for branching simulations."""
        new_world = CrossWorld.__new__(CrossWorld)
        new_world.rows = self.rows
        new_world.cols = self.cols
        new_world.raw = [row[:] for row in self.raw]
        new_world.cells = [
            [CrossCell(self.cells[r][c].color, self.cells[r][c].role)
             for c in range(self.cols)]
            for r in range(self.rows)
        ]
        new_world.player_cells = self.player_cells[:]
        new_world.player_pos = self.player_pos
        new_world.player_shape = self.player_shape[:]
        new_world.lock_pos = self.lock_pos
        new_world.lock_cells = self.lock_cells[:]
        return new_world

    def set_color(self, r: int, c: int, color: int, role: str = None):
        """Mutate a cell's color (and optionally role) in place."""
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.raw[r][c] = color
            self.cells[r][c].color = color
            if role is not None:
                self.cells[r][c].role = role

    # ── Pathfinding ───────────────────────────────────────────

    def find_optimal_path(self) -> List[int]:
        """BFS on simulated world → shortest path from player to lock.
        Zero API actions — all internal computation."""
        if self.player_pos is None or self.lock_pos is None:
            return []

        start = self.player_pos
        goal = self.lock_pos

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (r, c), path = queue.popleft()
            if abs(r - goal[0]) <= 5 and abs(c - goal[1]) <= 5:
                return path
            for action in [1, 2, 3, 4]:
                nr, nc = self.simulate_move(r, c, action)
                if (nr, nc) != (r, c) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [action]))
        return []

    def find_all_reachable(self) -> Dict[Tuple[int, int], List[int]]:
        """Explore ALL reachable positions from player start."""
        if self.player_pos is None:
            return {}
        start = self.player_pos
        reachable = {start: []}
        queue = deque([(start, [])])
        while queue:
            (r, c), path = queue.popleft()
            for action in [1, 2, 3, 4]:
                nr, nc = self.simulate_move(r, c, action)
                if (nr, nc) != (r, c) and (nr, nc) not in reachable:
                    new_path = path + [action]
                    reachable[(nr, nc)] = new_path
                    queue.append(((nr, nc), new_path))
        return reachable
