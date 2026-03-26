"""
primitives.py — Game Rule Primitives as Cross Operations

Each primitive is an atomic game rule that takes a CrossWorld and returns
a modified CrossWorld. These are the building blocks the RuleMixer uses to
discover which rules govern a particular puzzle.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulator import CrossWorld


class RulePrimitive:
    """Base class for all rule primitives."""
    name: str = 'base'
    description: str = 'Base rule primitive'

    def can_apply(self, world: 'CrossWorld') -> bool:
        """Check if this rule is applicable to the current world state."""
        return True

    def apply(self, world: 'CrossWorld') -> 'CrossWorld':
        """Apply the rule, return new world state (immutable)."""
        raise NotImplementedError

    def score(self, world: 'CrossWorld') -> float:
        """How well does this rule explain the world? 0.0–1.0"""
        return 0.5

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


# ── Concrete Rule Primitives ───────────────────────────────────────────────────

class MazeRule(RulePrimitive):
    """Movement through corridors. Color 3 = passable, color 4/5 = wall.
    This is the baseline rule — standard BFS on the unmodified grid."""

    name = 'maze'
    description = 'Basic corridor movement: color 3 passable, 4/5 wall'

    def can_apply(self, world: 'CrossWorld') -> bool:
        return world.player_pos is not None

    def apply(self, world: 'CrossWorld') -> 'CrossWorld':
        # MazeRule doesn't modify the world — the world *is* a maze.
        return world.clone()

    def score(self, world: 'CrossWorld') -> float:
        # Always applicable as fallback.
        return 0.5 if world.player_pos is not None else 0.0


class KeyMatchRule(RulePrimitive):
    """When player is adjacent to the lock, the lock wall opens.
    
    Logic:
    1. Detect the lock border (color 5 cells surrounding lock area)
    2. Change lock border cells (color 5) → corridor (color 3)
    3. Now BFS can path through the previously-blocked area
    
    This covers both 'pattern-match opens door' and 'proximity opens door'
    scenarios — in both cases the effective result is the same: the wall opens.
    """

    name = 'key_match'
    description = 'Lock border (color 5) opens → becomes corridor (color 3)'

    def can_apply(self, world: 'CrossWorld') -> bool:
        return world.lock_pos is not None and len(world.lock_cells) > 0

    def apply(self, world: 'CrossWorld') -> 'CrossWorld':
        new_world = world.clone()
        # Open all lock border cells
        for r, c in new_world.lock_cells:
            new_world.set_color(r, c, 3, 'corridor')
        # Also open color-9 cells inside the lock area that might block movement
        if new_world.lock_pos:
            lr, lc = new_world.lock_pos
            for r in range(max(0, lr - 8), min(new_world.rows, lr + 8)):
                for c in range(max(0, lc - 8), min(new_world.cols, lc + 8)):
                    if new_world.raw[r][c] in (9, 8, 0, 1):
                        if new_world._is_in_lock_area(r, c):
                            new_world.set_color(r, c, 3, 'corridor')
        return new_world

    def score(self, world: 'CrossWorld') -> float:
        """Score based on how much lock structure is present."""
        if world.lock_pos is None:
            return 0.0
        lock_cell_count = len(world.lock_cells)
        if lock_cell_count == 0:
            return 0.0
        # High score if there's a clear lock structure
        return min(1.0, lock_cell_count / 16.0)


class PatternMatchRule(RulePrimitive):
    """Lock opens only if player's color-9 pattern matches the lock's color-9 pattern.
    
    Stricter than KeyMatchRule: compares normalized patterns first.
    If they match, opens the lock. If they don't, no-op.
    """

    name = 'pattern_match'
    description = 'Lock opens when player pattern matches lock pattern'

    def can_apply(self, world: 'CrossWorld') -> bool:
        return world.lock_pos is not None and world.player_pos is not None

    def apply(self, world: 'CrossWorld') -> 'CrossWorld':
        player_pat = set(world.get_player_pattern())
        lock_pat = set(world.get_lock_pattern())

        # Patterns match (or one/both is empty — uniform player = any lock opens)
        if not player_pat or not lock_pat or player_pat == lock_pat:
            return KeyMatchRule().apply(world)
        return world.clone()  # No match → no change

    def score(self, world: 'CrossWorld') -> float:
        player_pat = world.get_player_pattern()
        lock_pat = world.get_lock_pattern()
        if not player_pat and not lock_pat:
            return 0.3
        if not player_pat or not lock_pat:
            return 0.2
        common = set(player_pat) & set(lock_pat)
        return len(common) / max(len(player_pat), len(lock_pat))


class ProximityRule(RulePrimitive):
    """Lock opens when player is adjacent to the lock border.
    
    The player doesn't need any specific pattern — just being close is enough.
    """

    name = 'proximity'
    description = 'Lock opens when player is adjacent to lock area'

    def can_apply(self, world: 'CrossWorld') -> bool:
        return world.lock_pos is not None and world.player_pos is not None

    def apply(self, world: 'CrossWorld') -> 'CrossWorld':
        # Same effect as KeyMatchRule — open the lock
        return KeyMatchRule().apply(world)

    def score(self, world: 'CrossWorld') -> float:
        if world.lock_pos is None or world.player_pos is None:
            return 0.0
        pr, pc = world.player_pos
        lr, lc = world.lock_pos
        dist = abs(pr - lr) + abs(pc - lc)
        return max(0.0, 1.0 - dist / 50.0)


class ReversiRule(RulePrimitive):
    """Cells between two same-colored endpoints flip color.
    When player moves to position X, any cells of different color between X
    and another cell of player's color get flipped."""

    name = 'reversi'
    description = 'Cells between same-colored endpoints flip'

    def can_apply(self, world: 'CrossWorld') -> bool:
        return world.player_pos is not None

    def apply(self, world: 'CrossWorld') -> 'CrossWorld':
        new_world = world.clone()
        if world.player_pos is None:
            return new_world

        pr, pc = world.player_pos
        player_color = 12  # color 12 = player top

        # For each of 4 directions, find same-colored endpoint and flip between
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cells_between = []
            r, c = pr + dr, pc + dc
            while 0 <= r < world.rows and 0 <= c < world.cols:
                if world.raw[r][c] == player_color:
                    # Found endpoint — flip all between
                    for fr, fc in cells_between:
                        cur = new_world.raw[fr][fc]
                        # Flip: corridor↔wall
                        if cur == 3:
                            new_world.set_color(fr, fc, 4, 'wall')
                        elif cur == 4:
                            new_world.set_color(fr, fc, 3, 'corridor')
                    break
                cells_between.append((r, c))
                r += dr
                c += dc

        return new_world

    def score(self, world: 'CrossWorld') -> float:
        return 0.1  # Low prior — reversi is unusual


class GravityRule(RulePrimitive):
    """Objects without support fall down.
    After any move, unsupported blocks drop until they hit floor or another block."""

    name = 'gravity'
    description = 'Unsupported objects fall downward'

    def can_apply(self, world: 'CrossWorld') -> bool:
        return world.player_pos is not None

    def apply(self, world: 'CrossWorld') -> 'CrossWorld':
        new_world = world.clone()
        if world.player_pos is None:
            return new_world

        pr, pc = world.player_pos
        # Move player block down until it hits something
        # Find lowest position player can occupy
        best_r = pr
        test_r = pr + 5
        while test_r < world.rows and new_world.can_move_to(test_r, pc):
            best_r = test_r
            test_r += 1

        if best_r != pr:
            # Update raw grid: clear old position, set new
            for dr, dc in new_world.player_shape:
                old_r, old_c = pr + dr, pc + dc
                if 0 <= old_r < new_world.rows and 0 <= old_c < new_world.cols:
                    new_world.set_color(old_r, old_c, 3, 'corridor')
            for dr, dc in new_world.player_shape:
                new_r, new_c = best_r + dr, pc + dc
                if 0 <= new_r < new_world.rows and 0 <= new_c < new_world.cols:
                    new_world.set_color(new_r, new_c, 9, 'player')
            new_world.player_pos = (best_r, pc)
            new_world.player_cells = [(best_r + dr, pc + dc) for dr, dc in new_world.player_shape]

        return new_world

    def score(self, world: 'CrossWorld') -> float:
        return 0.15  # Gravity is possible but not the most common rule


class FillRule(RulePrimitive):
    """When a region is completely enclosed by one color, it fills with that color.
    Or: when a line/row is complete, it clears."""

    name = 'fill'
    description = 'Enclosed regions fill; complete lines clear'

    def can_apply(self, world: 'CrossWorld') -> bool:
        return True

    def apply(self, world: 'CrossWorld') -> 'CrossWorld':
        new_world = world.clone()
        # Flood-fill from border with corridor color
        # Any unreachable interior region is "enclosed" → fill with wall color
        visited = set()
        queue = []

        # Start flood from all border cells
        for r in range(world.rows):
            for c in [0, world.cols - 1]:
                if world.raw[r][c] == 3:
                    queue.append((r, c))
                    visited.add((r, c))
        for c in range(world.cols):
            for r in [0, world.rows - 1]:
                if world.raw[r][c] == 3 and (r, c) not in visited:
                    queue.append((r, c))
                    visited.add((r, c))

        while queue:
            r, c = queue.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < world.rows and 0 <= nc < world.cols
                        and (nr, nc) not in visited
                        and world.raw[nr][nc] == 3):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        # Isolated corridor cells → fill them (enclosed region)
        for r in range(1, world.rows - 1):
            for c in range(1, world.cols - 1):
                if world.raw[r][c] == 3 and (r, c) not in visited:
                    new_world.set_color(r, c, 4, 'wall')

        return new_world

    def score(self, world: 'CrossWorld') -> float:
        return 0.1


class PatternStampRule(RulePrimitive):
    """When player overlaps a template area, the player's pattern is 'stamped' onto it.
    The overlapping cells take on the player's colors."""

    name = 'pattern_stamp'
    description = 'Player pattern stamps onto overlapping template area'

    def can_apply(self, world: 'CrossWorld') -> bool:
        return world.player_pos is not None and world.lock_pos is not None

    def apply(self, world: 'CrossWorld') -> 'CrossWorld':
        new_world = world.clone()
        if world.player_pos is None or world.lock_pos is None:
            return new_world

        pr, pc = world.player_pos
        lr, lc = world.lock_pos

        # Stamp the player's pattern centered on the lock
        for dr, dc in world.player_shape:
            nr, nc = lr + dr, lc + dc
            if 0 <= nr < world.rows and 0 <= nc < world.cols:
                # Get source cell from player
                src_r, src_c = pr + dr, pc + dc
                if 0 <= src_r < world.rows and 0 <= src_c < world.cols:
                    new_world.set_color(nr, nc, world.raw[src_r][src_c])

        return new_world

    def score(self, world: 'CrossWorld') -> float:
        return 0.15


class ConnectRule(RulePrimitive):
    """When two similar objects are adjacent/aligned, they merge or create a path between them."""

    name = 'connect'
    description = 'Similar adjacent/aligned objects merge or create paths'

    def can_apply(self, world: 'CrossWorld') -> bool:
        return world.player_pos is not None and world.lock_pos is not None

    def apply(self, world: 'CrossWorld') -> 'CrossWorld':
        new_world = world.clone()
        if world.player_pos is None or world.lock_pos is None:
            return new_world

        pr, pc = world.player_pos
        lr, lc = world.lock_pos

        # Draw a corridor from player to lock (horizontal then vertical)
        # Horizontal segment
        c_start = min(pc, lc)
        c_end = max(pc, lc)
        for c in range(c_start, c_end + 1):
            if new_world.raw[pr][c] in (4, 5):
                new_world.set_color(pr, c, 3, 'corridor')

        # Vertical segment
        r_start = min(pr, lr)
        r_end = max(pr, lr)
        for r in range(r_start, r_end + 1):
            if new_world.raw[r][lc] in (4, 5):
                new_world.set_color(r, lc, 3, 'corridor')

        return new_world

    def score(self, world: 'CrossWorld') -> float:
        return 0.1


# ── Convenience list of all primitives ────────────────────────────────────────

def all_primitives() -> list:
    """Return one instance of every available rule primitive."""
    return [
        MazeRule(),
        KeyMatchRule(),
        PatternMatchRule(),
        ProximityRule(),
        ReversiRule(),
        GravityRule(),
        FillRule(),
        PatternStampRule(),
        ConnectRule(),
    ]
