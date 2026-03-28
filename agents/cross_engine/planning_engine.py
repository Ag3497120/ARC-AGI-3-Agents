"""
planning_engine.py — Cross Planning Engine v25

Plans efficient routes combining goal-reaching and information-gathering.
No LLM. No game-specific logic. All paths derived from StructuralMap.

Responsibilities:
  1. Smart probe ordering (maximize progress toward goals while probing)
  2. BFS route planning respecting wall/corridor structure
  3. Detour optimization (go via interactive objects on the way to goal)
  4. Re-planning API (call plan() anytime map changes)
"""

from __future__ import annotations

from collections import deque
from typing import List, Dict, Tuple, Optional, Set, Any

from .structure_analyzer import StructuralMap, Region


# Direction index conventions (matches standard ARC-AGI-3 ACTION1-4)
DIR_UP    = 0   # ACTION1
DIR_DOWN  = 1   # ACTION2
DIR_LEFT  = 2   # ACTION3
DIR_RIGHT = 3   # ACTION4


class PlanResult:
    """Result of plan()."""
    __slots__ = ('action_queue', 'waypoints', 'plan_score', 'method')

    def __init__(self, action_queue: List[int], waypoints: List[Tuple[int, int]],
                 plan_score: float = 0.0, method: str = 'bfs'):
        self.action_queue = action_queue
        self.waypoints = waypoints
        self.plan_score = plan_score
        self.method = method

    def __repr__(self) -> str:
        return (f"PlanResult(actions={len(self.action_queue)}, "
                f"waypoints={len(self.waypoints)}, "
                f"score={self.plan_score:.3f}, method={self.method})")


class PlanningEngine:
    """
    Game-agnostic route planner over a StructuralMap.

    Usage:
        engine = PlanningEngine()
        result = engine.plan(smap, player_pos, movement_vectors, goal_candidates, interactives)
    """

    MAX_BFS_NODES = 12_000
    MAX_PATH_LEN = 200
    GOAL_VALUE = 100.0
    INTERACTIVE_VALUE = 10.0

    def plan_smart_probe(self, player_pos: Tuple[int, int],
                         goal_candidates: List[Region]) -> List[int]:
        """
        Return probe direction order [0,1,2,3] that prioritizes movement
        toward the most important goal candidate.
        """
        if not goal_candidates:
            return [DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT]

        best_goal = goal_candidates[0]
        pr, pc = player_pos
        gr, gc = int(best_goal.centroid[0]), int(best_goal.centroid[1])
        dr = gr - pr
        dc = gc - pc

        order: List[int] = []
        if dr < 0:
            order.append(DIR_UP)
        if dr > 0:
            order.append(DIR_DOWN)
        if dc < 0:
            order.append(DIR_LEFT)
        if dc > 0:
            order.append(DIR_RIGHT)

        # Fill remaining directions
        for d in [DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT]:
            if d not in order:
                order.append(d)

        return order

    def plan(
        self,
        smap: StructuralMap,
        player_pos: Tuple[int, int],
        movement_vectors: Dict[int, Tuple[int, int]],
        goal_candidates: List[Region],
        interactive_objects: List[Region],
        budget: int = 200,
    ) -> PlanResult:
        """
        Plan the best action sequence from player_pos toward goals.

        Args:
            smap: current structural map (used for passability)
            player_pos: (row, col) of player centroid
            movement_vectors: {action_idx: (delta_row, delta_col)} from probing
            goal_candidates: ordered list of goal regions
            interactive_objects: regions to pick up on the way
            budget: max actions to return

        Returns:
            PlanResult with action_queue and waypoints
        """
        if not movement_vectors:
            return PlanResult([], [], method='no_movement_vectors')

        # Build goal positions
        goal_positions = [
            (int(g.centroid[0]), int(g.centroid[1]))
            for g in goal_candidates
        ]
        interactive_positions = [
            (int(o.centroid[0]), int(o.centroid[1]))
            for o in interactive_objects
        ]

        # Player footprint offsets (will be empty if not tracked; BFS uses centroid)
        player_offsets: List[Tuple[int, int]] = [(0, 0)]
        if smap.player_region:
            pr, pc = player_pos
            player_offsets = [
                (r - pr, c - pc)
                for r, c in smap.player_region.cells
                if abs(r - pr) <= 10 and abs(c - pc) <= 10
            ]
            if not player_offsets:
                player_offsets = [(0, 0)]

        # BFS from player to all reachable positions
        distances, paths = self._bfs_all(
            player_pos, smap, movement_vectors, player_offsets
        )

        if not goal_positions and not interactive_positions:
            return PlanResult([], [], method='no_goals')

        # Direct routes to each goal
        best_direct = self._best_direct(
            goal_positions, paths, self.GOAL_VALUE
        )

        # Detour routes: player → interactive → goal
        best_detour = self._best_detour(
            player_pos, goal_positions, interactive_positions,
            paths, distances, smap, movement_vectors, player_offsets
        )

        # Pick the highest-scoring plan
        chosen: Optional[Tuple[List[int], List[Tuple[int, int]], float, str]] = None

        if best_direct:
            actions, wps, score = best_direct
            chosen = (actions, wps, score, 'direct')

        if best_detour:
            actions, wps, score = best_detour
            if chosen is None or score > chosen[2]:
                chosen = (actions, wps, score, 'detour')

        if chosen is None:
            # Fallback: push toward first goal
            if goal_positions:
                push = self._push_toward(player_pos, goal_positions[0], movement_vectors)
                return PlanResult(push[:budget], [goal_positions[0]],
                                  method='push_fallback')
            return PlanResult([], [], method='no_plan')

        action_queue, waypoints, score, method = chosen
        return PlanResult(action_queue[:budget], waypoints,
                          plan_score=score, method=method)

    # ── Internal BFS ──────────────────────────────────────────────────────────

    def _bfs_all(
        self,
        start: Tuple[int, int],
        smap: StructuralMap,
        movement_vectors: Dict[int, Tuple[int, int]],
        player_offsets: List[Tuple[int, int]],
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], List[int]]]:
        """
        BFS from start across passable cells.
        Returns (distances, paths) where paths[pos] = list of action indices.
        """
        distances: Dict[Tuple[int, int], int] = {start: 0}
        paths: Dict[Tuple[int, int], List[int]] = {start: []}
        queue: deque = deque()
        queue.append((start, []))

        while queue and len(distances) < self.MAX_BFS_NODES:
            (cr, cc), path = queue.popleft()
            if len(path) >= self.MAX_PATH_LEN:
                continue

            for aidx, (dr, dc) in movement_vectors.items():
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in distances:
                    continue
                # Check all cells of player footprint
                if not self._can_occupy(nr, nc, player_offsets, smap):
                    continue
                new_path = path + [aidx]
                distances[(nr, nc)] = len(new_path)
                paths[(nr, nc)] = new_path
                queue.append(((nr, nc), new_path))

        return distances, paths

    def _can_occupy(
        self,
        r: int, c: int,
        offsets: List[Tuple[int, int]],
        smap: StructuralMap,
    ) -> bool:
        """Check if player centroid can be at (r, c).
        
        BFS explores by centroid only — the full footprint collision is
        handled at runtime via blocked-movement detection and wall learning.
        This allows BFS to find paths through corridors that are narrower
        than the full player footprint (the game handles clipping).
        """
        if r < 0 or r >= 60 or c < 0 or c >= 64:
            return False
        return smap.is_passable(r, c)

    # ── Route scoring ─────────────────────────────────────────────────────────

    def _best_direct(
        self,
        goal_positions: List[Tuple[int, int]],
        paths: Dict[Tuple[int, int], List[int]],
        goal_value: float,
    ) -> Optional[Tuple[List[int], List[Tuple[int, int]], float]]:
        best_score = -1.0
        best_actions: List[int] = []
        best_wps: List[Tuple[int, int]] = []

        for gpos in goal_positions:
            # Check nearby cells if exact centroid unreachable
            candidates = [gpos] + self._neighbors(gpos, 3)
            for pos in candidates:
                if pos not in paths:
                    continue
                p = paths[pos]
                if not p:
                    continue
                score = goal_value / max(len(p), 1)
                if score > best_score:
                    best_score = score
                    best_actions = p
                    best_wps = [pos]

        if best_actions:
            return best_actions, best_wps, best_score
        return None

    def _best_detour(
        self,
        player_pos: Tuple[int, int],
        goal_positions: List[Tuple[int, int]],
        interactive_positions: List[Tuple[int, int]],
        paths: Dict[Tuple[int, int], List[int]],
        distances: Dict[Tuple[int, int], int],
        smap: StructuralMap,
        movement_vectors: Dict[int, Tuple[int, int]],
        player_offsets: List[Tuple[int, int]],
    ) -> Optional[Tuple[List[int], List[Tuple[int, int]], float]]:
        if not interactive_positions or not goal_positions:
            return None

        best_score = -1.0
        best_actions: List[int] = []
        best_wps: List[Tuple[int, int]] = []

        for ipos in interactive_positions[:5]:
            # Path player → interactive
            i_candidates = [ipos] + self._neighbors(ipos, 2)
            i_path: List[int] = []
            i_pos_actual: Optional[Tuple[int, int]] = None

            for pos in i_candidates:
                if pos in paths and paths[pos]:
                    if not i_path or len(paths[pos]) < len(i_path):
                        i_path = paths[pos]
                        i_pos_actual = pos

            if not i_path or i_pos_actual is None:
                continue

            # BFS from interactive to goals
            _, i_paths = self._bfs_all(
                i_pos_actual, smap, movement_vectors, player_offsets
            )

            for gpos in goal_positions[:3]:
                g_candidates = [gpos] + self._neighbors(gpos, 3)
                for gp in g_candidates:
                    if gp not in i_paths or not i_paths[gp]:
                        continue
                    total_cost = len(i_path) + len(i_paths[gp])
                    score = (self.GOAL_VALUE + self.INTERACTIVE_VALUE) / max(total_cost, 1)
                    if score > best_score:
                        best_score = score
                        best_actions = i_path + i_paths[gp]
                        best_wps = [i_pos_actual, gp]

        if best_actions:
            return best_actions, best_wps, best_score
        return None

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _neighbors(self, pos: Tuple[int, int], radius: int) -> List[Tuple[int, int]]:
        """Return nearby grid positions within Manhattan distance radius."""
        r, c = pos
        result = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) <= radius and (dr, dc) != (0, 0):
                    result.append((r + dr, c + dc))
        return result

    def _push_toward(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        movement_vectors: Dict[int, Tuple[int, int]],
    ) -> List[int]:
        """Greedy push toward target when BFS fails."""
        sr, sc = start
        tr, tc = target
        dr = tr - sr
        dc = tc - sc

        push: List[int] = []
        for aidx, (mr, mc) in movement_vectors.items():
            if mr < 0 and dr < 0:
                push.extend([aidx] * max(1, abs(dr) // 5))
            elif mr > 0 and dr > 0:
                push.extend([aidx] * max(1, abs(dr) // 5))
            elif mc < 0 and dc < 0:
                push.extend([aidx] * max(1, abs(dc) // 5))
            elif mc > 0 and dc > 0:
                push.extend([aidx] * max(1, abs(dc) // 5))

        return push[:20]
