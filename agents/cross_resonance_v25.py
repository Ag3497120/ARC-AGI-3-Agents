"""
CrossResonanceV25 — Game-agnostic Cross Structural Agent

v25: Replaces all game-specific heuristics with pure structural reasoning.
  - StructureAnalyzer: 64×64 grid → StructuralMap (wall/corridor/player/goal/...)
  - PlanningEngine: BFS route + smart probe ordering
  - LiveMonitor: frame-by-frame diff, wall/corridor learning, reaction detection

No LLM. No hardcoded game knowledge. Decisions flow entirely from:
  1. Structural analysis (what IS each region?)
  2. Movement physics (how does the player move?)
  3. Reaction patterns (what happens when we interact?)
  4. Goal inference (what looks like a goal by structure?)

MAX_ACTIONS = 800
"""

from __future__ import annotations

import sys
import logging
from collections import deque
from typing import List, Dict, Tuple, Set, Optional, Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from .agent import Agent
from .cross_engine.cross_sensor import CrossSensor, CrossSnapshot
from .cross_engine.structure_analyzer import StructureAnalyzer, StructuralMap, Region
from .cross_engine.planning_engine import PlanningEngine, PlanResult
from .cross_engine.live_monitor import LiveMonitor, Reaction

logger = logging.getLogger(__name__)


# ─── Action index ↔ GameAction mapping ───────────────────────────────────────

ALL_ACTIONS: List[GameAction] = [
    GameAction.ACTION1, GameAction.ACTION2,
    GameAction.ACTION3, GameAction.ACTION4,
    GameAction.ACTION5, GameAction.ACTION6,
    GameAction.ACTION7,
]
ACTION_IDX_TO_GAMEACTION: Dict[int, GameAction] = {
    i: a for i, a in enumerate(ALL_ACTIONS)
}
CLICK_ACTION_IDX = 5   # ACTION6 = click

# Direction indices (convention: 0=UP 1=DOWN 2=LEFT 3=RIGHT)
DIR_UP    = 0
DIR_DOWN  = 1
DIR_LEFT  = 2
DIR_RIGHT = 3


# ─── Execution phases ────────────────────────────────────────────────────────

PHASE_RESET    = 'reset'
PHASE_ANALYZE  = 'analyze'
PHASE_PROBE    = 'probe'
PHASE_PLAN     = 'plan'
PHASE_EXECUTE  = 'execute'


# ─── Main agent class ────────────────────────────────────────────────────────

class CrossResonanceV25(Agent):
    """
    Game-agnostic Cross Structural Agent v25.

    Works on any ARC-AGI-3 game (maze, click puzzle, mixed) without
    game-specific code. All behavior emerges from structural analysis.
    """

    MAX_ACTIONS: int = 800

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def __init__(self, card_id, game_id, agent_name, ROOT_URL,
                 record, arc_env, tags=None):
        super().__init__(card_id, game_id, agent_name, ROOT_URL,
                         record, arc_env, tags)
        self._do_reset()

    def _do_reset(self) -> None:
        """Full reset (called at start and on level transitions)."""
        # Engines
        self.cross_sensor    = CrossSensor()
        self.struct_analyzer = StructureAnalyzer()
        self.planner         = PlanningEngine()
        self.monitor: Optional[LiveMonitor] = None

        # State
        self.phase: str = PHASE_ANALYZE
        self.prev_grid: Optional[np.ndarray] = None
        self.smap: Optional[StructuralMap] = None

        # Player tracking
        self.player_pos: Optional[Tuple[int, int]] = None
        self.player_offsets: List[Tuple[int, int]] = [(0, 0)]

        # Movement learning (from probing)
        self.movement_vectors: Dict[int, Tuple[int, int]] = {}   # aidx → (dr, dc)
        self.available_action_idxs: List[int] = [0, 1, 2, 3]
        self.is_click_game: bool = False

        # Probe state
        self.probe_order: List[int] = []
        self.probe_results: Dict[int, Tuple[int, int]] = {}   # aidx → (dr,dc)
        self.probe_corridor_colors: Set[int] = set()

        # Execution state
        self.action_queue: List[int] = []
        self.waypoints: List[Tuple[int, int]] = []
        self.last_action_idx: int = 0
        self.replan_cooldown: int = 0
        self.no_progress_count: int = 0
        self.prev_player_pos: Optional[Tuple[int, int]] = None

        # Click state
        self.click_targets: List[Tuple[int, int]] = []

        # Level tracking
        self.prev_levels_completed: int = 0
        self.total_actions: int = 0
        self._last_was_blocked: bool = False
        self._fallback_cycle: int = 0

        logger.debug("CrossResonanceV25: full reset")

    # ── Agent interface ──────────────────────────────────────────────────────

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in (GameState.WIN, GameState.GAME_OVER)

    def choose_action(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self.total_actions += 1
        grid = latest_frame.frame[0]

        # ── Handle game-over / not-played ─────────────────────────────────
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._do_reset()
            return GameAction.RESET

        # ── Level transition ───────────────────────────────────────────────
        if latest_frame.levels_completed != self.prev_levels_completed:
            logger.info(
                f"V25: level {self.prev_levels_completed}→{latest_frame.levels_completed}"
            )
            self.prev_levels_completed = latest_frame.levels_completed
            self._do_reset()
            self.prev_levels_completed = latest_frame.levels_completed

        # ── Phase: ANALYZE (frame 0 of each level) ────────────────────────
        if self.phase == PHASE_ANALYZE:
            return self._phase_analyze(grid, latest_frame)

        # ── Post-action monitoring ─────────────────────────────────────────
        self._post_action_update(grid)

        # ── Phase: PROBE ──────────────────────────────────────────────────
        if self.phase == PHASE_PROBE:
            return self._phase_probe(grid, latest_frame)

        # ── Phase: PLAN ───────────────────────────────────────────────────
        if self.phase == PHASE_PLAN:
            self._phase_plan(grid)

        # ── Phase: EXECUTE ────────────────────────────────────────────────
        return self._phase_execute(grid, latest_frame)

    # ── Phase implementations ────────────────────────────────────────────────

    def _phase_analyze(self, grid, latest_frame: FrameData) -> GameAction:
        """Phase 1: Structural analysis of initial frame."""
        self.prev_grid = np.array(grid, dtype=np.int32)

        # Structural analysis
        self.smap = self.struct_analyzer.analyze(grid)
        logger.info(f"V25 ANALYZE: {self.smap.cross_signature}")

        # Detect available actions
        avail = latest_frame.available_actions or ALL_ACTIONS[:4]
        self.available_action_idxs = [
            i for i, a in enumerate(ALL_ACTIONS) if a in avail
        ]
        if not self.available_action_idxs:
            self.available_action_idxs = [0, 1, 2, 3]

        # Detect click game
        self.is_click_game = (
            CLICK_ACTION_IDX in self.available_action_idxs
            and len(self.available_action_idxs) <= 2
        )

        # Initialize monitor
        self.monitor = LiveMonitor(
            initial_grid=grid,
            initial_signature=self.smap.cross_signature,
        )

        # Find initial player position
        if self.smap.player_region:
            pr = self.smap.player_region
            self.player_pos = (int(pr.centroid[0]), int(pr.centroid[1]))
            pr_cells = list(pr.cells)
            if pr_cells:
                self.player_offsets = [
                    (r - self.player_pos[0], c - self.player_pos[1])
                    for r, c in pr_cells
                ]
        elif self.smap.interactive_objects:
            first = self.smap.interactive_objects[0]
            self.player_pos = (int(first.centroid[0]), int(first.centroid[1]))

        # Plan probe order toward most important goal
        probe_dirs = self.planner.plan_smart_probe(
            self.player_pos or (32, 32),
            self.smap.goal_candidates,
        )
        self.probe_order = [
            d for d in probe_dirs if d in self.available_action_idxs
        ]
        # Ensure all available non-click actions are probed
        for aidx in self.available_action_idxs:
            if aidx != CLICK_ACTION_IDX and aidx not in self.probe_order:
                self.probe_order.append(aidx)

        logger.info(
            f"V25 ANALYZE done: player={self.player_pos} "
            f"goals={len(self.smap.goal_candidates)} "
            f"interactives={len(self.smap.interactive_objects)} "
            f"probe_order={self.probe_order}"
        )

        self.phase = PHASE_PROBE

        # If click game, skip movement probe and go straight to click plan
        if self.is_click_game:
            self.phase = PHASE_PLAN
            return self._phase_plan_click(grid)

        # First probe action
        return self._do_probe()

    def _phase_probe(self, grid, latest_frame: FrameData) -> GameAction:
        """Phase 2: Execute smart probes to learn movement vectors."""
        # Learn movement from last probe
        if self.prev_player_pos is not None and self.player_pos is not None:
            expected = self.probe_results.get(self.last_action_idx, (0, 0))
            if expected != (0, 0):
                result = self.monitor.learn_from_movement(
                    self.last_action_idx,
                    self.prev_player_pos,
                    self.player_pos,
                    expected,
                )
                if result.blocked and self.smap:
                    for r, c in result.new_walls:
                        self.smap.mark_wall(r, c)
                elif self.smap:
                    for r, c in result.new_corridors:
                        self.smap.mark_corridor(r, c)

        if self.probe_order:
            return self._do_probe()

        # All probes done — promote learned corridor colors to full map
        if self.probe_corridor_colors and self.smap:
            for color in self.probe_corridor_colors:
                promoted = self.smap.promote_color_to_corridor(color)
                if promoted > 0:
                    logger.info(f"V25 CORRIDOR_PROMOTE: color={color} → {promoted} cells")

        logger.info(
            f"V25 PROBE done: movement_vectors={self.movement_vectors} "
            f"corridor_colors={self.probe_corridor_colors}"
        )
        self.phase = PHASE_PLAN
        self._phase_plan(grid)
        return self._phase_execute(grid, None)

    def _do_probe(self) -> GameAction:
        """Execute the next probe direction."""
        aidx = self.probe_order.pop(0)
        self.last_action_idx = aidx
        self.prev_player_pos = self.player_pos

        # Optimistic expected movement (will be confirmed by actual position change)
        expected = self._aidx_to_expected_movement(aidx)
        self.probe_results[aidx] = expected

        ga = ACTION_IDX_TO_GAMEACTION.get(aidx, GameAction.ACTION1)
        ga.reasoning = f"v25 probe dir={aidx}"
        return ga

    def _aidx_to_expected_movement(self, aidx: int) -> Tuple[int, int]:
        """Best-guess movement for a direction index (before confirmation)."""
        # If we have confirmed movement for this idx, use it
        if aidx in self.movement_vectors:
            return self.movement_vectors[aidx]
        # Default conventions: 0=UP(-5,0), 1=DOWN(+5,0), 2=LEFT(0,-5), 3=RIGHT(0,+5)
        defaults = {
            DIR_UP: (-5, 0),
            DIR_DOWN: (5, 0),
            DIR_LEFT: (0, -5),
            DIR_RIGHT: (0, 5),
        }
        return defaults.get(aidx, (0, 0))

    def _phase_plan(self, grid) -> None:
        """Phase 3: Plan action queue from current position."""
        if not self.smap or not self.player_pos:
            return

        # Rebuild structural map with learned corridor colors
        if self.probe_corridor_colors:
            self.smap = self.struct_analyzer.analyze(grid)
            for r, c in self.monitor.get_all_confirmed_corridors():
                self.smap.mark_corridor(r, c)
            for r, c in self.monitor.get_all_confirmed_walls():
                self.smap.mark_wall(r, c)

        result: PlanResult = self.planner.plan(
            smap=self.smap,
            player_pos=self.player_pos,
            movement_vectors=self.movement_vectors,
            goal_candidates=self.smap.goal_candidates,
            interactive_objects=self.smap.interactive_objects,
        )

        self.action_queue = result.action_queue
        self.waypoints = result.waypoints
        self.phase = PHASE_EXECUTE
        self.no_progress_count = 0

        logger.info(
            f"V25 PLAN: method={result.method} "
            f"actions={len(self.action_queue)} "
            f"score={result.plan_score:.3f}"
        )

    def _phase_plan_click(self, grid) -> GameAction:
        """Plan click targets for click/puzzle games."""
        if not self.smap:
            ga = GameAction.ACTION6
            ga.coordinate = (32, 32)
            ga.reasoning = 'v25 click fallback'
            return ga

        # Collect all interactive and pattern regions as click targets
        targets: List[Tuple[int, int]] = []
        for reg in self.smap.interactive_objects:
            targets.append((int(reg.centroid[0]), int(reg.centroid[1])))
        for reg in self.smap.goal_candidates:
            targets.append((int(reg.centroid[0]), int(reg.centroid[1])))

        self.click_targets = targets
        self.phase = PHASE_EXECUTE
        return self._do_click()

    def _phase_execute(self, grid, latest_frame) -> GameAction:
        """Phase 4: Execute planned actions with live monitoring."""
        # If last action was blocked, purge same direction from queue and try alt
        if (hasattr(self, '_last_was_blocked') and self._last_was_blocked
                and self.action_queue):
            blocked_dir = self.last_action_idx
            # Remove leading actions in the blocked direction
            while self.action_queue and self.action_queue[0] == blocked_dir:
                self.action_queue.pop(0)
            self._last_was_blocked = False
            # If queue now empty, try perpendicular direction
            if not self.action_queue:
                perp = self._perpendicular_dirs(blocked_dir)
                self.action_queue = perp[:3]
                logger.info(f"V25 DETOUR: blocked={blocked_dir} → trying {perp[:3]}")

        # Replan if queue empty
        if not self.action_queue and not self.is_click_game:
            self.no_progress_count += 1
            if self.no_progress_count >= 3 and self.replan_cooldown <= 0:
                logger.info("V25 EXECUTE: queue empty, replanning")
                self._phase_plan(grid)
                self.replan_cooldown = max(len(self.action_queue), 3)

        # Decrement cooldown
        if self.replan_cooldown > 0:
            self.replan_cooldown -= 1

        # Click game branch
        if self.is_click_game:
            return self._do_click()

        # Movement game branch
        if self.action_queue:
            aidx = self.action_queue.pop(0)
        elif self.movement_vectors:
            # Cycle through all available directions to avoid getting stuck
            tried = getattr(self, '_fallback_cycle', 0)
            dirs = list(self.movement_vectors.keys())
            aidx = dirs[tried % len(dirs)]
            self._fallback_cycle = tried + 1
        else:
            aidx = DIR_UP

        self.last_action_idx = aidx
        self.prev_player_pos = self.player_pos

        ga = ACTION_IDX_TO_GAMEACTION.get(aidx, GameAction.ACTION1)
        ga.reasoning = (
            f"v25 exec q={len(self.action_queue)} "
            f"pos={self.player_pos} act={self.total_actions}"
        )
        return ga

    def _perpendicular_dirs(self, blocked_dir: int) -> List[int]:
        """Return perpendicular directions to try when blocked."""
        perp_map = {
            DIR_UP: [DIR_LEFT, DIR_RIGHT, DIR_DOWN],
            DIR_DOWN: [DIR_LEFT, DIR_RIGHT, DIR_UP],
            DIR_LEFT: [DIR_UP, DIR_DOWN, DIR_RIGHT],
            DIR_RIGHT: [DIR_UP, DIR_DOWN, DIR_LEFT],
        }
        return perp_map.get(blocked_dir, [DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT])

    def _do_click(self) -> GameAction:
        """Execute next click action."""
        if self.click_targets:
            pos = self.click_targets.pop(0)
        else:
            # Re-plan clicks from current smap
            if self.smap:
                self.click_targets = [
                    (int(r.centroid[0]), int(r.centroid[1]))
                    for r in self.smap.interactive_objects + self.smap.goal_candidates
                ]
            pos = self.click_targets.pop(0) if self.click_targets else (32, 32)

        ga = GameAction.ACTION6
        ga.coordinate = pos
        ga.reasoning = f"v25 click pos={pos} remaining={len(self.click_targets)}"
        self.last_action_idx = CLICK_ACTION_IDX
        return ga

    # ── Post-action state update ──────────────────────────────────────────────

    def _post_action_update(self, grid) -> None:
        """Called at the start of every non-first choose_action call."""
        if self.prev_grid is None:
            self.prev_grid = np.array(grid, dtype=np.int32)
            return

        curr_grid = np.array(grid, dtype=np.int32)

        # Update monitor
        if self.monitor:
            reaction = self.monitor.observe(grid)
            if reaction is not None:
                self._handle_reaction(reaction, grid)

        # Detect player movement from frame diff
        self._update_player_position(curr_grid)

        # Learn movement vector from confirmed movement
        if (self.prev_player_pos is not None
                and self.player_pos is not None):
            dr = self.player_pos[0] - self.prev_player_pos[0]
            dc = self.player_pos[1] - self.prev_player_pos[1]
            if (dr != 0 or dc != 0):
                if self.last_action_idx not in self.movement_vectors:
                    self.movement_vectors[self.last_action_idx] = (dr, dc)
                    # Infer opposite direction
                    opp = self._opposite_dir(self.last_action_idx)
                    if opp not in self.movement_vectors:
                        self.movement_vectors[opp] = (-dr, -dc)
                    logger.info(
                        f"V25 LEARNED: aidx={self.last_action_idx} → ({dr},{dc}), "
                        f"total vectors={len(self.movement_vectors)}"
                    )
                self.no_progress_count = 0
            elif self.last_action_idx in self.movement_vectors:
                # Expected to move but didn't → blocked
                expected = self.movement_vectors[self.last_action_idx]
                if self.monitor and self.prev_player_pos:
                    result = self.monitor.learn_from_movement(
                        self.last_action_idx,
                        self.prev_player_pos,
                        self.player_pos,
                        expected,
                    )
                    if result.blocked and self.smap:
                        for r, c in result.new_walls:
                            self.smap.mark_wall(r, c)
                        self._last_was_blocked = True
                        logger.info(f"V25 BLOCKED: aidx={self.last_action_idx}, new walls={len(result.new_walls)}")

        self.prev_grid = curr_grid.copy()

    def _update_player_position(self, curr_grid: np.ndarray) -> None:
        """Track player via diff — find the moved block matching player colors."""
        if self.prev_grid is None:
            return

        prev = self.prev_grid
        rows, cols = curr_grid.shape
        limit = min(rows, 60)  # exclude timer

        # Find cells where color changed
        diff_mask = (prev[:limit] != curr_grid[:limit])
        if not np.any(diff_mask):
            return

        # Find where player colors APPEARED (new position)
        player_colors: Set[int] = set()
        if self.smap and self.smap.player_region:
            # Use the stored multi-color set if available
            if self.smap.player_region.colors:
                player_colors = set(self.smap.player_region.colors)
            else:
                # Fallback: get ALL colors in player region cells
                for r, c in self.smap.player_region.cells:
                    player_colors.add(int(self.smap.grid[r, c]))
        if not player_colors:
            return

        # Cells where diff happened AND new color is a player color
        appeared: Set[Tuple[int, int]] = set()
        for r in range(limit):
            for c in range(cols):
                if diff_mask[r, c] and int(curr_grid[r, c]) in player_colors:
                    appeared.add((r, c))

        if len(appeared) < 3:
            return

        # Find largest connected component among appeared cells
        visited: Set[Tuple[int, int]] = set()
        best_component: List[Tuple[int, int]] = []
        for seed in appeared:
            if seed in visited:
                continue
            component: List[Tuple[int, int]] = []
            queue: deque = deque([seed])
            visited.add(seed)
            while queue:
                r, c = queue.popleft()
                component.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in appeared and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            if len(component) > len(best_component):
                best_component = component

        if len(best_component) < 3:
            return

        # Compute centroid of largest component
        rs = [r for r, c in best_component]
        cs = [c for r, c in best_component]
        new_pos = (sum(rs) // len(rs), sum(cs) // len(cs))

        if new_pos != self.player_pos:
            self.prev_player_pos = self.player_pos
            self.player_pos = new_pos
            self.player_offsets = [(r - new_pos[0], c - new_pos[1]) for r, c in best_component]

            # Mark traversed cells as corridor
            if self.smap:
                for r, c in best_component:
                    color_under = int(prev[r, c])
                    if color_under not in self.probe_corridor_colors:
                        self.probe_corridor_colors.add(color_under)
                        promoted = self.smap.promote_color_to_corridor(color_under)
                        if promoted > 0:
                            logger.info(f"V25 NEW_CORRIDOR: color={color_under} → {promoted} cells")
                    self.smap.mark_corridor(r, c)
                if self.monitor:
                    for r, c in best_component:
                        self.monitor.confirmed_corridors.add((r, c))
                    self.monitor.update_player_footprint(new_pos, self.player_offsets)

    def _handle_reaction(self, reaction: Reaction, grid) -> None:
        """React to a detected structural change — replan if significant."""
        if self.replan_cooldown > 0:
            return

        # Filter out player-movement-only changes:
        # If all changed cells are near the player, it's just the player moving
        if self.player_pos:
            pr, pc = self.player_pos
            near_player = sum(
                1 for r, c in reaction.changed_cells
                if abs(r - pr) <= 10 and abs(c - pc) <= 10
            )
            if near_player >= len(reaction.changed_cells) * 0.7:
                return  # Just player movement, not a real reaction

        logger.info(f"V25 REACTION: {reaction}")

        # Replan based on reaction type
        from .cross_engine.live_monitor import (
            REACTION_WALL_OPENED, REACTION_WALL_CLOSED,
        )
        if reaction.reaction_type in (REACTION_WALL_OPENED, REACTION_WALL_CLOSED):
            # Full reanalysis + replan
            self.smap = self.struct_analyzer.analyze(grid)
            self._apply_monitor_knowledge()
            self._phase_plan(grid)
            self.replan_cooldown = max(len(self.action_queue), 5)
        elif len(reaction.changed_cells) >= 30:
            # Significant structural change — replan
            self._phase_plan(grid)
            self.replan_cooldown = 5

    def _apply_monitor_knowledge(self) -> None:
        """Apply confirmed wall/corridor knowledge to current smap."""
        if not self.smap or not self.monitor:
            return
        for r, c in self.monitor.get_all_confirmed_walls():
            self.smap.mark_wall(r, c)
        for r, c in self.monitor.get_all_confirmed_corridors():
            self.smap.mark_corridor(r, c)

    def _opposite_dir(self, aidx: int) -> int:
        """Return the action index for the opposite direction."""
        opp_map = {
            DIR_UP: DIR_DOWN,
            DIR_DOWN: DIR_UP,
            DIR_LEFT: DIR_RIGHT,
            DIR_RIGHT: DIR_LEFT,
        }
        return opp_map.get(aidx, aidx ^ 1)
