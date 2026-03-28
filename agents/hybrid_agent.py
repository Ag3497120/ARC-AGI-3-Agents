"""
HybridAgent — Verantyx CrossEngine × DeepSeek V3.2

Architecture:
1. OBSERVE:    CrossSensor analyzes initial frame → structural snapshot
2. PROBE:      Send UP/DOWN/LEFT/RIGHT and record what changes (same as v24)
3. REPORT:     Verantyx compresses the observation into a concise text report (~50 lines)
4. STRATEGIZE: Send report to DeepSeek V3.2 → get high-level strategy
5. PLAN:       Use DeepSeek's strategy + Verantyx BFS/routing to compute action sequence
6. EXECUTE:    Run actions, monitor diffs
7. RE-STRATEGIZE: On significant changes or getting stuck, re-query DeepSeek with updated state
"""

import json
import logging
import os
import sys
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests

from arcengine import FrameData, GameAction, GameState

from .agent import Agent
from .cross_engine.cross_sensor import CrossSensor, CrossSnapshot, CrossObject
from .cross_resonance_agent import (
    CrossStructuralMap,
    RoutePlanner,
    DiffMonitor,
    ClickPlanner,
    ActionModel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_ACTIONS = [
    GameAction.ACTION1, GameAction.ACTION2,
    GameAction.ACTION3, GameAction.ACTION4,
    GameAction.ACTION5, GameAction.ACTION6,
    GameAction.ACTION7,
]
ACTION_IDS = {a: i for i, a in enumerate(ALL_ACTIONS)}

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_SYSTEM_PROMPT = (
    "You are a game strategy advisor for ARC-AGI-3 puzzles. "
    "Given structural analysis of a 64x64 grid game, determine the game type, goal, and optimal strategy. "
    "Be specific and actionable. "
    'Output a JSON with: {game_type, goal_description, targets: [{r,c,priority,reason}], '
    "strategy_steps: [str], action_type: 'keyboard'|'click'|'mixed'}"
)

# Minimum actions between DeepSeek calls to avoid excessive API usage
DEEPSEEK_CALL_INTERVAL = 15


# ---------------------------------------------------------------------------
# HybridAgent
# ---------------------------------------------------------------------------

class HybridAgent(Agent):
    """Hybrid agent: Verantyx CrossEngine for computation + DeepSeek V3.2 for reasoning."""

    MAX_ACTIONS = 800

    def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags)
        self._deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not self._deepseek_api_key:
            logger.warning("DEEPSEEK_API_KEY not set — DeepSeek calls will be skipped; falling back to heuristics")
        self._full_reset()

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def _full_reset(self):
        self.sensor = CrossSensor()
        self._snap: Optional[CrossSnapshot] = None
        self.prev_grid = None
        self.action_queue: List[int] = []
        self._prev_levels = 0
        self._phase = "observe"

        self._model = ActionModel()
        self._smap: Optional[CrossStructuralMap] = None
        self._planner: Optional[RoutePlanner] = None
        self._monitor: Optional[DiffMonitor] = None
        self._click = ClickPlanner()

        self._probe_queue: List[int] = []
        self._frame = 0
        self._actions = 0
        self._last_aidx = 0

        self._ctrl_pos: Optional[Tuple[int, int]] = None
        self._ctrl_offsets: List[Tuple[int, int]] = []
        self._prev_ctrl: Optional[Tuple[int, int]] = None
        self._last_click: Optional[Tuple[int, int]] = None
        self._click_targets: List[Tuple[int, int]] = []
        self._replan_cooldown = 0
        self._probe_corridor_colors: Set[int] = set()

        # DeepSeek state
        self._strategy: Optional[Dict[str, Any]] = None
        self._last_deepseek_action = -DEEPSEEK_CALL_INTERVAL  # allow call immediately
        self._deepseek_events: List[str] = []
        self._probe_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _observe(self, grid) -> CrossSnapshot:
        snap = self.sensor.observe(grid)
        self._snap = snap
        return snap

    def _budget(self, snap: CrossSnapshot) -> int:
        for obj in snap.objects:
            s = obj.descriptor.shape_axis
            p = obj.descriptor.position_axis
            if (s.get("is_hbar") or s.get("type") == "line") and p.get("in_timer_area"):
                n = obj.descriptor.scale_axis.get("cell_count", 0)
                if n > 5:
                    return max(n // 4 - 1, 8)
        cds = self.sensor.find_by_role_hint(snap, "countdown")
        if cds:
            return max(cds[0].descriptor.scale_axis.get("cell_count", 0) // 4 - 1, 8)
        return 50

    def _find_ctrl(self, snap: CrossSnapshot):
        if snap.diff and snap.diff.moved:
            for obj in sorted(snap.diff.moved, key=lambda o: -o.descriptor.scale_axis.get("cell_count", 0)):
                if obj.descriptor.position_axis.get("in_timer_area"):
                    continue
                if obj.descriptor.scale_axis.get("cell_count", 0) >= 3:
                    pos = obj.descriptor.position_axis.get("centroid_int")
                    cells = list(obj.cells)
                    self._prev_ctrl = self._ctrl_pos
                    self._ctrl_pos = pos
                    self._ctrl_offsets = [(r - pos[0], c - pos[1]) for r, c in cells]
                    if self.prev_grid is not None:
                        g = np.array(self.prev_grid)
                        for cr, cc in cells:
                            if 0 <= cr < 64 and 0 <= cc < 64:
                                self._probe_corridor_colors.add(int(g[cr, cc]))
                                if self._smap:
                                    self._smap.mark_passable(cr, cc)
                    return

    def _get_ctrl_mv(self, snap: CrossSnapshot) -> Tuple[int, int]:
        if snap.diff and snap.diff.moved:
            best = None
            best_sz = 0
            for obj in snap.diff.moved:
                if obj.descriptor.position_axis.get("in_timer_area"):
                    continue
                sc = obj.descriptor.scale_axis.get("cell_count", 0)
                if sc >= 3 and sc > best_sz:
                    mv = obj.descriptor.temporal_axis.get("movement", (0, 0))
                    if abs(mv[0]) >= 3 or abs(mv[1]) >= 3:
                        best = mv
                        best_sz = sc
            if best:
                return best
        return (0, 0)

    def _classify_targets(self, snap: CrossSnapshot) -> Tuple[List, List]:
        goals = []
        detours = []
        for obj in snap.objects:
            c = obj.descriptor.color_axis
            p = obj.descriptor.position_axis
            s = obj.descriptor.scale_axis
            rel = obj.descriptor.relation_axis
            if p.get("in_timer_area"):
                continue
            pos = p.get("centroid_int", (32, 32))
            if rel.get("contains_count", 0) > 0 and s.get("cell_count", 0) > 10:
                goals.append(pos)
            elif (rel.get("contained_by_count", 0) > 0
                  and s.get("size_category") in ("small", "medium")
                  and not c.get("is_dominant")):
                goals.append(pos)
            if c.get("is_rare") and s.get("size_category") in ("point", "tiny", "small"):
                detours.append(pos)
        return goals, detours

    def _make_plan(self, grid, snap: CrossSnapshot, budget: int) -> List[int]:
        pcl = self._probe_corridor_colors if self._probe_corridor_colors else None
        self._smap = CrossStructuralMap(grid, snap, probe_corridor_colors=pcl)
        logger.debug("SMAP: %s", self._smap.summary())

        mv_actions = self._model.get_mv_actions()
        if not mv_actions or not self._ctrl_pos:
            return []

        # Apply DeepSeek targets as goals if available
        goals, detours = self._classify_targets(snap)
        if self._strategy and self._strategy.get("targets"):
            ds_goals = [(t["r"], t["c"]) for t in sorted(
                self._strategy["targets"], key=lambda x: x.get("priority", 99)
            )]
            goals = ds_goals + goals  # DeepSeek targets take priority

        if self.prev_grid is not None:
            for dr, dc in self._ctrl_offsets:
                r, c = self._ctrl_pos[0] + dr, self._ctrl_pos[1] + dc
                if 0 <= r < 64 and 0 <= c < 64:
                    self._smap.mark_passable(r, c)

        self._planner = RoutePlanner(self._smap, mv_actions)
        route = self._planner.plan_route(
            self._ctrl_pos, goals, detours, self._ctrl_offsets, budget
        )
        logger.debug("PLAN: goals=%s detours=%s route=%d ctrl=%s", goals[:3], detours[:3], len(route), self._ctrl_pos)
        return route

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        grid,
        snap: CrossSnapshot,
        probe_results: Dict[str, Any],
        events: List[str],
    ) -> str:
        """Compress current game state into a text report for DeepSeek."""
        g = np.array(grid)
        unique, counts = np.unique(g[:60], return_counts=True)
        total_cells = 60 * 64

        # Color stats
        color_stats = []
        for color, count in sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1]):
            pct = count * 100 // total_cells
            color_stats.append((color, count, pct))

        # Panorama info
        n_colors = snap.panorama.get("unique_colors", len(unique))
        rare_colors = snap.panorama.get("rare_colors", [])

        # Player / ctrl info
        ctrl_info = "unknown"
        if self._ctrl_pos:
            ctrl_colors = []
            for dr, dc in self._ctrl_offsets:
                r, c = self._ctrl_pos[0] + dr, self._ctrl_pos[1] + dc
                if 0 <= r < 64 and 0 <= c < 64:
                    ctrl_colors.append(int(g[r, c]))
            ctrl_size = len(self._ctrl_offsets)
            w = int(np.sqrt(ctrl_size)) if ctrl_size > 0 else 1
            ctrl_info = f"position=({self._ctrl_pos[0]},{self._ctrl_pos[1]}), size={ctrl_size}({w}x{w}), colors={list(set(ctrl_colors))}"

        # Color roles (from smap if available)
        role_summary = ""
        if self._smap:
            for color, role in sorted(self._smap.color_roles.items()):
                if role not in ("unknown",):
                    role_summary += f"  color{color}: {role}\n"
        else:
            # Basic estimation from color frequency
            for color, count, pct in color_stats[:6]:
                if pct > 40:
                    role_summary += f"  color{color}: likely wall/background ({pct}%)\n"
                elif pct > 10:
                    role_summary += f"  color{color}: likely corridor ({pct}%)\n"

        # Objects of interest
        objects_desc = ""
        obj_count = 0
        for obj in snap.objects:
            p = obj.descriptor.position_axis
            if p.get("in_timer_area"):
                continue
            s = obj.descriptor.scale_axis
            c = obj.descriptor.color_axis
            rel = obj.descriptor.relation_axis
            pos = p.get("centroid_int", (0, 0))
            cells = s.get("cell_count", 0)
            size_cat = s.get("size_category", "?")
            color = c.get("primary_color", "?")
            is_rare = c.get("is_rare", False)
            contains = rel.get("contains_count", 0)
            contained_by = rel.get("contained_by_count", 0)

            if cells < 3:
                continue
            role_hint = ""
            if contains > 0:
                role_hint = "border/container"
            elif contained_by > 0:
                role_hint = "pattern/content"
            elif is_rare and size_cat in ("point", "tiny", "small"):
                role_hint = "marker"

            if role_hint or is_rare:
                obj_count += 1
                bbox = p.get("bounding_box", ((pos[0], pos[1]), (pos[0], pos[1])))
                tl = bbox[0] if bbox else pos
                br = bbox[1] if bbox else pos
                objects_desc += f"{obj_count}. {role_hint or size_cat} at ({tl[0]},{tl[1]})-({br[0]},{br[1]}), color={color}, cells={cells}\n"
            if obj_count >= 10:
                break

        # Probe results
        probe_lines = ""
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "ACTION5", 5: "CLICK"}
        for aidx, result in probe_results.items():
            aname = action_names.get(aidx, f"ACTION{aidx}")
            mv = result.get("movement", (0, 0))
            changed = result.get("cells_changed", 0)
            if mv != (0, 0):
                probe_lines += f"{aname}: player moved ({mv[0]:+d},{mv[1]:+d}), {changed} cells changed\n"
            else:
                probe_lines += f"{aname}: blocked (no movement), {changed} cells changed\n"

        # Recent events
        events_text = "\n".join(events[-10:]) if events else "No notable events yet"

        # Level info
        level_info = f"{self._prev_levels}"

        # Question based on phase
        if self._actions < 10:
            question = "What is the goal of this game? What strategy should I follow?"
        elif self._strategy:
            question = "Based on new events, should I update my strategy? What targets should I prioritize next?"
        else:
            question = "I'm stuck or making no progress. What should I do differently?"

        report = f"""=== GAME STATE ===
Game: {self.game_id}, Frame: {self._frame}, Level: {level_info}
Grid: 64x64, Colors: {n_colors} unique (rare: {rare_colors})
Available actions: [UP, DOWN, LEFT, RIGHT, CLICK, RESET]

=== STRUCTURAL ANALYSIS ===
Player: {ctrl_info}
Color roles:
{role_summary}
=== OBJECTS OF INTEREST ===
{objects_desc}
=== PROBE RESULTS ===
{probe_lines}
=== RECENT EVENTS ===
{events_text}

=== CURRENT QUESTION ===
{question}"""

        return report

    # ------------------------------------------------------------------
    # DeepSeek API
    # ------------------------------------------------------------------

    def _call_deepseek(self, report: str) -> Optional[Dict[str, Any]]:
        """Call DeepSeek V3.2 API with the game state report. Returns parsed strategy or None."""
        if not self._deepseek_api_key:
            logger.warning("Skipping DeepSeek call: no API key")
            return None

        logger.info("Calling DeepSeek (game=%s, frame=%d, actions=%d)", self.game_id, self._frame, self._actions)
        logger.debug("--- DeepSeek report ---\n%s\n---", report)

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                {"role": "user", "content": report},
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
        }
        headers = {
            "Authorization": f"Bearer {self._deepseek_api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            logger.info("DeepSeek raw response: %s", raw_content)

            # Try to parse JSON from response
            strategy = self._parse_deepseek_response(raw_content)
            if strategy:
                logger.info(
                    "DeepSeek strategy: type=%s goal=%s action_type=%s targets=%d steps=%d",
                    strategy.get("game_type", "?"),
                    strategy.get("goal_description", "?")[:60],
                    strategy.get("action_type", "?"),
                    len(strategy.get("targets", [])),
                    len(strategy.get("strategy_steps", [])),
                )
                self._last_deepseek_action = self._actions
                return strategy
            else:
                logger.warning("DeepSeek response JSON parse failed; falling back to heuristics")
                return None

        except requests.exceptions.Timeout:
            logger.error("DeepSeek API timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.error("DeepSeek API error: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error calling DeepSeek: %s", e)
            return None

    def _parse_deepseek_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from DeepSeek response text."""
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown code fences
        import re
        patterns = [
            r"```json\s*([\s\S]+?)\s*```",
            r"```\s*([\s\S]+?)\s*```",
            r"(\{[\s\S]+\})",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        return None

    def _should_call_deepseek(self) -> bool:
        """Return True if we should call DeepSeek now."""
        if not self._deepseek_api_key:
            return False
        actions_since_last = self._actions - self._last_deepseek_action
        return actions_since_last >= DEEPSEEK_CALL_INTERVAL

    def _log_event(self, event: str):
        self._deepseek_events.append(f"Frame {self._frame}: {event}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        grid = latest_frame.frame[0]
        self._frame += 1
        if self._replan_cooldown > 0:
            self._replan_cooldown -= 1

        # Handle reset / level up
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._full_reset()
            return GameAction.RESET

        if latest_frame.levels_completed != self._prev_levels:
            old = self._prev_levels
            self._full_reset()
            self._prev_levels = latest_frame.levels_completed
            self.prev_grid = [row[:] for row in grid]
            self._log_event(f"Level up {old}→{self._prev_levels}")
            logger.info("LEVEL_UP: %d→%d", old, self._prev_levels)
            return GameAction.ACTION1

        snap = self._observe(grid)
        budget = self._budget(snap)

        # Post-action analysis
        if self._actions > 0:
            ctrl_mv = self._get_ctrl_mv(snap)
            self._find_ctrl(snap)
            self._model.record(
                self._last_aidx,
                snap.diff is not None and snap.diff.has_changes,
                ctrl_mv,
            )

            # Record probe results
            if self._phase == "probe":
                aidx = self._last_aidx
                changed = 0
                if snap.diff and snap.diff.has_changes:
                    changed = (
                        len(snap.diff.moved)
                        + len(snap.diff.appeared)
                        + len(snap.diff.disappeared)
                        + len(snap.diff.size_changed)
                    )
                self._probe_results[aidx] = {
                    "movement": ctrl_mv,
                    "cells_changed": changed,
                }

            # Update smap with wall hits
            if self._smap and self._ctrl_pos:
                ctrl_mv = self._get_ctrl_mv(snap)
                if ctrl_mv == (0, 0):
                    known_mv = self._model.get_movement(self._last_aidx)
                    if known_mv != (0, 0):
                        for dr, dc in self._ctrl_offsets:
                            self._smap.mark_wall(
                                self._ctrl_pos[0] + known_mv[0] + dr,
                                self._ctrl_pos[1] + known_mv[1] + dc,
                            )

            # Monitor for significant reactions
            if self._monitor and self._replan_cooldown <= 0:
                reaction = self._monitor.check(snap)
                if reaction and reaction["total"] >= 3:
                    self._log_event(f"{reaction['total']} cells changed (reaction)")
                    logger.info("REACTION: f=%d %d changes — replanning", self._frame, reaction["total"])
                    # Re-strategize if we haven't called DeepSeek recently
                    if self._should_call_deepseek() and self._snap:
                        report = self._generate_report(
                            grid, snap, self._probe_results, self._deepseek_events
                        )
                        new_strategy = self._call_deepseek(report)
                        if new_strategy:
                            self._strategy = new_strategy
                    route = self._make_plan(grid, snap, budget)
                    self.action_queue = route
                    self._replan_cooldown = 5

            if self._model.is_click_game and self._last_click is not None:
                self._click.record(
                    self._last_click,
                    snap.diff is not None and snap.diff.has_changes,
                )
                self._last_click = None

        # ---- PHASE: OBSERVE ----
        if self._phase == "observe":
            available = latest_frame.available_actions or ALL_ACTIONS[:4]
            indices = [ACTION_IDS[a] for a in available if a in ACTION_IDS] or list(range(4))
            self._model.set_available(indices)
            self._monitor = DiffMonitor(snap)
            logger.info(
                "OBSERVE: objs=%d colors=%d rare=%s budget=%d",
                len(snap.objects),
                snap.panorama.get("unique_colors", 0),
                snap.panorama.get("rare_colors", []),
                budget,
            )
            self._probe_queue = list(indices)
            self._phase = "probe"

        # ---- PHASE: PROBE ----
        if self._phase == "probe":
            if self._probe_queue:
                idx = self._probe_queue.pop(0)
                self._last_aidx = idx
                self._prev_ctrl = self._ctrl_pos
                self._actions += 1
                self.prev_grid = [row[:] for row in grid]
                if self._model.is_click_game and idx == 5:
                    self._click.plan(snap, self.sensor)
                    cp = self._click.next()
                    if cp:
                        self._last_click = cp
                        a = GameAction.ACTION6
                        a.coordinate = cp
                        a.reasoning = f"probe click={cp}"
                        return a
                a = ALL_ACTIONS[idx] if idx < len(ALL_ACTIONS) else GameAction.ACTION1
                a.reasoning = f"probe idx={idx}"
                return a
            else:
                logger.info("PROBE_DONE: %s", self._model.summary())
                self._phase = "report"

        # ---- PHASE: REPORT + STRATEGIZE ----
        if self._phase == "report":
            report = self._generate_report(grid, snap, self._probe_results, self._deepseek_events)
            logger.debug("Generated report (%d chars)", len(report))

            strategy = self._call_deepseek(report)
            if strategy:
                self._strategy = strategy
                logger.info(
                    "Strategy set: %s | %s",
                    strategy.get("game_type", "?"),
                    strategy.get("goal_description", "?")[:80],
                )
                steps = strategy.get("strategy_steps", [])
                for i, step in enumerate(steps[:5]):
                    logger.info("  Step %d: %s", i + 1, step)
            else:
                logger.info("No DeepSeek strategy — using heuristics")

            self._phase = "plan"

        # ---- PHASE: PLAN ----
        if self._phase == "plan":
            if self._model.is_click_game:
                # For click games, use DeepSeek targets if available
                if self._strategy and self._strategy.get("action_type") in ("click", "mixed"):
                    targets = self._strategy.get("targets", [])
                    if targets:
                        self._click_targets = [(t["r"], t["c"]) for t in sorted(
                            targets, key=lambda x: x.get("priority", 99)
                        )]
                        self._click._queue = list(self._click_targets)
                        logger.info("Click targets from DeepSeek: %s", self._click_targets[:5])
                    else:
                        self._click_targets = self._click.plan(snap, self.sensor)
                else:
                    self._click_targets = self._click.plan(snap, self.sensor)
            else:
                route = self._make_plan(grid, snap, budget)
                self.action_queue = route
                logger.info("Plan: %d actions queued", len(route))
            self._phase = "execute"

        # ---- PHASE: EXECUTE ----
        if self._phase == "execute" and not self.action_queue and not self._model.is_click_game:
            if self._replan_cooldown <= 0:
                # Re-strategize if stuck and DeepSeek interval elapsed
                if self._should_call_deepseek():
                    self._log_event("Replanning (stuck or queue empty)")
                    report = self._generate_report(
                        grid, snap, self._probe_results, self._deepseek_events
                    )
                    new_strategy = self._call_deepseek(report)
                    if new_strategy:
                        self._strategy = new_strategy
                route = self._make_plan(grid, snap, budget)
                self.action_queue = route
                self._replan_cooldown = max(len(route), 3)

        self._prev_ctrl = self._ctrl_pos
        self.prev_grid = [row[:] for row in grid]
        self._actions += 1

        # --- Click game execution ---
        if self._model.is_click_game:
            cp = self._click.next()
            if cp:
                self._last_click = cp
                self._last_aidx = 5
                a = GameAction.ACTION6
                a.coordinate = cp
                a.reasoning = f"exec click={cp} left={self._click.remaining}"
                return a
            else:
                # Replan clicks: try DeepSeek if interval elapsed
                if self._should_call_deepseek():
                    self._log_event("Click replan (queue empty)")
                    report = self._generate_report(
                        grid, snap, self._probe_results, self._deepseek_events
                    )
                    new_strategy = self._call_deepseek(report)
                    if new_strategy:
                        self._strategy = new_strategy
                        targets = new_strategy.get("targets", [])
                        if targets:
                            self._click._queue = [(t["r"], t["c"]) for t in sorted(
                                targets, key=lambda x: x.get("priority", 99)
                            )]
                self._click_targets = self._click.plan(snap, self.sensor)
                cp = self._click.next()
                if cp:
                    self._last_click = cp
                    self._last_aidx = 5
                    a = GameAction.ACTION6
                    a.coordinate = cp
                    a.reasoning = "exec click replan"
                    return a
                a = GameAction.ACTION6
                a.coordinate = (32, 32)
                a.reasoning = "fallback click center"
                return a

        # --- Keyboard game execution ---
        if self.action_queue:
            aidx = self.action_queue.pop(0)
        else:
            mvs = self._model.get_mv_actions()
            aidx = list(mvs.keys())[0] if mvs else 0

        self._last_aidx = aidx
        a = ALL_ACTIONS[aidx] if aidx < len(ALL_ACTIONS) else GameAction.ACTION1
        a.reasoning = (
            f"exec q={len(self.action_queue)} act={self._actions} ctrl={self._ctrl_pos}"
        )
        return a
