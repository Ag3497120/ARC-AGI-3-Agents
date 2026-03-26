"""
CrossResonanceAgent v12 (.jcross dynamic generation)

Core loop:
  1. SCAN frame → build Cross structure (shapes, colors, positions)
  2. GENERATE .jcross rules dynamically from Cross structure
     - "rare color cluster at (X,Y)" → generate route-to-target rule
     - "enclosed pattern" → generate pattern-match rule
     - Rules are DATA, not hardcoded logic
  3. SIMULATE on Cross world using generated rules
  4. EXECUTE 6 probe steps + solution
  5. On failure → add constraint to Cross structure → re-generate rules

The key: rules are generated FROM the grid, not written in advance.
Every new grid generates different rules. This is .jcross dynamic generation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
from arcengine import FrameData, GameAction, GameState
from .agent import Agent


# ═══════════════════════════════════════════════════════════
# .jcross Dynamic Rule: generated from grid, not hardcoded
# ═══════════════════════════════════════════════════════════

class JCrossRule:
    """A dynamically generated rule from grid observation.
    This IS the .jcross — data that drives behavior."""
    def __init__(self, name: str, target: Tuple[int,int], 
                 route: List[int], priority: float, source: str):
        self.name = name
        self.target = target     # where to go
        self.route = route       # how to get there (action sequence)
        self.priority = priority # how important (from shape analysis)
        self.source = source     # why this rule was generated
        self.tested = False
        self.result = None       # what happened when we went there


class CrossResonanceAgent(Agent):
    MAX_ACTIONS = 500

    def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags)
        self._full_reset()

    def _full_reset(self):
        self.phase = 'scan'
        self.prev_grid = None
        self.initial_grid = None
        self.player_pos = None
        self.player_shape = []
        
        # .jcross dynamic rules
        self.rules: List[JCrossRule] = []
        self.current_rule: Optional[JCrossRule] = None
        self.rule_index = 0
        
        # Execution
        self.action_queue: List[int] = []
        self.observations: List[Dict] = []
        
        self._prev_levels = 0

    def _detect_player(self, grid):
        g = np.array(grid)
        c12 = np.where(g == 12)
        if len(c12[0]) == 0:
            return None
        r_min = int(c12[0].min())
        c_min = int(c12[1].min())
        c_max = int(c12[1].max())
        center = (r_min + 2, (c_min + c_max) // 2)
        self.player_shape = [(r - center[0], c - center[1])
                             for r in range(r_min, r_min + 5)
                             for c in range(c_min, c_max + 1)]
        return center

    def _scan_and_generate(self, grid):
        """Phase 1: Scan grid, generate .jcross rules dynamically.
        Every grid produces different rules. This is the core of dynamic generation."""
        g = np.array(grid)
        self.player_pos = self._detect_player(grid)
        if not self.player_pos:
            return
        
        # ── Step 1: Find all shapes/patterns in the grid ──
        from collections import Counter
        color_counts = Counter(int(v) for v in g[:60].flatten())
        total = sum(color_counts.values())
        
        # Rare colors = interesting shapes
        rare_threshold = total * 0.005
        rare_colors = {c for c, n in color_counts.items() 
                       if n < rare_threshold and c not in (9, 12)}  # exclude player colors
        
        # Cluster rare cells
        clusters = {}
        for r in range(60):
            for c in range(64):
                v = int(g[r, c])
                if v in rare_colors:
                    if v not in clusters:
                        clusters[v] = []
                    clusters[v].append((r, c))
        
        # ── Step 2: Build simulator for route planning ──
        from .cross_engine.simulator import CrossWorld
        
        class DynWorld(CrossWorld):
            def can_move_to(self, center_r, center_c):
                for dr, dc in self.player_shape:
                    nr, nc = center_r + dr, center_c + dc
                    if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                        return False
                    if self.raw[nr][nc] == 4:
                        return False
                return True
        
        world = DynWorld(grid)
        reachable = world.find_all_reachable()
        
        # ── Step 3: Generate .jcross rules from shapes ──
        self.rules = []
        
        for color, cells in sorted(clusters.items(), key=lambda x: len(x[1])):
            center_r = sum(r for r, c in cells) // len(cells)
            center_c = sum(c for r, c in cells) // len(cells)
            
            # Find the reachable position that makes player OVERLAP with these cells
            best_pos = None
            best_path = None
            best_overlap = 0
            
            for pos, path in reachable.items():
                # How many rare cells does the player block cover at this position?
                overlap = 0
                for dr, dc in world.player_shape:
                    pr, pc = pos[0] + dr, pos[1] + dc
                    if (pr, pc) in [(r, c) for r, c in cells]:
                        overlap += 1
                
                if overlap > best_overlap or (overlap == best_overlap and 
                    (best_path is None or len(path) < len(best_path))):
                    best_overlap = overlap
                    best_pos = pos
                    best_path = path
            
            if best_pos and best_path and best_overlap > 0:
                rarity = 1.0 / max(color_counts[color], 1)
                self.rules.append(JCrossRule(
                    name=f"visit_color{color}_at_{best_pos}",
                    target=best_pos,
                    route=best_path,
                    priority=rarity * best_overlap * 1000,
                    source=f"rare color {color} ({len(cells)} cells), {best_overlap} overlap"
                ))
            elif best_pos and best_path:
                # No overlap but close
                rarity = 1.0 / max(color_counts[color], 1)
                self.rules.append(JCrossRule(
                    name=f"approach_color{color}_near_{best_pos}",
                    target=best_pos,
                    route=best_path,
                    priority=rarity * 500,
                    source=f"rare color {color}, closest approach"
                ))
        
        # Also generate "visit all reachable" as fallback rule
        for pos, path in sorted(reachable.items(), 
                key=lambda x: len(x[1])):
            if pos != self.player_pos:
                self.rules.append(JCrossRule(
                    name=f"explore_{pos}",
                    target=pos,
                    route=path,
                    priority=1.0,
                    source="systematic exploration"
                ))
        
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: -r.priority)
        
        self.rule_index = 0
        self.phase = 'execute_rules'
    
    def _next_rule(self):
        """Get next untested rule and queue its route."""
        while self.rule_index < len(self.rules):
            rule = self.rules[self.rule_index]
            self.rule_index += 1
            if not rule.tested:
                self.current_rule = rule
                self.action_queue = list(rule.route)
                return True
        return False
    
    def _diff(self, old, new):
        og = np.array(old)
        ng = np.array(new)
        result = {'moved': False, 'pos': None, 'entered': set(), 
                  'structural': [], 'n_struct': 0}
        
        appeared = []
        structural = []
        for r in range(min(og.shape[0], 62)):
            for c in range(og.shape[1]):
                if og[r,c] != ng[r,c]:
                    ov, nv = int(og[r,c]), int(ng[r,c])
                    if nv in (9,12) and ov not in (9,12):
                        appeared.append((r,c,ov))
                    elif ov not in (9,12) and nv not in (9,12):
                        structural.append((r,c,ov,nv))
        
        if appeared:
            result['moved'] = True
            result['pos'] = (sum(r for r,c,_ in appeared)//len(appeared),
                            sum(c for r,c,_ in appeared)//len(appeared))
            result['entered'] = set(ov for _,_,ov in appeared)
        
        result['structural'] = structural
        result['n_struct'] = len(structural)
        return result

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        grid = latest_frame.frame[0]
        
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._full_reset()
            return GameAction.RESET
        
        if latest_frame.levels_completed != self._prev_levels:
            self._full_reset()
            self._prev_levels = latest_frame.levels_completed
        
        if self.initial_grid is None:
            self.initial_grid = [row[:] for row in grid]
        
        # Record observations
        if self.prev_grid is not None:
            obs = self._diff(self.prev_grid, grid)
            self.observations.append(obs)
            if obs['moved']:
                self.player_pos = obs['pos']
            
            # If current rule's route is done, mark it as tested
            if self.current_rule and not self.action_queue:
                self.current_rule.tested = True
                self.current_rule.result = {
                    'structural_effects': sum(o['n_struct'] for o in self.observations[-len(self.current_rule.route):]),
                    'final_pos': self.player_pos,
                }
        
        # Phase: scan grid and generate rules
        if self.phase == 'scan':
            self._scan_and_generate(grid)
            self._next_rule()
        
        # Execute current rule's route
        action_id = 1
        
        if self.phase == 'execute_rules':
            if self.action_queue:
                action_id = self.action_queue.pop(0)
            else:
                # Current rule done. Get next rule.
                if not self._next_rule():
                    action_id = 1  # all rules exhausted
        
        self.prev_grid = [row[:] for row in grid]
        
        action_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                      3: GameAction.ACTION3, 4: GameAction.ACTION4}
        action = action_map.get(action_id, GameAction.ACTION1)
        
        rule_name = self.current_rule.name if self.current_rule else 'none'
        action.reasoning = f"rule={rule_name} queue={len(self.action_queue)} pos={self.player_pos}"
        return action
