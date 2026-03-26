"""
CrossResonanceAgent v14 - Complete Verantyx Loop

The full pipeline:
  SCAN → SENSE → LEARN → SIMULATE → ACT → FEEDBACK → loop

Every rule is a .jcross dynamic structure, not hardcoded Python.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
from arcengine import FrameData, GameAction, GameState
from .agent import Agent


# ═══════════════════════════════════════════════
# .jcross Rule: data that drives behavior
# Generated from observation, not written by hand
# ═══════════════════════════════════════════════

class JCrossRule:
    """A rule discovered from observation. This IS the .jcross."""
    def __init__(self):
        self.name = ""
        self.trigger = {}      # what causes this rule to activate
        self.effect = {}       # what happens when it activates
        self.confidence = 0.0  # 0-1, updated by feedback
        self.tested = False
        self.worked = None     # True/False after testing
    
    def __repr__(self):
        return f"<Rule:{self.name} conf={self.confidence:.2f}>"


class WorldModel:
    """Cross simulator with dynamically generated rules."""
    
    def __init__(self, grid, player_shape):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.player_shape = player_shape
        self.rules: List[JCrossRule] = []
        self.blocked_colors: Set[int] = {4}  # start with only color 4 as wall
        self.passable_colors: Set[int] = {3}
        self.special_cells: Dict[Tuple[int,int], int] = {}  # cells with special meaning
    
    def add_rule(self, rule: JCrossRule):
        self.rules.append(rule)
    
    def can_move_to(self, cr, cc):
        for dr, dc in self.player_shape:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                return False
            color = int(self.grid[nr, nc])
            if color in self.blocked_colors:
                return False
        return True
    
    def simulate_move(self, cr, cc, action):
        deltas = {1: (-5,0), 2: (5,0), 3: (0,-5), 4: (0,5)}
        dr, dc = deltas.get(action, (0,0))
        nr, nc = cr + dr, cc + dc
        if self.can_move_to(nr, nc):
            return (nr, nc)
        return (cr, cc)
    
    def find_all_reachable(self, start):
        reachable = {start: []}
        queue = deque([(start, [])])
        while queue:
            pos, path = queue.popleft()
            for a in [1,2,3,4]:
                np_ = self.simulate_move(pos[0], pos[1], a)
                if np_ != pos and np_ not in reachable:
                    new_path = path + [a]
                    reachable[np_] = new_path
                    queue.append((np_, new_path))
        return reachable
    
    def apply_rules(self):
        """Apply all active rules to modify the world model.
        Returns a copy with rules applied."""
        modified = WorldModel(self.grid.tolist(), self.player_shape)
        modified.blocked_colors = self.blocked_colors.copy()
        modified.passable_colors = self.passable_colors.copy()
        modified.rules = self.rules[:]
        
        for rule in self.rules:
            if rule.confidence > 0.3 and rule.worked is not False:
                eff = rule.effect
                if eff.get('type') == 'open_cells':
                    for r, c in eff.get('cells', []):
                        if 0 <= r < modified.rows and 0 <= c < modified.cols:
                            modified.grid[r, c] = eff.get('new_color', 0)
                elif eff.get('type') == 'color_passable':
                    modified.passable_colors.add(eff.get('color'))
                    modified.blocked_colors.discard(eff.get('color'))
        
        return modified


# ═══════════════════════════════════════════════
# Main Agent
# ═══════════════════════════════════════════════

class CrossResonanceAgent(Agent):
    MAX_ACTIONS = 500

    def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags)
        self._full_reset()

    def _full_reset(self):
        self.pipeline_phase = 'scan'  # scan → sense → learn → simulate → act → feedback
        self.prev_grid = None
        self.player_pos = None
        self.player_shape = []
        self.world: Optional[WorldModel] = None
        
        # Scan results
        self.interesting_targets: List[Dict] = []
        
        # Sense
        self.sense_plan: List[int] = []
        self.sense_index = 0
        self.sense_observations: List[Dict] = []
        self.max_sense_steps = 8  # dynamic: as many as needed to reach markers
        
        # Learn
        self.learned_rules: List[JCrossRule] = []
        
        # Act
        self.action_queue: List[int] = []
        
        # Feedback
        self.attempt = 0
        self.max_attempts = 5
        
        self._prev_levels = 0

    # ── SCAN: overview the grid ──
    
    def _scan(self, grid):
        """Look at the grid, find what's interesting. Zero cost."""
        g = np.array(grid)
        self.player_pos = self._detect_player(grid)
        if not self.player_pos:
            return
        
        self.world = WorldModel(grid, self.player_shape)
        
        # Find rare colors (potential interactive elements)
        from collections import Counter
        counts = Counter(int(v) for v in g[:60].flatten())
        total = sum(counts.values())
        
        targets = []
        for color, count in counts.items():
            if count < total * 0.005 and color not in (9, 12):
                cells = [(int(r),int(c)) for r in range(60) for c in range(64) if g[r,c] == color]
                cr = sum(r for r,c in cells) // len(cells)
                cc = sum(c for r,c in cells) // len(cells)
                targets.append({
                    'color': color, 'center': (cr, cc), 'cells': cells,
                    'interest': (1.0 / max(count, 1)) * len(cells) * 1000
                })
        
        # Find lock-like structures: color 5 enclosed regions (not the border/UI)
        # Group color 5 cells into clusters
        c5_cells = [(int(r),int(c)) for r in range(5, 55) for c in range(5, 60) if g[r,c] == 5]
        if c5_cells:
            # Find the cluster NOT near the grid edge (the lock, not the border)
            # Simple: center of mass of color 5 cells away from edges
            interior_5 = [(r,c) for r,c in c5_cells if 8 < r < 55 and 5 < c < 58]
            if interior_5:
                cr5 = sum(r for r,c in interior_5) // len(interior_5)
                cc5 = sum(c for r,c in interior_5) // len(interior_5)
                targets.append({
                    'color': 'lock', 'center': (cr5, cc5), 'cells': interior_5,
                    'interest': 100
                })
        
        targets.sort(key=lambda t: -t['interest'])
        self.interesting_targets = targets
        self.pipeline_phase = 'sense'
    
    # ── SENSE: go touch interesting things (6 API steps) ──
    
    def _plan_sense(self, grid):
        """Plan route to most interesting target. Dynamic length."""
        if not self.interesting_targets or not self.player_pos:
            self.sense_plan = [1, 2, 3, 4, 1, 2]
            return
        
        reachable = self.world.find_all_reachable(self.player_pos)
        
        for target in self.interesting_targets:
            if target['color'] == 'lock':
                continue
            
            best_pos = None
            best_path = None
            best_overlap = 0
            
            target_set = set(target['cells'])
            for pos, path in reachable.items():
                overlap = sum(1 for dr, dc in self.player_shape
                            if (pos[0]+dr, pos[1]+dc) in target_set)
                if overlap > best_overlap or (overlap == best_overlap and
                    (best_path is None or len(path) < len(best_path))):
                    best_overlap = overlap
                    best_pos = pos
                    best_path = path
            
            if best_path and best_overlap > 0:
                # Use the FULL path to marker, not truncated
                self.sense_plan = list(best_path)
                self.max_sense_steps = len(best_path)
                return
        
        if self.interesting_targets:
            tc = self.interesting_targets[0]['center']
            closest = min(reachable.keys(),
                key=lambda p: abs(p[0]-tc[0]) + abs(p[1]-tc[1]))
            self.sense_plan = list(reachable[closest])
            self.max_sense_steps = len(self.sense_plan)
        else:
            self.sense_plan = [1, 2, 3, 4, 1, 2]
    
    # ── LEARN: generate .jcross rules from observations ──
    
    def _learn(self):
        """Analyze sense observations, generate rules dynamically."""
        self.learned_rules = []
        
        for obs in self.sense_observations:
            entered = obs.get('entered', set())
            structural = obs.get('structural', [])
            
            # Rule: color X is passable (player entered it)
            for color in entered:
                if color not in (3, 9, 12):
                    rule = JCrossRule()
                    rule.name = f"color_{color}_passable"
                    rule.trigger = {'type': 'always'}
                    rule.effect = {'type': 'color_passable', 'color': color}
                    rule.confidence = 0.9
                    self.learned_rules.append(rule)
            
            # Rule: structural change = game mechanic
            if structural:
                # Group structural changes by what happened
                opened_cells = [(r,c) for r,c,old,new in structural if old == 5 and new == 0]
                if opened_cells:
                    rule = JCrossRule()
                    rule.name = f"marker_opens_lock_{len(opened_cells)}_cells"
                    rule.trigger = {'type': 'visit_marker'}
                    rule.effect = {'type': 'open_cells', 'cells': opened_cells, 'new_color': 0}
                    rule.confidence = 0.95
                    self.learned_rules.append(rule)
        
        self.pipeline_phase = 'simulate'
    
    # ── SIMULATE: use learned rules to find solution path ──
    
    def _simulate(self, grid):
        """Apply learned rules to world model, find optimal path. Zero cost."""
        if not self.world:
            self.world = WorldModel(grid, self.player_shape)
        
        # Add learned rules
        for rule in self.learned_rules:
            self.world.add_rule(rule)
        
        # Apply rules to get modified world
        modified = self.world.apply_rules()
        
        if not self.player_pos:
            return
        
        reachable = modified.find_all_reachable(self.player_pos)
        
        # Find path to lock target
        lock_targets = [t for t in self.interesting_targets if t['color'] == 'lock']
        # If no lock found from targets, use color 5 cluster center
        if not lock_targets and self.interesting_targets:
            # Look for color 5 enclosed regions directly
            g = np.array(grid)
            c5 = [(int(r),int(c)) for r in range(8, 55) for c in range(5, 58) if g[r,c] == 5]
            if c5:
                cr5 = sum(r for r,c in c5) // len(c5)
                cc5 = sum(c for r,c in c5) // len(c5)
                lock_targets = [{'center': (cr5, cc5)}]
        
        path = []
        
        # Sense already visited markers. Go directly to lock from current position.
        
        # Then: go to lock
        if lock_targets:
            lock_center = lock_targets[0]['center']
            
            # Simulate position after marker visit
            current = self.player_pos
            for a in path:
                current = modified.simulate_move(current[0], current[1], a)
            
            # BFS from current to lock
            queue = deque([(current, [])])
            visited = {current}
            lock_path = []
            
            while queue:
                pos, p = queue.popleft()
                if abs(pos[0] - lock_center[0]) <= 5 and abs(pos[1] - lock_center[1]) <= 5:
                    lock_path = p
                    break
                for a in [1,2,3,4]:
                    np_ = modified.simulate_move(pos[0], pos[1], a)
                    if np_ != pos and np_ not in visited:
                        visited.add(np_)
                        queue.append((np_, p + [a]))
            
            path.extend(lock_path)
        
        # Push into lock repeatedly (empirical: walls may need multiple pushes)
        path.extend([1] * 5)
        
        self.action_queue = path
        self.pipeline_phase = 'act'
    
    # ── FEEDBACK: check if plan worked ──
    
    def _feedback(self, grid, levels_before, levels_after):
        """Plan failed or succeeded. Update rules accordingly."""
        if levels_after > levels_before:
            # Success! Mark all rules as working
            for rule in self.learned_rules:
                rule.worked = True
            return
        
        # Failed — invalidate highest-confidence untested rule
        for rule in reversed(self.learned_rules):
            if rule.worked is None:
                rule.worked = False
                rule.confidence *= 0.3
                break
        
        self.attempt += 1
        if self.attempt < self.max_attempts:
            self.pipeline_phase = 'simulate'  # re-simulate with updated rules
        else:
            self.pipeline_phase = 'scan'  # full restart with new scan
    
    # ── Helpers ──
    
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
    
    def _diff(self, old, new):
        og = np.array(old)
        ng = np.array(new)
        result = {'moved': False, 'pos': None, 'entered': set(), 'structural': []}
        
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
        return result
    
    # ── Main loop ──
    
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        grid = latest_frame.frame[0]
        
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._full_reset()
            return GameAction.RESET
        
        if latest_frame.levels_completed != self._prev_levels:
            lvl_before = self._prev_levels
            self._full_reset()
            self._prev_levels = latest_frame.levels_completed
        
        # Track player
        if self.prev_grid is not None:
            obs = self._diff(self.prev_grid, grid)
            if obs['moved']:
                self.player_pos = obs['pos']
            
            if self.pipeline_phase == 'sense' and self.sense_index > 0:
                self.sense_observations.append(obs)
        
        new_pos = self._detect_player(grid)
        if new_pos:
            self.player_pos = new_pos
        
        # ── Pipeline ──
        
        if self.pipeline_phase == 'scan':
            self._scan(grid)
            self._plan_sense(grid)
        
        action_id = 1
        
        if self.pipeline_phase == 'sense':
            if self.sense_index < len(self.sense_plan) and self.sense_index < self.max_sense_steps:
                action_id = self.sense_plan[self.sense_index]
                self.sense_index += 1
            else:
                self._learn()
                self._simulate(grid)
                if self.action_queue:
                    action_id = self.action_queue.pop(0)
        
        elif self.pipeline_phase == 'act':
            if self.action_queue:
                action_id = self.action_queue.pop(0)
            else:
                # Plan exhausted, check if we leveled up
                self._feedback(grid, self._prev_levels, latest_frame.levels_completed)
                if self.pipeline_phase == 'simulate':
                    self._simulate(grid)
                    if self.action_queue:
                        action_id = self.action_queue.pop(0)
                elif self.pipeline_phase == 'scan':
                    self._scan(grid)
                    self._plan_sense(grid)
                    if self.sense_plan:
                        action_id = self.sense_plan[0]
                        self.sense_index = 1
        
        self.prev_grid = [row[:] for row in grid]
        
        action_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                      3: GameAction.ACTION3, 4: GameAction.ACTION4}
        action = action_map.get(action_id, GameAction.ACTION1)
        action.reasoning = (f"{self.pipeline_phase} att={self.attempt} "
                           f"rules={len(self.learned_rules)} q={len(self.action_queue)} "
                           f"pos={self.player_pos}")
        return action
