"""
CrossResonanceAgent v12 - Shape-Guided 6-Probe

1. OVERVIEW (0 cost): ShapeEye scans grid, finds interesting shapes
2. PLAN (0 cost): Simulator finds shortest route to most interesting shape
3. PROBE (6 API actions): Execute route, record ALL changes
4. THINK (0 cost): Build rules from observed changes
5. EXECUTE (API): Run solution
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from arcengine import FrameData, GameAction, GameState
from .agent import Agent


class CrossResonanceAgent(Agent):
    MAX_ACTIONS = 500

    def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags)
        self._reset_state()

    def _reset_state(self):
        self.phase = 'overview'
        self.prev_grid = None
        self.initial_grid = None
        self.player_pos = None
        self.player_shape = []
        
        self.action_queue: List[int] = []
        self.probe_observations: List[Dict] = []
        self.probe_count = 0
        self.max_probes = 6
        
        self.solution: List[int] = []
        self.sol_index = 0
        self._prev_levels = 0

    def _detect_player(self, grid):
        g = np.array(grid)
        c12 = np.where(g == 12)
        if len(c12[0]) == 0:
            return None
        r_min = int(c12[0].min())
        r_max = int(c12[0].max())
        c_min = int(c12[1].min())
        c_max = int(c12[1].max())
        center = (r_min + 2, (c_min + c_max) // 2)
        self.player_shape = [(r - center[0], c - center[1])
                             for r in range(r_min, r_min + 5)
                             for c in range(c_min, c_max + 1)]
        return center

    def _find_interesting_targets(self, grid):
        """Find shapes/patterns worth investigating, ranked by interest."""
        g = np.array(grid)
        
        # Count colors to find what's "normal" vs "special"
        from collections import Counter
        color_counts = Counter(int(v) for v in g[:60].flatten())
        total = sum(color_counts.values())
        
        # Rare colors = interesting (less than 1% of grid)
        rare_colors = {c for c, n in color_counts.items() if n < total * 0.01}
        
        # Find clusters of rare-colored cells
        targets = []
        
        # Group rare cells by proximity
        rare_cells = {}
        for r in range(60):
            for c in range(64):
                v = int(g[r, c])
                if v in rare_colors:
                    if v not in rare_cells:
                        rare_cells[v] = []
                    rare_cells[v].append((r, c))
        
        for color, cells in rare_cells.items():
            if cells:
                cr = sum(r for r, c in cells) // len(cells)
                cc = sum(c for r, c in cells) // len(cells)
                # Interest = rarity * cluster size
                interest = (1.0 / max(color_counts[color], 1)) * len(cells) * 1000
                targets.append({
                    'center': (cr, cc),
                    'color': color,
                    'cells': cells,
                    'n_cells': len(cells),
                    'interest': interest,
                    'type': 'rare_cluster'
                })
        
        # Also find pattern structures: enclosed regions, borders
        # Color 5 bordered regions with non-5 content = lock-like structures
        # Look for 9-cells that are enclosed by 5-cells (lock pattern)
        lock_9 = [(r, c) for r in range(25) for c in range(64) if g[r, c] == 9]
        if lock_9:
            cr = sum(r for r, c in lock_9) // len(lock_9)
            cc = sum(c for r, c in lock_9) // len(lock_9)
            targets.append({
                'center': (cr, cc),
                'color': 9,
                'cells': lock_9,
                'n_cells': len(lock_9),
                'interest': 50,  # lock is important but not the first priority
                'type': 'lock_pattern'
            })
        
        # Sort by interest (most interesting first)
        targets.sort(key=lambda t: -t['interest'])
        return targets

    def _plan_route_to(self, grid, target_pos):
        """Use simulator to find shortest route from player to target."""
        from .cross_engine.simulator import CrossWorld
        
        class OpenWorld(CrossWorld):
            def can_move_to(self, center_r, center_c):
                for dr, dc in self.player_shape:
                    nr, nc = center_r + dr, center_c + dc
                    if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                        return False
                    if self.raw[nr][nc] == 4:
                        return False
                return True
        
        world = OpenWorld(grid)
        reachable = world.find_all_reachable()
        
        # Find closest reachable position to target
        best_pos = None
        best_dist = float('inf')
        for pos in reachable:
            d = abs(pos[0] - target_pos[0]) + abs(pos[1] - target_pos[1])
            if d < best_dist:
                best_dist = d
                best_pos = pos
        
        if best_pos and best_pos in reachable:
            return reachable[best_pos], best_pos, best_dist
        return [], None, float('inf')

    def _overview_and_plan(self, grid):
        """Phase 0+1: Find interesting shapes, plan 6-step route to investigate."""
        self.player_pos = self._detect_player(grid)
        if not self.player_pos:
            self.action_queue = [1, 2, 3, 4, 1, 2]
            return
        
        targets = self._find_interesting_targets(grid)
        
        if not targets:
            self.action_queue = [1, 2, 3, 4, 1, 2]
            return
        
        # Plan: go to the most interesting target within 6 steps
        best_route = None
        best_target = None
        
        for target in targets:
            route, dest, dist = self._plan_route_to(grid, target['center'])
            if route and len(route) <= self.max_probes:
                best_route = route
                best_target = target
                break
            elif route and len(route) <= self.max_probes + 2:
                # Close enough — take first 6 steps
                best_route = route[:self.max_probes]
                best_target = target
                break
        
        if best_route:
            self.action_queue = list(best_route)
        else:
            # Can't reach any target in 6 steps. Take the route that gets
            # closest to the most interesting target.
            for target in targets:
                route, dest, dist = self._plan_route_to(grid, target['center'])
                if route:
                    self.action_queue = list(route[:self.max_probes])
                    best_target = target
                    break
        
        if not self.action_queue:
            self.action_queue = [1, 2, 3, 4, 1, 2]
        
        self.phase = 'probe'

    def _diff(self, old, new):
        og = np.array(old)
        ng = np.array(new)
        result = {'player_moved': False, 'new_pos': None, 'structural': [], 
                  'entered_colors': set(), 'delta': (0,0)}
        
        appeared = []
        structural = []
        for r in range(min(og.shape[0], 62)):
            for c in range(og.shape[1]):
                if og[r,c] != ng[r,c]:
                    ov, nv = int(og[r,c]), int(ng[r,c])
                    if nv in (9,12) and ov not in (9,12):
                        appeared.append((r, c, ov))
                    elif ov not in (9,12) and nv not in (9,12):
                        structural.append((r, c, ov, nv))
        
        if appeared:
            cr = sum(r for r,c,_ in appeared) // len(appeared)
            cc = sum(c for r,c,_ in appeared) // len(appeared)
            result['player_moved'] = True
            result['new_pos'] = (cr, cc)
            result['entered_colors'] = set(ov for _,_,ov in appeared)
            if self.player_pos:
                result['delta'] = (cr - self.player_pos[0], cc - self.player_pos[1])
        
        result['structural'] = structural
        return result

    def _think_and_solve(self, grid):
        """After probing, use observations to build solution."""
        from .cross_engine.simulator import CrossWorld
        
        # What did we learn from probes?
        passable = {3}
        for obs in self.probe_observations:
            passable.update(obs.get('entered_colors', set()))
        
        # Check if any structural effects happened (game mechanic clue)
        has_effects = any(len(obs.get('structural', [])) > 0 for obs in self.probe_observations)
        
        class ObsWorld(CrossWorld):
            def can_move_to(self, center_r, center_c):
                for dr, dc in self.player_shape:
                    nr, nc = center_r + dr, center_c + dc
                    if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                        return False
                    if self.raw[nr][nc] == 4:
                        return False
                return True
        
        world = ObsWorld(grid)
        reachable = world.find_all_reachable()
        
        # Visit ALL reachable positions, ordered by proximity to structural effects
        effect_zones = set()
        for obs in self.probe_observations:
            for r, c, ov, nv in obs.get('structural', []):
                if r < 50:
                    effect_zones.add((r, c))
        
        path = []
        current = self.player_pos or world.player_pos
        if not current:
            self.solution = []
            return
        
        # Sort positions: near structural effects first, then near rare colors
        targets_order = sorted(reachable.keys(), key=lambda p: 
            min((abs(p[0]-er) + abs(p[1]-ec) for er, ec in effect_zones), default=100))
        
        visited = {current}
        for pos in targets_order:
            if pos in visited:
                continue
            
            # BFS from current to pos
            queue = deque([(current, [])])
            seen = {current}
            found = False
            while queue:
                p, route = queue.popleft()
                if p == pos:
                    path.extend(route)
                    current = pos
                    visited.add(pos)
                    found = True
                    break
                for a in [1,2,3,4]:
                    nr, nc = world.simulate_move(p[0], p[1], a)
                    if (nr,nc) != p and (nr,nc) not in seen:
                        seen.add((nr,nc))
                        queue.append(((nr,nc), route + [a]))
            
            if not found:
                continue
        
        self.solution = path
        self.sol_index = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        grid = latest_frame.frame[0]
        
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._reset_state()
            return GameAction.RESET
        
        if latest_frame.levels_completed != self._prev_levels:
            self._reset_state()
            self._prev_levels = latest_frame.levels_completed
        
        if self.initial_grid is None:
            self.initial_grid = [row[:] for row in grid]
        
        # Record probe observation
        if self.prev_grid is not None and self.phase == 'probe':
            obs = self._diff(self.prev_grid, grid)
            self.probe_observations.append(obs)
            if obs['new_pos']:
                self.player_pos = obs['new_pos']
            self.probe_count += 1
        elif self.prev_grid is not None and self.phase == 'execute':
            new_pos = self._detect_player(grid)
            if new_pos:
                self.player_pos = new_pos
        
        # Phase routing
        if self.phase == 'overview':
            self._overview_and_plan(grid)
        
        action_id = 1
        
        if self.phase == 'probe':
            if self.action_queue and self.probe_count < self.max_probes:
                action_id = self.action_queue.pop(0)
            else:
                # Probing done → solve
                self._think_and_solve(grid)
                self.phase = 'execute'
                if self.solution:
                    action_id = self.solution[self.sol_index]
                    self.sol_index += 1
        
        elif self.phase == 'execute':
            if self.sol_index < len(self.solution):
                action_id = self.solution[self.sol_index]
                self.sol_index += 1
        
        self.prev_grid = [row[:] for row in grid]
        
        action_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                      3: GameAction.ACTION3, 4: GameAction.ACTION4}
        action = action_map.get(action_id, GameAction.ACTION1)
        action.reasoning = f"{self.phase} probe={self.probe_count} sol={self.sol_index}/{len(self.solution)} pos={self.player_pos}"
        return action
