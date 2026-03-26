"""
CrossResonanceAgent v13 - Empirical Solution

Observed winning sequence from recording 4144fa53:
1. Go to markers (rare colors 0/1)
2. Go to lock wall
3. Push into lock wall repeatedly (3-4 times) 
4. Wall opens, player enters lock → level clear

Strategy:
1. Scan grid for rare colors (markers) and lock
2. Route to markers first
3. Route from markers to lock
4. Push into lock wall until it opens
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque
from arcengine import FrameData, GameAction, GameState
from .agent import Agent


class CrossResonanceAgent(Agent):
    MAX_ACTIONS = 500

    def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags)
        self._full_reset()
    
    def _full_reset(self):
        self.phase = 'plan'
        self.prev_grid = None
        self.player_pos = None
        self.player_shape = []
        self.action_queue: List[int] = []
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

    def _build_plan(self, grid):
        """Build complete plan: markers → lock → push."""
        from .cross_engine.simulator import CrossWorld
        g = np.array(grid)
        
        # Find player
        self.player_pos = self._detect_player(grid)
        if not self.player_pos:
            return
        
        # Find markers (rare colors 0, 1)
        markers = [(int(r), int(c)) for r in range(60) for c in range(64) if g[r,c] in (0, 1)]
        
        # Find lock (color 8 cells or color 9 in upper area enclosed by 5)
        c8 = np.where(g == 8)
        if len(c8[0]) > 0:
            lock_pos = (int(np.mean(c8[0])), int(np.mean(c8[1])))
        else:
            c9_upper = [(int(r),int(c)) for r in range(25) for c in range(64) if g[r,c] == 9]
            if c9_upper:
                lock_pos = (sum(r for r,c in c9_upper)//len(c9_upper), 
                           sum(c for r,c in c9_upper)//len(c9_upper))
            else:
                lock_pos = None
        
        # Build world (only color 4 blocks)
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
        
        plan = []
        
        # Step 1: Route to markers
        if markers:
            marker_center = (sum(r for r,c in markers)//len(markers),
                           sum(c for r,c in markers)//len(markers))
            
            # Find reachable position that overlaps with markers
            best_pos = None
            best_path = None
            best_overlap = 0
            
            for pos, path in reachable.items():
                overlap = sum(1 for dr, dc in world.player_shape 
                            if (pos[0]+dr, pos[1]+dc) in markers)
                if overlap > best_overlap or (overlap == best_overlap and 
                    (best_path is None or len(path) < len(best_path))):
                    best_overlap = overlap
                    best_pos = pos
                    best_path = path
            
            if best_path:
                plan.extend(best_path)
        
        # Step 2: Route from marker position toward lock
        if lock_pos and plan:
            # Simulate position after reaching markers
            current = self.player_pos
            for action in plan:
                nr, nc = world.simulate_move(current[0], current[1], action)
                current = (nr, nc)
            
            # BFS from current to closest-to-lock
            target_r = lock_pos[0] + 5  # aim for just below lock (will push in)
            target_c = lock_pos[1]
            
            queue = deque([(current, [])])
            visited = {current}
            best_lock_path = None
            best_lock_dist = float('inf')
            
            while queue:
                pos, path = queue.popleft()
                dist = abs(pos[0] - target_r) + abs(pos[1] - target_c)
                if dist < best_lock_dist:
                    best_lock_dist = dist
                    best_lock_path = path
                if dist == 0:
                    break
                for a in [1,2,3,4]:
                    nr, nc = world.simulate_move(pos[0], pos[1], a)
                    if (nr,nc) != pos and (nr,nc) not in visited:
                        visited.add((nr,nc))
                        queue.append(((nr,nc), path + [a]))
            
            if best_lock_path:
                plan.extend(best_lock_path)
        
        # Step 3: Push into lock (UP repeatedly)
        # The lock is above the player, so push UP 10 times
        plan.extend([1] * 10)  # UP UP UP... until wall opens
        
        self.action_queue = plan
        self.phase = 'execute'

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
        
        # Track player
        new_pos = self._detect_player(grid)
        if new_pos:
            self.player_pos = new_pos
        
        if self.phase == 'plan':
            self._build_plan(grid)
        
        action_id = 1
        if self.action_queue:
            action_id = self.action_queue.pop(0)
        
        self.prev_grid = [row[:] for row in grid]
        
        action_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                      3: GameAction.ACTION3, 4: GameAction.ACTION4}
        action = action_map.get(action_id, GameAction.ACTION1)
        action.reasoning = f"q={len(self.action_queue)} pos={self.player_pos} lvl={latest_frame.levels_completed}"
        return action
