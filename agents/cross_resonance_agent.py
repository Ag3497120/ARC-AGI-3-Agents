"""
CrossResonanceAgent v15 - Unified Marker→Lock Path

For each level:
1. Find player, markers (color 0/1), ALL color-5 enclosed regions
2. Lock = the color-5 region closest to markers (not key template)
3. Compute unified path: player → marker → lock → push
4. Execute
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from collections import deque
from arcengine import FrameData, GameAction, GameState
from .agent import Agent


class CrossResonanceAgent(Agent):
    MAX_ACTIONS = 500

    def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags)
        self._full_reset()

    def _full_reset(self):
        self.prev_grid = None
        self.player_pos = None
        self.player_shape = []
        self.action_queue: List[int] = []
        self._prev_levels = 0
        self._planned = False

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

    def _can_move(self, grid, cr, cc):
        g = grid if isinstance(grid, np.ndarray) else np.array(grid)
        for dr, dc in self.player_shape:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= 64 or nc < 0 or nc >= 64:
                return False
            if g[nr, nc] == 4:
                return False
        return True

    def _sim_move(self, grid, cr, cc, action):
        deltas = {1: (-5,0), 2: (5,0), 3: (0,-5), 4: (0,5)}
        dr, dc = deltas[action]
        nr, nc = cr + dr, cc + dc
        if self._can_move(grid, nr, nc):
            return (nr, nc)
        return (cr, cc)

    def _bfs(self, grid, start):
        """Full BFS from start. Returns {pos: path}."""
        reachable = {start: []}
        queue = deque([(start, [])])
        while queue:
            pos, path = queue.popleft()
            for a in [1, 2, 3, 4]:
                np_ = self._sim_move(grid, pos[0], pos[1], a)
                if np_ != pos and np_ not in reachable:
                    reachable[np_] = path + [a]
                    queue.append((np_, path + [a]))
        return reachable

    def _find_markers(self, grid):
        """Find color 0/1 cells (markers)."""
        g = np.array(grid)
        return set((int(r), int(c)) for r in range(60) for c in range(64) if g[r,c] in (0, 1))

    def _find_lock(self, grid, markers, player_pos):
        """Find the lock: color 9 enclosed by color 5, CLOSEST to markers."""
        g = np.array(grid)
        
        # Find all color 9 cells enclosed by color 5
        player_cells = set()
        if player_pos:
            for dr, dc in self.player_shape:
                player_cells.add((player_pos[0]+dr, player_pos[1]+dc))
        
        enclosed_9 = []
        for r in range(60):
            for c in range(64):
                if g[r,c] == 9 and (r,c) not in player_cells:
                    # Has color 5 neighbor?
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < 64 and 0 <= nc < 64 and g[nr,nc] == 5:
                            enclosed_9.append((r, c))
                            break
        
        if not enclosed_9:
            return None
        
        # Cluster enclosed_9 cells into groups (connected components)
        remaining = set(enclosed_9)
        clusters = []
        while remaining:
            seed = next(iter(remaining))
            cluster = set()
            stack = [seed]
            while stack:
                cell = stack.pop()
                if cell in remaining:
                    remaining.discard(cell)
                    cluster.add(cell)
                    r, c = cell
                    for dr in range(-3, 4):
                        for dc in range(-3, 4):
                            nb = (r+dr, c+dc)
                            if nb in remaining:
                                stack.append(nb)
            clusters.append(cluster)
        
        if not clusters:
            return None
        
        # Pick cluster CLOSEST to markers (this is the lock, not key template)
        if markers:
            marker_center = (sum(r for r,c in markers) // len(markers),
                           sum(c for r,c in markers) // len(markers))
            
            def cluster_dist(cluster):
                cr = sum(r for r,c in cluster) // len(cluster)
                cc = sum(c for r,c in cluster) // len(cluster)
                return abs(cr - marker_center[0]) + abs(cc - marker_center[1])
            
            clusters.sort(key=cluster_dist)
        
        best = clusters[0]
        return (sum(r for r,c in best) // len(best),
                sum(c for r,c in best) // len(best))

    def _plan(self, grid):
        """Unified path: player → marker → lock → push."""
        g = np.array(grid)
        self.player_pos = self._detect_player(grid)
        if not self.player_pos:
            self.action_queue = [1,2,3,4] * 5
            self._planned = True
            return

        markers = self._find_markers(grid)
        lock_center = self._find_lock(grid, markers, self.player_pos)
        
        reachable = self._bfs(g, self.player_pos)
        
        path = []
        current = self.player_pos

        # Step 1: Go to markers (position that overlaps marker cells)
        if markers:
            best_pos = None
            best_path = None
            best_overlap = 0
            
            for pos, p in reachable.items():
                overlap = sum(1 for dr, dc in self.player_shape
                            if (pos[0]+dr, pos[1]+dc) in markers)
                if overlap > best_overlap or (overlap == best_overlap and
                    (best_path is None or len(p) < len(best_path))):
                    best_overlap = overlap
                    best_pos = pos
                    best_path = p
            
            if best_path and best_overlap > 0:
                path.extend(best_path)
                current = best_pos

        # Step 2: Go toward lock
        if lock_center:
            # BFS from current position to near lock
            queue = deque([(current, [])])
            visited = {current}
            lock_path = []
            
            while queue:
                pos, p = queue.popleft()
                if abs(pos[0] - lock_center[0]) <= 7 and abs(pos[1] - lock_center[1]) <= 5:
                    lock_path = p
                    break
                for a in [1,2,3,4]:
                    np_ = self._sim_move(g, pos[0], pos[1], a)
                    if np_ != pos and np_ not in visited:
                        visited.add(np_)
                        queue.append((np_, p + [a]))
            
            path.extend(lock_path)
            
            # Simulate where we end up
            for a in lock_path:
                current = self._sim_move(g, current[0], current[1], a)
            
            # Step 3: Push toward lock center
            dr = lock_center[0] - current[0]
            dc = lock_center[1] - current[1]
            
            # Try primary direction first, then secondary
            pushes = []
            if abs(dr) >= abs(dc):
                pushes.append(1 if dr < 0 else 2)
                if dc != 0:
                    pushes.append(3 if dc < 0 else 4)
            else:
                pushes.append(3 if dc < 0 else 4)
                if dr != 0:
                    pushes.append(1 if dr < 0 else 2)
            
            # Push in primary direction, then alternate
            for push_dir in pushes:
                path.extend([push_dir] * 5)

        self.action_queue = path
        self._planned = True

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

        if not self._planned:
            self._plan(grid)

        new_pos = self._detect_player(grid)
        if new_pos:
            self.player_pos = new_pos

        action_id = 1
        if self.action_queue:
            action_id = self.action_queue.pop(0)

        self.prev_grid = [row[:] for row in grid]

        action_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                      3: GameAction.ACTION3, 4: GameAction.ACTION4}
        action = action_map.get(action_id, GameAction.ACTION1)
        action.reasoning = f"q={len(self.action_queue)} pos={self.player_pos} lvl={latest_frame.levels_completed}"
        return action
