"""
CrossResonanceAgent v17 - Dynamic Probe Count from Shape Analysis

No fixed probe count. Shape Index determines:
1. How many shapes exist, their distances, timer budget
2. Optimal probe count = min(marker_path + lock_path + push, timer_budget)
3. If total path fits in timer → execute all at once
4. If not → split across retries intelligently
"""

import numpy as np
from typing import List, Tuple, Set, Dict
from collections import deque, Counter
from arcengine import FrameData, GameAction, GameState
from .agent import Agent


class Shape:
    __slots__ = ('color','cells','center','bbox','size','role')
    def __init__(self, color, cells):
        self.color = color
        self.cells = cells
        self.size = len(cells)
        rs = [r for r,c in cells]
        cs = [c for r,c in cells]
        self.center = (sum(rs)//len(rs), sum(cs)//len(cs))
        self.bbox = (min(rs), min(cs), max(rs), max(cs))
        self.role = 'unknown'


class ShapeIndex:
    def __init__(self, grid):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.shapes: List[Shape] = []
        self.by_role: Dict[str, List[Shape]] = {}
        self.player_pos = None
        self.player_offsets = []
        self.marker_cells: Set[Tuple[int,int]] = set()
        self.lock_center = None
        self.timer_budget = 20
        
        self._extract()
        self._classify()
        self._relationships()
        self._calc_timer()
    
    def _extract(self):
        visited = set()
        for r in range(min(self.rows, 62)):
            for c in range(self.cols):
                if (r,c) not in visited:
                    color = int(self.grid[r,c])
                    cells = set()
                    stack = [(r,c)]
                    while stack:
                        cr,cc = stack.pop()
                        if (cr,cc) in visited or cr<0 or cr>=min(self.rows,62) or cc<0 or cc>=self.cols:
                            continue
                        if int(self.grid[cr,cc]) != color:
                            continue
                        visited.add((cr,cc))
                        cells.add((cr,cc))
                        stack.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                    if len(cells) >= 2:
                        self.shapes.append(Shape(color, cells))
    
    def _classify(self):
        counts = Counter(int(v) for v in self.grid[:60].flatten())
        total = sum(counts.values())
        
        player_12 = None
        for s in self.shapes:
            if s.color == 12:
                s.role = 'player_top'
                player_12 = s
            elif s.color in (0,1) and counts.get(s.color,0) < total * 0.005:
                s.role = 'marker'
                self.marker_cells |= s.cells
            elif s.color == 9 and s.size <= 30:
                if player_12 and any(abs(r-pr)<=3 and abs(c-pc)<=3 for r,c in s.cells for pr,pc in player_12.cells):
                    s.role = 'player_bottom'
                else:
                    has5 = any(0<=r+dr<self.rows and 0<=c+dc<self.cols and int(self.grid[r+dr,c+dc])==5
                              for r,c in s.cells for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
                    s.role = 'lock_or_template' if has5 else 'other_9'
            elif s.color == 5 and s.size < 100:
                s.role = 'lock_border'
            elif s.color == 3:
                s.role = 'corridor'
            elif s.color == 4:
                s.role = 'wall'
        
        # Build player
        if player_12:
            all_p = player_12.cells.copy()
            for s in self.shapes:
                if s.role == 'player_bottom':
                    all_p |= s.cells
            rs = [r for r,c in all_p]
            cs = [c for r,c in all_p]
            self.player_pos = (sum(rs)//len(rs), sum(cs)//len(cs))
            self.player_offsets = [(r-self.player_pos[0], c-self.player_pos[1]) for r,c in all_p]
        
        self.by_role = {}
        for s in self.shapes:
            self.by_role.setdefault(s.role, []).append(s)
    
    def _relationships(self):
        markers = self.by_role.get('marker', [])
        lt = self.by_role.get('lock_or_template', [])
        if markers and lt:
            mc = markers[0].center
            lt.sort(key=lambda s: abs(s.center[0]-mc[0])+abs(s.center[1]-mc[1]))
            lt[0].role = 'lock_pattern'
            self.lock_center = lt[0].center
            for s in lt[1:]:
                s.role = 'key_template'
        elif lt:
            # No markers? Use largest lock_or_template
            lt.sort(key=lambda s: -s.size)
            lt[0].role = 'lock_pattern'
            self.lock_center = lt[0].center
        
        if not self.lock_center:
            borders = self.by_role.get('lock_border', [])
            if borders and markers:
                mc = markers[0].center
                borders.sort(key=lambda s: abs(s.center[0]-mc[0])+abs(s.center[1]-mc[1]))
                self.lock_center = borders[0].center
    
    def _calc_timer(self):
        timer_cells = sum(1 for r in range(60,64) for c in range(self.cols) if self.grid[r,c] == 11)
        self.timer_budget = max(timer_cells // 4 - 1, 8)


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
        self._skip_frame = False
        self._retry_count = 0
        self._best_strategy = None

    def _can_move(self, g, cr, cc):
        for dr, dc in self.player_shape:
            nr, nc = cr+dr, cc+dc
            if nr<0 or nr>=64 or nc<0 or nc>=64: return False
            if g[nr,nc] == 4: return False
        return True

    def _sim(self, g, cr, cc, a):
        deltas = {1:(-5,0),2:(5,0),3:(0,-5),4:(0,5)}
        dr,dc = deltas[a]
        nr,nc = cr+dr, cc+dc
        return (nr,nc) if self._can_move(g, nr, nc) else (cr,cc)

    def _bfs_to(self, g, start, goal_fn):
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            if goal_fn(pos):
                return path, pos
            for a in [1,2,3,4]:
                np_ = self._sim(g, pos[0], pos[1], a)
                if np_ != pos and np_ not in visited:
                    visited.add(np_)
                    queue.append((np_, path + [a]))
        return [], start

    def _push_dir(self, from_pos, to_pos):
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        dirs = []
        if abs(dr) >= abs(dc):
            dirs.append(1 if dr < 0 else 2)
            if dc != 0: dirs.append(3 if dc < 0 else 4)
        else:
            dirs.append(3 if dc < 0 else 4)
            if dr != 0: dirs.append(1 if dr < 0 else 2)
        push = []
        for d in dirs:
            push.extend([d] * 4)
        return push

    def _plan(self, grid):
        g = np.array(grid)
        idx = ShapeIndex(grid)
        
        self.player_pos = idx.player_pos
        self.player_shape = idx.player_offsets
        
        if not self.player_pos or not self.player_shape:
            self.action_queue = [1,2,3,4] * 5
            self._planned = True
            return
        
        budget = idx.timer_budget
        
        # Strategy A: marker → lock
        marker_path, marker_end = [], self.player_pos
        if idx.marker_cells:
            def overlaps(pos):
                return any((pos[0]+dr,pos[1]+dc) in idx.marker_cells for dr,dc in self.player_shape)
            marker_path, marker_end = self._bfs_to(g, self.player_pos, overlaps)
        
        lock_from_marker, lock_end_a = [], marker_end
        if idx.lock_center:
            def near_lock(pos):
                return abs(pos[0]-idx.lock_center[0])<=7 and abs(pos[1]-idx.lock_center[1])<=5
            lock_from_marker, lock_end_a = self._bfs_to(g, marker_end, near_lock)
        
        push_a = self._push_dir(lock_end_a, idx.lock_center) if idx.lock_center else [1]*4
        strat_a = marker_path + lock_from_marker + push_a
        
        # Strategy B: direct to lock
        lock_direct, lock_end_b = [], self.player_pos
        if idx.lock_center:
            def near_lock2(pos):
                return abs(pos[0]-idx.lock_center[0])<=7 and abs(pos[1]-idx.lock_center[1])<=5
            lock_direct, lock_end_b = self._bfs_to(g, self.player_pos, near_lock2)
        
        push_b = self._push_dir(lock_end_b, idx.lock_center) if idx.lock_center else [1]*4
        strat_b = lock_direct + push_b
        
        # Strategy C: marker only (trigger, hope for effect on next retry)
        strat_c = list(marker_path) if marker_path else [1,2,3,4]
        
        # Choose best strategy that fits timer
        if len(strat_a) <= budget:
            self.action_queue = strat_a
            self._best_strategy = 'via_marker'
        elif len(strat_b) <= budget:
            self.action_queue = strat_b
            self._best_strategy = 'direct'
        else:
            # Neither fits. Adaptive: on retry 0, go to marker. On retry 1+, go direct to lock.
            if self._retry_count == 0 and marker_path:
                # Go to marker, then as far toward lock as budget allows
                remaining = budget - len(marker_path)
                if remaining > 0:
                    self.action_queue = marker_path + lock_from_marker[:remaining]
                else:
                    self.action_queue = marker_path[:budget]
                self._best_strategy = 'marker_first'
            else:
                # Go direct to lock (skip marker)
                self.action_queue = strat_b[:budget]
                self._best_strategy = 'direct_truncated'
        
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
            self._skip_frame = True
            return GameAction.ACTION1

        if self._skip_frame:
            self._skip_frame = False

        # Detect timer restart (player position reset)
        g = np.array(grid)
        c12 = np.where(g == 12)
        if len(c12[0]) > 0:
            new_pos = (int(c12[0].min())+2, int((c12[1].min()+c12[1].max())//2))
            if self.prev_grid is not None:
                if self.player_pos and abs(new_pos[0]-self.player_pos[0]) > 10:
                    # Timer restart detected. Immediate replan.
                    self._retry_count += 1
                    self._planned = False
                    self.player_pos = new_pos
                    self._plan(grid)
                    # Return first action immediately
                    if self.action_queue:
                        action_id = self.action_queue.pop(0)
                        self.prev_grid = [row[:] for row in grid]
                        action_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                                      3: GameAction.ACTION3, 4: GameAction.ACTION4}
                        action = action_map.get(action_id, GameAction.ACTION1)
                        action.reasoning = f"q={len(self.action_queue)} retry={self._retry_count}"
                        return action
            if self.player_pos and self._planned and not self.action_queue:
                self._retry_count += 1
                self._planned = False
            self.player_pos = new_pos

        if not self._planned:
            self._plan(grid)

        action_id = 1
        if self.action_queue:
            action_id = self.action_queue.pop(0)

        self.prev_grid = [row[:] for row in grid]

        action_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                      3: GameAction.ACTION3, 4: GameAction.ACTION4}
        action = action_map.get(action_id, GameAction.ACTION1)
        action.reasoning = f"q={len(self.action_queue)} strat={self._best_strategy} retry={self._retry_count} pos={self.player_pos}"
        return action
