"""
CrossResonanceAgent v18 - Flash-Skip + Immediate Replan

Key fix: detect timer restart (flash frame where player=None),
skip it, and immediately replan on the next frame.
"""

import numpy as np
from typing import List, Tuple, Set, Dict
from collections import deque, Counter
from arcengine import FrameData, GameAction, GameState
from .agent import Agent


class Shape:
    __slots__ = ('color','cells','center','size','role')
    def __init__(self, color, cells):
        self.color = color
        self.cells = cells
        self.size = len(cells)
        rs = [r for r,c in cells]
        cs = [c for r,c in cells]
        self.center = (sum(rs)//len(rs), sum(cs)//len(cs))
        self.role = 'unknown'


class ShapeIndex:
    def __init__(self, grid):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.shapes = []
        self.player_pos = None
        self.player_offsets = []
        self.marker_cells = set()
        self.lock_center = None
        self.timer_budget = 20
        self._build()
    
    def _build(self):
        visited = set()
        for r in range(min(self.rows,62)):
            for c in range(self.cols):
                if (r,c) not in visited:
                    color = int(self.grid[r,c])
                    cells = set()
                    stack = [(r,c)]
                    while stack:
                        cr,cc = stack.pop()
                        if (cr,cc) in visited or cr<0 or cr>=min(self.rows,62) or cc<0 or cc>=self.cols: continue
                        if int(self.grid[cr,cc]) != color: continue
                        visited.add((cr,cc))
                        cells.add((cr,cc))
                        stack.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                    if len(cells) >= 2:
                        self.shapes.append(Shape(color, cells))
        
        counts = Counter(int(v) for v in self.grid[:60].flatten())
        total = sum(counts.values())
        
        player_12 = None
        for s in self.shapes:
            if s.color == 12: s.role = 'player_top'; player_12 = s
            elif s.color in (0,1) and counts.get(s.color,0) < total*0.005:
                s.role = 'marker'; self.marker_cells |= s.cells
            elif s.color == 9 and s.size <= 30:
                has5 = any(0<=r+dr<self.rows and 0<=c+dc<self.cols and int(self.grid[r+dr,c+dc])==5
                          for r,c in s.cells for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
                if player_12 and any(abs(r-pr)<=3 and abs(c-pc)<=3 for r,c in s.cells for pr,pc in player_12.cells):
                    s.role = 'player_bottom'
                elif has5:
                    s.role = 'lock_or_template'
        
        if player_12:
            all_p = player_12.cells.copy()
            for s in self.shapes:
                if s.role == 'player_bottom': all_p |= s.cells
            rs = [r for r,c in all_p]; cs = [c for r,c in all_p]
            self.player_pos = (sum(rs)//len(rs), sum(cs)//len(cs))
            self.player_offsets = [(r-self.player_pos[0], c-self.player_pos[1]) for r,c in all_p]
        
        # Lock = lock_or_template closest to markers
        markers = [s for s in self.shapes if s.role == 'marker']
        lt = [s for s in self.shapes if s.role == 'lock_or_template']
        if markers and lt:
            mc = markers[0].center
            lt.sort(key=lambda s: abs(s.center[0]-mc[0])+abs(s.center[1]-mc[1]))
            self.lock_center = lt[0].center
        elif lt:
            self.lock_center = lt[0].center
        
        timer_cells = sum(1 for r in range(60,64) for c in range(self.cols) if self.grid[r,c]==11)
        self.timer_budget = max(timer_cells//4 - 1, 8)


class CrossResonanceAgent(Agent):
    MAX_ACTIONS = 500

    def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags)
        self._full_reset()

    def _full_reset(self):
        self.prev_grid = None
        self.player_pos = None
        self.player_shape = []
        self.action_queue = []
        self._prev_levels = 0
        self._planned = False
        self._needs_replan = False
        self._retry = 0

    def _can_move(self, g, cr, cc):
        for dr,dc in self.player_shape:
            nr,nc = cr+dr, cc+dc
            if nr<0 or nr>=64 or nc<0 or nc>=64: return False
            if g[nr,nc]==4: return False
        # Color 9 conflict: player's color 9 part can't overlap non-player color 9
        # Player bottom (dr >= 0) is color 9
        for dr,dc in self.player_shape:
            if dr >= 0:  # bottom part = color 9
                nr,nc = cr+dr, cc+dc
                if 0<=nr<64 and 0<=nc<64 and g[nr,nc]==9:
                    # This cell has color 9 — is it the player's own?
                    # If we're moving TO this position, the old position's 9s are gone
                    # So any 9 here is lock/template pattern = CONFLICT
                    return False
        return True

    def _sim(self, g, cr, cc, a):
        d = {1:(-5,0),2:(5,0),3:(0,-5),4:(0,5)}
        dr,dc = d[a]
        nr,nc = cr+dr, cc+dc
        return (nr,nc) if self._can_move(g,nr,nc) else (cr,cc)

    def _bfs_to(self, g, start, goal_fn):
        q = deque([(start,[])]); v = {start}
        while q:
            pos,path = q.popleft()
            if goal_fn(pos): return path, pos
            for a in [1,2,3,4]:
                np_ = self._sim(g,pos[0],pos[1],a)
                if np_!=pos and np_ not in v:
                    v.add(np_); q.append((np_,path+[a]))
        return [], start

    def _push_dir(self, fr, to):
        dr,dc = to[0]-fr[0], to[1]-fr[1]
        dirs = []
        if abs(dr)>=abs(dc):
            dirs.append(1 if dr<0 else 2)
            if dc!=0: dirs.append(3 if dc<0 else 4)
        else:
            dirs.append(3 if dc<0 else 4)
            if dr!=0: dirs.append(1 if dr<0 else 2)
        push = []
        for d in dirs: push.extend([d]*4)
        return push

    def _plan(self, grid):
        g = np.array(grid)
        idx = ShapeIndex(grid)
        
        self.player_pos = idx.player_pos
        self.player_shape = idx.player_offsets
        
        if not self.player_pos or not self.player_shape:
            self.action_queue = [1,2,3,4]*3
            self._planned = True
            return
        
        budget = idx.timer_budget
        
        # Strategy A: marker → lock (full path)
        mp, mend = [], self.player_pos
        if idx.marker_cells:
            mp, mend = self._bfs_to(g, self.player_pos,
                lambda p: any((p[0]+dr,p[1]+dc) in idx.marker_cells for dr,dc in self.player_shape))
        
        lp_m, lend_m = [], mend
        if idx.lock_center:
            lp_m, lend_m = self._bfs_to(g, mend,
                lambda p: abs(p[0]-idx.lock_center[0])<=7 and abs(p[1]-idx.lock_center[1])<=5)
        push_a = self._push_dir(lend_m, idx.lock_center) if idx.lock_center else [1]*4
        strat_a = mp + lp_m + push_a
        
        # Strategy B: direct to lock
        lp_d, lend_d = [], self.player_pos
        if idx.lock_center:
            lp_d, lend_d = self._bfs_to(g, self.player_pos,
                lambda p: abs(p[0]-idx.lock_center[0])<=7 and abs(p[1]-idx.lock_center[1])<=5)
        push_b = self._push_dir(lend_d, idx.lock_center) if idx.lock_center else [1]*4
        strat_b = lp_d + push_b
        
        # Choose: fits budget? Via marker preferred, else direct.
        if len(strat_a) <= budget:
            self.action_queue = strat_a
        elif len(strat_b) <= budget:
            self.action_queue = strat_b
        else:
            # Adaptive retry: r0=marker only, r1+=direct truncated
            if self._retry == 0 and mp:
                self.action_queue = mp[:budget]
            else:
                self.action_queue = strat_b[:budget]
        
        import sys; print(f"PLAN: retry={self._retry} budget={budget} strat_a={len(strat_a)} strat_b={len(strat_b)} chosen={len(self.action_queue)} q={self.action_queue[:5]}", file=sys.stderr)
        # Store lock push direction for fallback
        if idx.lock_center and self.player_pos:
            dr = idx.lock_center[0] - self.player_pos[0]
            dc = idx.lock_center[1] - self.player_pos[1]
            if abs(dr) >= abs(dc):
                self._lock_push_dir = 1 if dr < 0 else 2
            else:
                self._lock_push_dir = 3 if dc < 0 else 4
        else:
            self._lock_push_dir = 1
        
        self._planned = True
        self._needs_replan = False

    def is_done(self, frames, latest_frame):
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames, latest_frame):
        grid = latest_frame.frame[0]
        g = np.array(grid)

        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._full_reset()
            self._retry = 0  # only reset retry on full game reset
            return GameAction.RESET

        # Level transition
        if latest_frame.levels_completed != self._prev_levels:
            self._full_reset()
            self._prev_levels = latest_frame.levels_completed
            self._needs_replan = True
            # Don't plan on transition frame — grid may be stale
            self.prev_grid = [row[:] for row in grid]
            return GameAction.ACTION1

        # Detect flash frame (timer restart) — player disappears
        c12 = np.where(g == 12)
        player_visible = len(c12[0]) > 0
        
        if not player_visible:
            # Flash frame — skip, don't waste timer
            self.prev_grid = [row[:] for row in grid]
            self._needs_replan = True
            return GameAction.ACTION1
        
        new_pos = (int(c12[0].min())+2, int((c12[1].min()+c12[1].max())//2))
        
        # Detect restart: timer jumped back up
        timer = sum(1 for r in range(60,64) for c in range(64) if g[r,c]==11)
        prev_timer = 0
        if self.prev_grid is not None:
            prev_g = np.array(self.prev_grid)
            prev_timer = sum(1 for r in range(60,64) for c in range(64) if prev_g[r,c]==11)
        
        # Detect restart: timer jumped to normal from 0 or flash(>100)
        is_restart = (timer > prev_timer + 20) or (prev_timer > 100 and timer < 100 and timer > 50)
        if is_restart:
            import sys; print(f"TIMER_JUMP: timer={timer} prev={prev_timer} retry={self._retry}", file=sys.stderr)
            self._retry += 1
            self._needs_replan = True
        
        self.player_pos = new_pos
        
        # Plan if needed
        if self._needs_replan or not self._planned:
            self._plan(grid)

        action_id = 1
        if self.action_queue:
            action_id = self.action_queue.pop(0)
        else:
            # Queue empty. Push toward lock if known.
            if hasattr(self, "_lock_push_dir") and self._lock_push_dir:
                action_id = self._lock_push_dir

        self.prev_grid = [row[:] for row in grid]

        action_map = {1:GameAction.ACTION1, 2:GameAction.ACTION2,
                      3:GameAction.ACTION3, 4:GameAction.ACTION4}
        action = action_map.get(action_id, GameAction.ACTION1)
        action.reasoning = f"q={len(self.action_queue)} r={self._retry} t={timer} pos={self.player_pos}"
        return action
