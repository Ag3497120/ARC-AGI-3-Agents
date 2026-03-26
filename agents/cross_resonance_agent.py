"""
CrossResonanceAgent v16 - Shape Index + Precise Simulation

No "overview" step. Instead:
1. Convert grid to Cross structure with complete shape index
2. Every distinct shape/color cluster is catalogued
3. Relationships between shapes are mapped (contains, adjacent, similar)
4. Simulator uses shape index for precise pathfinding
5. Execute optimal path
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Dict, FrozenSet
from collections import deque, Counter
from arcengine import FrameData, GameAction, GameState
from .agent import Agent


# ═══════════════════════════════════════════════
# Shape: a connected cluster of same-color cells
# ═══════════════════════════════════════════════

class Shape:
    __slots__ = ('color', 'cells', 'center', 'bbox', 'size', 'normalized', 'role')
    
    def __init__(self, color: int, cells: Set[Tuple[int,int]]):
        self.color = color
        self.cells = cells
        self.size = len(cells)
        rs = [r for r,c in cells]
        cs = [c for r,c in cells]
        self.center = (sum(rs)//len(rs), sum(cs)//len(cs))
        self.bbox = (min(rs), min(cs), max(rs), max(cs))
        # Normalized pattern (translated to origin)
        mr, mc = min(rs), min(cs)
        self.normalized = frozenset((r-mr, c-mc) for r,c in cells)
        self.role = 'unknown'  # player, marker, lock, key_template, corridor, wall, timer


# ═══════════════════════════════════════════════
# Shape Index: complete catalogue of all shapes
# ═══════════════════════════════════════════════

class ShapeIndex:
    """Extracts and indexes ALL shapes from a grid."""
    
    def __init__(self, grid):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.shapes: List[Shape] = []
        self.by_role: Dict[str, List[Shape]] = {}
        self.player_shape_offsets = []
        
        self._extract_all()
        self._classify()
        self._find_relationships()
    
    def _extract_all(self):
        """Flood-fill to find all connected components of same color."""
        visited = set()
        for r in range(min(self.rows, 62)):  # skip timer rows
            for c in range(self.cols):
                if (r,c) not in visited:
                    color = int(self.grid[r,c])
                    cells = self._flood(r, c, color, visited)
                    if len(cells) >= 2:  # ignore single isolated cells
                        self.shapes.append(Shape(color, cells))
    
    def _flood(self, r, c, color, visited):
        cells = set()
        stack = [(r,c)]
        while stack:
            cr, cc = stack.pop()
            if (cr,cc) in visited or cr < 0 or cr >= min(self.rows,62) or cc < 0 or cc >= self.cols:
                continue
            if int(self.grid[cr,cc]) != color:
                continue
            visited.add((cr,cc))
            cells.add((cr,cc))
            stack.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
        return cells
    
    def _classify(self):
        """Assign roles to shapes based on properties."""
        color_counts = Counter(int(v) for v in self.grid[:60].flatten())
        total = sum(color_counts.values())
        
        for shape in self.shapes:
            c = shape.color
            
            if c == 12:
                shape.role = 'player_top'
            elif c == 9 and shape.size <= 30:
                # Could be player bottom, lock pattern, or key template
                # Check if adjacent to color 12 (player)
                has_12_neighbor = any(
                    int(self.grid[r+dr,c+dc]) == 12
                    for r,cc in shape.cells
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                    if 0 <= r+dr < self.rows and 0 <= cc+dc < self.cols
                    for c in [cc]  # hack to use cc
                )
                if has_12_neighbor:
                    shape.role = 'player_bottom'
                else:
                    # Check if enclosed by color 5
                    has_5_neighbor = any(
                        0 <= r+dr < self.rows and 0 <= c+dc < self.cols and
                        int(self.grid[r+dr,c+dc]) == 5
                        for r,c in shape.cells
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                    )
                    if has_5_neighbor:
                        shape.role = 'lock_or_template'
            elif c in (0, 1) and color_counts.get(c, 0) < total * 0.005:
                shape.role = 'marker'
            elif c == 5:
                if shape.size < 100:
                    shape.role = 'lock_border'
                else:
                    shape.role = 'ui_border'
            elif c == 3:
                shape.role = 'corridor'
            elif c == 4:
                shape.role = 'wall'
            elif c == 11:
                shape.role = 'timer'
        
        # Build role index
        self.by_role = {}
        for s in self.shapes:
            if s.role not in self.by_role:
                self.by_role[s.role] = []
            self.by_role[s.role].append(s)
        
        # Identify player and build player shape offsets
        player_tops = self.by_role.get('player_top', [])
        player_bots = self.by_role.get('player_bottom', [])
        if player_tops:
            pt = player_tops[0]
            all_player = pt.cells.copy()
            if player_bots:
                all_player |= player_bots[0].cells
            rs = [r for r,c in all_player]
            cs = [c for r,c in all_player]
            center = (sum(rs)//len(rs), sum(cs)//len(cs))
            self.player_pos = center
            self.player_shape_offsets = [(r-center[0], c-center[1]) for r,c in all_player]
        else:
            self.player_pos = None
            self.player_shape_offsets = []
    
    def _find_relationships(self):
        """Find which lock_or_template is closest to markers (= the lock)."""
        markers = self.by_role.get('marker', [])
        lock_templates = self.by_role.get('lock_or_template', [])
        
        if markers and lock_templates:
            mc = markers[0].center
            # Closest lock_or_template to markers = lock
            lock_templates.sort(key=lambda s: abs(s.center[0]-mc[0]) + abs(s.center[1]-mc[1]))
            lock_templates[0].role = 'lock_pattern'
            # Others are key_template
            for s in lock_templates[1:]:
                s.role = 'key_template'
            
            # Rebuild role index
            self.by_role = {}
            for s in self.shapes:
                if s.role not in self.by_role:
                    self.by_role[s.role] = []
                self.by_role[s.role].append(s)
    
    def get_marker_center(self):
        markers = self.by_role.get('marker', [])
        if markers:
            all_cells = set()
            for m in markers:
                all_cells |= m.cells
            rs = [r for r,c in all_cells]
            cs = [c for r,c in all_cells]
            return (sum(rs)//len(rs), sum(cs)//len(cs))
        return None
    
    def get_lock_center(self):
        locks = self.by_role.get('lock_pattern', [])
        if locks:
            return locks[0].center
        # Fallback: lock_border closest to markers
        borders = self.by_role.get('lock_border', [])
        mc = self.get_marker_center()
        if borders and mc:
            borders.sort(key=lambda s: abs(s.center[0]-mc[0]) + abs(s.center[1]-mc[1]))
            return borders[0].center
        return None


# ═══════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════

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

    def _can_move(self, grid, cr, cc):
        for dr, dc in self.player_shape:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= 64 or nc < 0 or nc >= 64:
                return False
            if grid[nr, nc] == 4:
                return False
        return True

    def _sim(self, grid, cr, cc, a):
        deltas = {1: (-5,0), 2: (5,0), 3: (0,-5), 4: (0,5)}
        dr, dc = deltas[a]
        nr, nc = cr+dr, cc+dc
        return (nr,nc) if self._can_move(grid, nr, nc) else (cr,cc)

    def _bfs_to(self, grid, start, goal_fn):
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            if goal_fn(pos):
                return path, pos
            for a in [1,2,3,4]:
                np_ = self._sim(grid, pos[0], pos[1], a)
                if np_ != pos and np_ not in visited:
                    visited.add(np_)
                    queue.append((np_, path + [a]))
        return [], start

    def _plan(self, grid):
        """Build complete plan using Shape Index."""
        g = np.array(grid)
        
        # Build shape index
        idx = ShapeIndex(grid)
        
        self.player_pos = idx.player_pos
        self.player_shape = idx.player_shape_offsets
        
        if not self.player_pos or not self.player_shape:
            self.action_queue = [1,2,3,4] * 5
            self._planned = True
            return
        
        marker_center = idx.get_marker_center()
        lock_center = idx.get_lock_center()
        
        # Estimate timer: count color 11 cells / 4 per frame
        timer_cells = sum(1 for r in range(60,64) for c in range(64) if g[r,c] == 11)
        timer_budget = max(timer_cells // 4 - 2, 10)
        
        # Get marker cells for overlap check
        marker_cells = set()
        for m in idx.by_role.get('marker', []):
            marker_cells |= m.cells
        
        # BFS: find marker overlap position
        marker_path = []
        marker_pos = self.player_pos
        if marker_cells:
            def overlaps(pos):
                return any((pos[0]+dr, pos[1]+dc) in marker_cells
                          for dr, dc in self.player_shape)
            marker_path, marker_pos = self._bfs_to(g, self.player_pos, overlaps)
        
        # BFS: find lock approach from marker
        lock_path = []
        if lock_center:
            def near_lock(pos):
                return abs(pos[0]-lock_center[0]) <= 7 and abs(pos[1]-lock_center[1]) <= 5
            lock_path, _ = self._bfs_to(g, marker_pos, near_lock)
        
        # BFS: direct to lock (skip markers)
        direct_path = []
        if lock_center:
            def near_lock2(pos):
                return abs(pos[0]-lock_center[0]) <= 7 and abs(pos[1]-lock_center[1]) <= 5
            direct_path, _ = self._bfs_to(g, self.player_pos, near_lock2)
        
        # Push direction toward lock
        push_actions = []
        if lock_center:
            approach_pos = marker_pos
            if lock_path:
                for a in lock_path:
                    approach_pos = self._sim(g, approach_pos[0], approach_pos[1], a)
            dr = lock_center[0] - approach_pos[0]
            dc = lock_center[1] - approach_pos[1]
            pushes = []
            if abs(dr) >= abs(dc):
                pushes.append(1 if dr < 0 else 2)
                if dc != 0: pushes.append(3 if dc < 0 else 4)
            else:
                pushes.append(3 if dc < 0 else 4)
                if dr != 0: pushes.append(1 if dr < 0 else 2)
            for p in pushes:
                push_actions.extend([p] * 4)
        
        # Choose strategy based on timer budget
        via_marker_total = len(marker_path) + len(lock_path) + len(push_actions)
        direct_total = len(direct_path) + len(push_actions)
        
        if via_marker_total <= timer_budget:
            self.action_queue = marker_path + lock_path + push_actions
        elif direct_total <= timer_budget:
            self.action_queue = direct_path + push_actions
        else:
            # Neither fits. Use shortest available.
            if via_marker_total <= direct_total:
                self.action_queue = marker_path + lock_path + push_actions
            else:
                self.action_queue = direct_path + push_actions
        
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
            # Don't plan yet, just observe this frame

        if not self._planned:
            self._plan(grid)

        # Track player
        g = np.array(grid)
        c12 = np.where(g == 12)
        if len(c12[0]) > 0:
            self.player_pos = (int(c12[0].min())+2, int((c12[1].min()+c12[1].max())//2))

        action_id = 1
        if self.action_queue:
            action_id = self.action_queue.pop(0)

        self.prev_grid = [row[:] for row in grid]

        action_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                      3: GameAction.ACTION3, 4: GameAction.ACTION4}
        action = action_map.get(action_id, GameAction.ACTION1)
        action.reasoning = f"q={len(self.action_queue)} pos={self.player_pos} lvl={latest_frame.levels_completed}"
        return action
