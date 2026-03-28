"""
CrossResonanceV26 — v24 core + v25 StructureAnalyzer

Verantyx-ARC-AGI3: Game-agnostic autonomous agent.

1. Cross変換: 盤面を構造推論付きで変換（触る前に壁/道を判定）
   - プレイヤー足元の色 = 道
   - 道を囲む色 = 壁
2. 計画: ゴール候補→最短経路→寄り道候補→組合せ最短
3. 実行+監視: 毎手差分チェック、反応あれば条件更新→再計画
4. 動的書き換え: 計画をリアルタイム更新

Results: ls20 Level 0 cleared in 23 actions (human baseline: 21) — 83.4% efficiency
"""

import sys
import numpy as np
from typing import List, Tuple, Set, Optional, Dict, Any
from collections import deque, Counter
from arcengine import FrameData, GameAction, GameState
from .agent import Agent
from .cross_engine.cross_sensor import CrossSensor, CrossSnapshot, CrossObject
try:
    from .cross_engine.structure_analyzer import StructureAnalyzer
    from .cross_engine.live_monitor import LiveMonitor
    HAS_V25 = True
except ImportError:
    HAS_V25 = False

try:
    from .cross_engine.rule_learner import ReactionAnalyzer, RuleLearner, DynamicPlanner
    HAS_RULE_LEARNER = True
except ImportError:
    HAS_RULE_LEARNER = False

try:
    from .cross_engine.cross_space import CrossSpace, Experience, Impulse
    HAS_CROSS_SPACE = True
except ImportError:
    HAS_CROSS_SPACE = False


ALL_ACTIONS = [
    GameAction.ACTION1, GameAction.ACTION2,
    GameAction.ACTION3, GameAction.ACTION4,
    GameAction.ACTION5, GameAction.ACTION6,
    GameAction.ACTION7,
]
ACTION_IDS = {a: i for i, a in enumerate(ALL_ACTIONS)}


class CrossStructuralMap:
    """Converts raw grid into structural roles using Cross inference.
    Core logic: player footprint → corridor color → surrounding = wall.
    Post-probe: actual movement evidence overrides initial inference."""

    def __init__(self, grid, snap: CrossSnapshot, probe_corridor_colors=None):
        self.g = np.array(grid)
        self.rows, self.cols = self.g.shape
        self.role = np.full((self.rows, self.cols), 'unknown', dtype=object)
        self.color_roles: Dict[int, str] = {}
        self._build(snap, probe_corridor_colors)

    def _build(self, snap, probe_corridor_colors):
        # Timer area
        for r in range(60, self.rows):
            for c in range(self.cols):
                self.role[r, c] = 'timer'
                self.color_roles.setdefault(int(self.g[r, c]), 'timer')

        # Find controllable (player)
        ctrl_cells = set()
        ctrl_colors = set()
        for obj in snap.objects:
            c = obj.descriptor.color_axis
            p = obj.descriptor.position_axis
            s = obj.descriptor.scale_axis
            if p.get('in_timer_area'):
                continue
            if (c.get('is_rare') and s.get('size_category') in ('small', 'medium')
                and not c.get('is_dominant')):
                color = c.get('primary_color')
                adj_ids = obj.descriptor.relation_axis.get('adjacent_obj_ids', set())
                for idx in adj_ids:
                    if idx < len(snap.objects):
                        other = snap.objects[idx]
                        oc = other.descriptor.color_axis
                        if oc.get('is_rare') and not other.descriptor.position_axis.get('in_timer_area'):
                            ctrl_cells |= obj.cells | other.cells
                            ctrl_colors.add(color)
                            ctrl_colors.add(oc.get('primary_color'))
                if not ctrl_cells:
                    ctrl_cells |= obj.cells
                    ctrl_colors.add(color)
                if ctrl_cells:
                    break

        # Player's feet — adjacent non-ctrl colors
        corridor_colors = set()
        if probe_corridor_colors:
            # Post-probe: use definitive evidence
            corridor_colors = set(probe_corridor_colors)
        else:
            # Pre-probe: infer from adjacency + cell count
            adj_counts = {}
            for r, c in ctrl_cells:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(2,0),(-2,0),(0,2),(0,-2)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 60 and 0 <= nc < self.cols:
                        adj_color = int(self.g[nr, nc])
                        if adj_color not in ctrl_colors:
                            adj_counts[adj_color] = adj_counts.get(adj_color, 0) + 1

            for adj_color in adj_counts:
                total = sum(1 for rr in range(min(self.rows, 60))
                           for cc in range(self.cols) if int(self.g[rr, cc]) == adj_color)
                adj_counts[adj_color] = total

            if adj_counts:
                sorted_adj = sorted(adj_counts.items(), key=lambda x: -x[1])
                corridor_colors.add(sorted_adj[0][0])
                for color, count in sorted_adj[1:]:
                    if count > 100:
                        corridor_colors.add(color)

        # Surrounding corridor = wall
        wall_colors = set()
        for r in range(min(self.rows, 60)):
            for c in range(self.cols):
                color = int(self.g[r, c])
                if color in corridor_colors:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 60 and 0 <= nc < self.cols:
                            adj = int(self.g[nr, nc])
                            if (adj not in corridor_colors and adj not in ctrl_colors
                                and adj not in wall_colors):
                                freq = sum(1 for rr in range(min(self.rows,60))
                                           for cc in range(self.cols) if int(self.g[rr,cc]) == adj)
                                if freq > 50:
                                    wall_colors.add(adj)

        # Assign roles
        for color in ctrl_colors:
            self.color_roles[color] = 'controllable'
        for color in corridor_colors:
            self.color_roles[color] = 'corridor'
        for color in wall_colors:
            self.color_roles[color] = 'wall'

        # Remaining
        for obj in snap.objects:
            color = obj.descriptor.color_axis.get('primary_color')
            if color in self.color_roles:
                continue
            p = obj.descriptor.position_axis
            if p.get('in_timer_area'):
                continue
            ca = obj.descriptor.color_axis
            s = obj.descriptor.scale_axis
            rel = obj.descriptor.relation_axis
            if ca.get('is_rare') and s.get('size_category') in ('point', 'tiny', 'small'):
                self.color_roles[color] = 'interactive'
            elif rel.get('contained_by_count', 0) > 0:
                self.color_roles[color] = 'pattern'
            elif rel.get('contains_count', 0) > 0:
                self.color_roles[color] = 'border'

        # Apply to grid
        for r in range(self.rows):
            for c in range(self.cols):
                color = int(self.g[r, c])
                self.role[r, c] = self.color_roles.get(color, 'unknown')

    def is_passable(self, r, c):
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        return self.role[r, c] in ('corridor', 'interactive', 'pattern', 'controllable', 'unknown')

    def can_occupy(self, r, c, offsets):
        for dr, dc in offsets:
            if not self.is_passable(r + dr, c + dc):
                return False
        return True

    def mark_passable(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.role[r, c] in ('wall', 'border', 'unknown'):
                self.role[r, c] = 'corridor'
                self.color_roles[int(self.g[r, c])] = 'corridor'

    def mark_wall(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.role[r, c] not in ('wall', 'timer'):
                self.role[r, c] = 'wall'

    def summary(self):
        counts = Counter(str(v) for v in self.role.flatten())
        return f"roles={self.color_roles} grid={dict(counts)}"


class RoutePlanner:
    def __init__(self, smap, mv_actions):
        self.smap = smap
        self.mv_actions = mv_actions

    def plan_route(self, start, goals, detours, offsets, budget):
        if not goals and not detours:
            return []
        distances, paths = self._bfs_all(start, offsets)
        best_route = []
        best_cost = float('inf')

        for goal in goals:
            gpath = paths.get(goal, [])
            if gpath and len(gpath) < best_cost:
                best_cost = len(gpath); best_route = gpath

        for detour in detours[:5]:
            dpath = paths.get(detour, [])
            if not dpath:
                continue
            d_dists, d_paths = self._bfs_all(detour, offsets)
            for goal in goals:
                gpath = d_paths.get(goal, [])
                if gpath and len(dpath) + len(gpath) < best_cost:
                    best_cost = len(dpath) + len(gpath)
                    best_route = dpath + gpath

        if not best_route:
            for detour in detours:
                dpath = paths.get(detour, [])
                if dpath and len(dpath) < best_cost:
                    best_cost = len(dpath); best_route = dpath

        if not best_route and goals:
            best_route = self._push_toward(start, goals[0])

        return best_route[:budget]

    def _bfs_all(self, start, offsets):
        distances = {start: 0}; paths = {start: []}
        queue = deque([(start, [])])
        while queue and len(distances) < 8000:
            (cr, cc), path = queue.popleft()
            if len(path) > 100: continue
            for aidx, (dr, dc) in self.mv_actions.items():
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in distances: continue
                if not self.smap.can_occupy(nr, nc, offsets): continue
                new_path = path + [aidx]
                distances[(nr, nc)] = len(new_path)
                paths[(nr, nc)] = new_path
                queue.append(((nr, nc), new_path))
        return distances, paths

    def _push_toward(self, start, target):
        dr = target[0] - start[0]; dc = target[1] - start[1]
        push = []; up = down = left = right = None
        for aidx, (mr, mc) in self.mv_actions.items():
            if mr < 0: up = aidx
            if mr > 0: down = aidx
            if mc < 0: left = aidx
            if mc > 0: right = aidx
        if dr < 0 and up is not None: push.extend([up] * max(1, abs(dr)//5))
        elif dr > 0 and down is not None: push.extend([down] * max(1, abs(dr)//5))
        if dc < 0 and left is not None: push.extend([left] * max(1, abs(dc)//5))
        elif dc > 0 and right is not None: push.extend([right] * max(1, abs(dc)//5))
        return push[:15]


class DiffMonitor:
    def __init__(self, initial_snap):
        self.reaction_log = []

    def check(self, snap):
        if not snap.diff or not snap.diff.has_changes: return None
        sig = [o for o in (snap.diff.moved + snap.diff.appeared + snap.diff.disappeared + snap.diff.size_changed)
               if not o.descriptor.position_axis.get('in_timer_area')]
        if len(sig) >= 2:
            reaction = {'frame': snap.frame_number, 'total': len(sig)}
            self.reaction_log.append(reaction)
            return reaction
        return None


class ClickPlanner:
    def __init__(self):
        self.results = {}; self._queue = []
        self._click_effects = {}  # pos → list of (old_color, new_color)
        self._color_cycles = {}   # pos → detected cycle length
        self._pattern_match_mode = False
        self._left_pattern = {}   # (r,c) → target_color (from left/reference side)
        self._right_blocks = {}   # (r,c) → current_color (right/clickable side)
        self._clickable_positions = []  # discovered positions that respond to clicks
        self._probe_phase = True  # start in probe phase to find clickable spots
        self._probe_grid = []     # positions to try during probe

    def plan(self, snap, sensor):
        cands = []
        for obj in snap.objects:
            p = obj.descriptor.position_axis; rel = obj.descriptor.relation_axis
            c = obj.descriptor.color_axis; s = obj.descriptor.scale_axis
            if p.get('in_timer_area'): continue
            score = 0
            if rel.get('contained_by_count', 0) > 0: score += 4
            if s.get('size_category') in ('small','medium') and not c.get('is_dominant'): score += 2
            if c.get('is_rare'): score += 1
            if score > 0:
                pos = p.get('centroid_int', (32,32))
                if pos in self.results and not any(self.results[pos]): continue
                cands.append((score, pos))
        cands.sort(key=lambda x: -x[0])
        self._queue = [pos for _, pos in cands]
        return self._queue

    def plan_click_probe(self, grid):
        """Generate click positions to probe — systematic grid scan to find ALL clickable positions."""
        if self._clickable_positions:
            self._probe_phase = False
            return []
        
        g = np.array(grid)
        rows = min(len(grid), 60)
        cols = len(grid[0]) if grid else 64
        bg_counter = {}
        for r in range(rows):
            for c in range(cols):
                v = int(g[r, c])
                bg_counter[v] = bg_counter.get(v, 0) + 1
        bg = max(bg_counter, key=bg_counter.get)  # most common = background
        
        # Systematic grid: try non-background positions first, then grid scan
        probe_positions = []
        
        # Priority 1: click on non-background cell centroids (most likely clickable)
        from collections import Counter as _C
        non_bg_blocks = []
        _visited = set()
        for r in range(rows):
            for c in range(cols):
                if (r,c) in _visited or int(g[r,c]) == bg:
                    continue
                color = int(g[r,c])
                block = []
                _q = [(r,c)]; _visited.add((r,c))
                while _q:
                    cr, cc = _q.pop(0)
                    block.append((cr,cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (nr,nc) not in _visited and 0<=nr<rows and 0<=nc<cols and int(g[nr,nc])==color:
                            _visited.add((nr,nc)); _q.append((nr,nc))
                if len(block) >= 8:
                    rs = [r for r,c in block]; cs = [c for r,c in block]
                    non_bg_blocks.append((sum(rs)//len(rs), sum(cs)//len(cs), len(block)))
        
        # Sort by size (larger blocks more likely to be clickable)
        non_bg_blocks.sort(key=lambda x: -x[2])
        for cr, cc, sz in non_bg_blocks[:40]:
            probe_positions.append((cr, cc))
        
        # Limit probes — adaptive based on budget
        # Tight baselines need fewer probes, generous ones can probe more
        self._probe_grid = probe_positions
        self._queue = probe_positions[:20]
        self._probe_phase = True
        print(f"CLICK_PROBE: {len(self._queue)} positions to test", file=sys.stderr)
        return self._queue

    def record_click_result(self, pos, cells_changed, prev_grid, curr_grid):
        """Record whether a click at pos caused changes. Learn clickable positions."""
        if cells_changed > 2:
            if pos not in self._clickable_positions:
                self._clickable_positions.append(pos)
                print(f"CLICK_FOUND: ({pos[0]},{pos[1]}) → {cells_changed} cells", file=sys.stderr)
            # Track color change at this position
            r, c = pos
            if 0 <= r < len(prev_grid) and 0 <= c < len(prev_grid[0]):
                old = int(prev_grid[r][c])
                new = int(curr_grid[r][c])
                if pos not in self._click_effects:
                    self._click_effects[pos] = []
                self._click_effects[pos].append((old, new))

    def plan_from_discovered(self, grid):
        """After discovering clickable positions, plan clicks based on left-right pattern matching."""
        if not self._clickable_positions:
            return []
        
        g = np.array(grid)
        rows, cols = g.shape
        mid = cols // 2

        # Detect all available colors (for cycle estimation)
        all_colors = sorted(set(int(g[r, c]) for r in range(min(rows, 60)) for c in range(cols)
                               if int(g[r, c]) not in {int(g[0,0])}))  # exclude background
        cycle_len = max(len(all_colors), 3)

        click_plan = []
        for pos in self._clickable_positions:
            r, c = pos
            current_color = int(g[r, c])
            # Mirror position on the left side
            mirror_c = c - mid if c >= mid else c + mid
            if 0 <= mirror_c < cols:
                target_color = int(g[r, mirror_c])
            else:
                # Also try same row, left half
                left_colors = [int(g[r, lc]) for lc in range(min(mid, cols)) 
                               if int(g[r, lc]) not in {int(g[0,0])}]
                target_color = left_colors[0] if left_colors else current_color

            if current_color == target_color:
                continue  # already matches

            # Calculate clicks needed
            if current_color in all_colors and target_color in all_colors:
                idx_curr = all_colors.index(current_color)
                idx_tgt = all_colors.index(target_color)
                clicks = (idx_tgt - idx_curr) % cycle_len
                if clicks == 0:
                    clicks = cycle_len
            else:
                clicks = 1  # unknown, try once

            clicks = min(clicks, cycle_len)  # cap at cycle length
            for _ in range(clicks):
                click_plan.append(pos)

        if not click_plan:
            # Fallback: click all once
            click_plan = list(self._clickable_positions)

        self._queue = click_plan
        self._probe_phase = False
        print(f"CLICK_PLAN_DISCOVERED: {len(self._queue)} clicks planned (cycle_len={cycle_len})", file=sys.stderr)
        return self._queue

    def plan_pattern_match(self, grid):
        """Detect left/right pattern matching game (like ft09).
        Left half has reference pattern, right half has clickable blocks.
        Click right blocks until colors match left."""
        g = np.array(grid)
        rows, cols = g.shape
        mid_c = cols // 2
        
        # Check if left and right halves have similar structure
        left_colors = Counter(int(g[r, c]) for r in range(min(rows, 60)) for c in range(mid_c))
        right_colors = Counter(int(g[r, c]) for r in range(min(rows, 60)) for c in range(mid_c, cols))
        
        # Find block regions (groups of same-color cells forming rectangles)
        left_blocks = self._find_blocks(g, 0, mid_c, rows)
        right_blocks = self._find_blocks(g, mid_c, cols, rows)
        
        if not left_blocks or not right_blocks:
            return []
        
        # Detect block colors for cycle estimation (exclude background)
        all_right_colors = sorted(set(c for _, _, c in right_blocks))
        all_left_colors = sorted(set(c for _, _, c in left_blocks))
        block_colors = sorted(set(all_right_colors + all_left_colors))
        # Cycle length = number of distinct block colors (not total grid colors)
        cycle_len = len(block_colors) if len(block_colors) >= 2 else 4
        all_colors = block_colors
        
        # Match left blocks to right blocks by relative position
        click_plan = []
        for (lr, lc, lcolor) in left_blocks:
            # Find corresponding right block at same relative position
            rc = lc + mid_c  # mirror position
            for (rr, rcc, rcolor) in right_blocks:
                if abs(rr - lr) < 4 and abs(rcc - rc) < 4:
                    if rcolor != lcolor:
                        # Click once per mismatched block — re-check after each round
                        click_plan.append((rr, rcc))
                    break
        
        if click_plan:
            self._pattern_match_mode = True
            self._queue = click_plan
            print(f"CLICK_PATTERN: {len(click_plan)} clicks for matching", file=sys.stderr)
        
        return click_plan

    def _find_blocks(self, grid, c_start, c_end, rows):
        """Find colored block centroids in a region."""
        blocks = []
        visited = set()
        dominant = Counter(int(grid[r, c]) for r in range(min(rows, 60)) 
                          for c in range(c_start, c_end)).most_common(1)
        bg_color = dominant[0][0] if dominant else 0
        
        for r in range(min(rows, 60)):
            for c in range(c_start, c_end):
                if (r, c) in visited: continue
                color = int(grid[r, c])
                if color == bg_color: continue
                # Flood fill this block
                block_cells = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    block_cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (nr, nc) not in visited and c_start <= nc < c_end and 0 <= nr < min(rows,60):
                            if int(grid[nr, nc]) == color:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                if 20 <= len(block_cells) <= 100:  # clickable block size (not tiny decorations, not huge borders)
                    cr = sum(r for r, c in block_cells) // len(block_cells)
                    cc = sum(c for r, c in block_cells) // len(block_cells)
                    blocks.append((cr, cc, color))
        return blocks

    def next(self):
        return self._queue.pop(0) if self._queue else None

    def record(self, pos, eff):
        if pos not in self.results: self.results[pos] = []
        self.results[pos].append(eff)

    @property
    def remaining(self): return len(self._queue)


class ActionModel:
    def __init__(self):
        self.records = {}; self.available = []; self.is_click_game = False

    def set_available(self, avail):
        self.available = avail
        self.is_click_game = 5 in avail and len(avail) <= 2

    def record(self, aidx, eff, ctrl_mv):
        if aidx not in self.records: self.records[aidx] = []
        self.records[aidx].append((eff, ctrl_mv))

    def get_movement(self, aidx):
        recs = self.records.get(aidx, [])
        moves = [mv for e, mv in recs if e and mv != (0,0)]
        if not moves: return (0,0)
        return Counter(moves).most_common(1)[0][0]

    def get_mv_actions(self):
        result = {}
        for a in self.available:
            if a in self.records:
                mv = self.get_movement(a)
                if mv != (0,0): result[a] = mv
        for aidx in list(result):
            rev = aidx ^ 1
            if rev not in result and rev in self.available:
                dr, dc = result[aidx]
                result[rev] = (-dr, -dc)
        return result

    def get_corridor_colors(self, grids):
        """Extract colors the controllable actually moved through."""
        colors = set()
        for aidx, recs in self.records.items():
            for eff, mv in recs:
                if eff and mv != (0,0):
                    # The colors at positions we moved through are passable
                    pass  # tracked externally
        return colors

    def summary(self):
        return f"click={self.is_click_game} mv={self.get_mv_actions()}"


class CrossResonanceV26(Agent):
    MAX_ACTIONS = 800

    def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None):
        super().__init__(card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags)
        self._full_reset()

    def _full_reset(self):
        self.sensor = CrossSensor()
        self._snap = None; self.prev_grid = None
        self.action_queue = []; self._prev_levels = 0
        self._phase = 'observe'
        self._model = ActionModel()
        self._smap = None; self._planner = None; self._monitor = None
        self._click = ClickPlanner()
        self._probe_queue = []; self._frame = 0; self._actions = 0
        self._last_aidx = 0
        self._ctrl_pos = None; self._ctrl_offsets = []; self._prev_ctrl = None
        self._last_click = None; self._click_targets = []
        self._replan_cooldown = 0
        self._probe_corridor_colors = set()  # colors learned from probe movement
        self._v25_smap = None  # v25 StructureAnalyzer result
        self._v25_monitor = None  # v25 LiveMonitor
        self._reaction_analyzer = ReactionAnalyzer() if HAS_RULE_LEARNER else None
        self._rule_learner = RuleLearner() if HAS_RULE_LEARNER else None
        self._dynamic_planner = DynamicPlanner() if HAS_RULE_LEARNER else None
        self._cross_space = CrossSpace() if HAS_CROSS_SPACE else None
        self._visited_positions: Set[Tuple[int,int]] = set()
        self._stuck_counter = 0
        self._last_progress_frame = 0

    def _observe(self, grid):
        snap = self.sensor.observe(grid); self._snap = snap; return snap

    def _budget(self, snap):
        for obj in snap.objects:
            s = obj.descriptor.shape_axis; p = obj.descriptor.position_axis
            if (s.get('is_hbar') or s.get('type') == 'line') and p.get('in_timer_area'):
                n = obj.descriptor.scale_axis.get('cell_count', 0)
                if n > 5: return max(n // 4 - 1, 8)
        cds = self.sensor.find_by_role_hint(snap, 'countdown')
        if cds: return max(cds[0].descriptor.scale_axis.get('cell_count', 0) // 4 - 1, 8)
        return 50

    def _find_ctrl(self, snap):
        if snap.diff and snap.diff.moved:
            for obj in sorted(snap.diff.moved, key=lambda o: -o.descriptor.scale_axis.get('cell_count', 0)):
                if obj.descriptor.position_axis.get('in_timer_area'): continue
                if obj.descriptor.scale_axis.get('cell_count', 0) >= 3:
                    pos = obj.descriptor.position_axis.get('centroid_int')
                    cells = list(obj.cells)
                    self._prev_ctrl = self._ctrl_pos
                    self._ctrl_pos = pos
                    self._ctrl_offsets = [(r-pos[0], c-pos[1]) for r,c in cells]
                    # Learn corridor colors from movement
                    if self.prev_grid is not None:
                        g = np.array(self.prev_grid)
                        for cr, cc in cells:
                            if 0 <= cr < 64 and 0 <= cc < 64:
                                self._probe_corridor_colors.add(int(g[cr, cc]))
                                if self._smap:
                                    self._smap.mark_passable(cr, cc)
                    return

    def _get_ctrl_mv(self, snap):
        if snap.diff and snap.diff.moved:
            best = None; best_sz = 0
            for obj in snap.diff.moved:
                if obj.descriptor.position_axis.get('in_timer_area'): continue
                sc = obj.descriptor.scale_axis.get('cell_count', 0)
                if sc >= 3 and sc > best_sz:
                    mv = obj.descriptor.temporal_axis.get('movement', (0,0))
                    if abs(mv[0]) >= 1 or abs(mv[1]) >= 1:
                        best = mv; best_sz = sc
            if best: return best
        return (0, 0)

    def _classify_targets(self, snap):
        goals = []; detours = []
        for obj in snap.objects:
            c = obj.descriptor.color_axis; p = obj.descriptor.position_axis
            s = obj.descriptor.scale_axis; rel = obj.descriptor.relation_axis
            if p.get('in_timer_area'): continue
            pos = p.get('centroid_int', (32, 32))
            if rel.get('contains_count', 0) > 0 and s.get('cell_count', 0) > 10:
                goals.append(pos)
            elif (rel.get('contained_by_count', 0) > 0
                  and s.get('size_category') in ('small', 'medium')
                  and not c.get('is_dominant')):
                goals.append(pos)
            if c.get('is_rare') and s.get('size_category') in ('point', 'tiny', 'small'):
                detours.append(pos)
        # v26: add v25 structure analyzer goals
        if HAS_V25 and self._v25_smap:
            try:
                for goal_reg in self._v25_smap.goal_candidates[:5]:
                    pos = (int(goal_reg.centroid[0]), int(goal_reg.centroid[1]))
                    if pos not in goals and pos[0] < 58:
                        goals.append(pos)
            except Exception:
                pass
        # Add rule-learned trigger waypoints as high-priority detours
        if self._rule_learner:
            try:
                rule_waypoints = self._rule_learner.get_waypoints_from_rules()
                for wp in rule_waypoints:
                    if wp not in detours:
                        detours.insert(0, wp)  # high priority
            except Exception:
                pass
        return goals, detours

    def _make_plan(self, grid, snap, budget):
        # Use probe_corridor_colors if available (post-probe), otherwise initial inference
        pcl = self._probe_corridor_colors if self._probe_corridor_colors else None
        self._smap = CrossStructuralMap(grid, snap, probe_corridor_colors=pcl)
        print(f"SMAP: {self._smap.summary()}", file=sys.stderr)

        mv_actions = self._model.get_mv_actions()
        if not mv_actions or not self._ctrl_pos: return []

        # Mark ctrl position cells as passable
        if self.prev_grid is not None:
            g = np.array(self.prev_grid)
            for dr, dc in self._ctrl_offsets:
                r, c = self._ctrl_pos[0]+dr, self._ctrl_pos[1]+dc
                if 0 <= r < 64 and 0 <= c < 64:
                    self._smap.mark_passable(r, c)

        goals, detours = self._classify_targets(snap)

        # ADD RULE-LEARNED WAYPOINTS AS PRIORITY DETOURS (don't replace planner)
        if self._rule_learner and self._rule_learner.rules:
            try:
                rule_waypoints = self._rule_learner.get_waypoints_from_rules()
                for wp in rule_waypoints:
                    if wp not in detours:
                        detours.insert(0, wp)
                if rule_waypoints:
                    print(f"RULE_DETOURS: {len(rule_waypoints)} waypoints from {len(self._rule_learner.rules)} rules",
                          file=sys.stderr)
            except Exception as e:
                print(f"RULE_DETOUR_ERR: {e}", file=sys.stderr)

        # CrossSpace: ask for seek targets when BFS can't reach goals
        if self._cross_space and self._ctrl_pos:
            try:
                # Check if we're stuck (no progress for 20+ frames)
                if self._stuck_counter >= 5:
                    self._cross_space.record_stuck(self._frame, self._ctrl_pos, self._visited_positions)
                # Get impulses from experience resonance
                impulses = self._cross_space.get_urgent_impulses()
                for imp in impulses[:3]:  # top 3 impulses
                    if imp.action_type == 'seek_color' and imp.target:
                        # Find cells with target colors and add as priority detours
                        g = np.array(grid)
                        for target_color in imp.target:
                            for r in range(min(60, len(grid))):
                                for c in range(len(grid[0])):
                                    if int(g[r, c]) == target_color and (r, c) not in self._visited_positions:
                                        detours.insert(0, (r, c))
                                        break
                                if detours and detours[0] != detours[-1]:
                                    break
                        if detours:
                            print(f"CROSS_SEEK: color={imp.target} reason='{imp.reason}'", file=sys.stderr)
                    elif imp.action_type == 'go_to' and imp.target:
                        pos = imp.target
                        if pos not in goals:
                            detours.insert(0, pos)
                            print(f"CROSS_GOTO: pos={pos} reason='{imp.reason}'", file=sys.stderr)
                    elif imp.action_type == 'explore_unknown':
                        # Find unvisited corridor cells
                        for r in range(min(60, len(grid))):
                            for c in range(len(grid[0])):
                                if self._smap and self._smap.is_passable(r, c) and (r, c) not in self._visited_positions:
                                    detours.append((r, c))
                                    break
                            if len(detours) > len(goals) + 3:
                                break
                        if detours:
                            print(f"CROSS_EXPLORE: reason='{imp.reason}'", file=sys.stderr)
            except Exception as e:
                print(f"CROSS_SPACE_ERR: {e}", file=sys.stderr)

        # EXISTING BFS PLANNING (with rule detours now included)
        self._planner = RoutePlanner(self._smap, mv_actions)
        route = self._planner.plan_route(self._ctrl_pos, goals, detours, self._ctrl_offsets, budget)
        print(f"PLAN: goals={goals[:3]} detours={detours[:3]} route={len(route)} ctrl={self._ctrl_pos}", file=sys.stderr)
        return route

    def is_done(self, frames, latest_frame):
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames, latest_frame):
        grid = latest_frame.frame[0]
        self._frame += 1
        if self._replan_cooldown > 0: self._replan_cooldown -= 1

        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._full_reset(); return GameAction.RESET

        if latest_frame.levels_completed != self._prev_levels:
            old = self._prev_levels; self._full_reset()
            self._prev_levels = latest_frame.levels_completed
            self.prev_grid = [row[:] for row in grid]
            print(f"LEVEL_UP: {old}→{self._prev_levels}", file=sys.stderr)
            # Don't send an action — let observe phase handle it on next call
            return GameAction.ACTION1

        snap = self._observe(grid)
        budget = self._budget(snap)

        if self._actions > 0:
            ctrl_mv = self._get_ctrl_mv(snap)
            self._find_ctrl(snap)
            self._model.record(self._last_aidx, snap.diff is not None and snap.diff.has_changes, ctrl_mv)
            if self._cross_space and ctrl_mv != (0, 0) and self._ctrl_pos:
                try:
                    color_under = int(np.array(grid)[self._ctrl_pos[0], self._ctrl_pos[1]]) if self._ctrl_pos[0] < 64 and self._ctrl_pos[1] < 64 else 0
                    self._cross_space.record_movement(self._frame, self._ctrl_pos, self._last_aidx, color_under)
                    self._visited_positions.add(self._ctrl_pos)
                    self._stuck_counter = 0
                    self._last_progress_frame = self._frame
                except Exception:
                    pass
            if self._smap and ctrl_mv == (0,0) and self._ctrl_pos:
                known_mv = self._model.get_movement(self._last_aidx)
                if known_mv != (0,0):
                    for dr, dc in self._ctrl_offsets:
                        self._smap.mark_wall(self._ctrl_pos[0]+known_mv[0]+dr, self._ctrl_pos[1]+known_mv[1]+dc)
                    if self._cross_space and self._ctrl_pos:
                        try:
                            wall_r = self._ctrl_pos[0] + known_mv[0]
                            wall_c = self._ctrl_pos[1] + known_mv[1]
                            wall_colors = set()
                            g = np.array(grid)
                            for dr, dc in self._ctrl_offsets:
                                wr, wc = wall_r + dr, wall_c + dc
                                if 0 <= wr < 64 and 0 <= wc < 64:
                                    wall_colors.add(int(g[wr, wc]))
                            self._cross_space.record_blocked(self._frame, self._ctrl_pos, self._last_aidx, wall_colors)
                            self._stuck_counter += 1
                        except Exception:
                            pass
            if self._monitor and self._replan_cooldown <= 0:
                reaction = self._monitor.check(snap)
                if reaction and reaction['total'] >= 3:
                    print(f"REACTION: f={self._frame} {reaction['total']} changes — replanning", file=sys.stderr)
                    route = self._make_plan(grid, snap, budget)
                    self.action_queue = route; self._replan_cooldown = 5
                if reaction and self._reaction_analyzer and self.prev_grid is not None:
                    try:
                        prev_g = np.array(self.prev_grid) if not isinstance(self.prev_grid, np.ndarray) else self.prev_grid
                        curr_g = np.array(grid) if not isinstance(grid, np.ndarray) else np.array(grid)
                        event = self._reaction_analyzer.analyze(
                            prev_grid=prev_g,
                            curr_grid=curr_g,
                            player_pos=self._ctrl_pos or (32, 32),
                            action_idx=self._last_aidx,
                            corridor_colors=self._probe_corridor_colors,
                        )
                        if event and self._rule_learner:
                            # Record what color the player is standing on as potential trigger
                            if self._ctrl_pos and self.prev_grid is not None:
                                pr, pc = self._ctrl_pos
                                if 0 <= pr < 64 and 0 <= pc < 64:
                                    event.trigger_color = int(prev_g[pr, pc])
                            learned = self._rule_learner.observe(event)
                            if learned:
                                print(f"RULE_LEARNED: id={learned.rule_id} type={learned.effect_type} "
                                      f"trigger={learned.trigger_type} confirmed={learned.confirmed} "
                                      f"enables_path={learned.enables_path} "
                                      f"region={learned.trigger_region}", file=sys.stderr)
                                # If wall_opened, mark the opened cells as passable
                                if learned.enables_path and learned.effect_region and self._smap:
                                    er = learned.effect_region
                                    for r in range(max(0,er[0]), min(60,er[2]+1)):
                                        for c in range(max(0,er[1]), min(64,er[3]+1)):
                                            self._smap.mark_passable(r, c)
                    except Exception as e:
                        print(f"RULE_LEARN_ERR: {e}", file=sys.stderr)
                if self._cross_space and reaction and self.prev_grid is not None:
                    try:
                        prev_g = np.array(self.prev_grid) if not isinstance(self.prev_grid, np.ndarray) else self.prev_grid
                        curr_g = np.array(grid)
                        changed = []
                        for r in range(min(60, len(grid))):
                            for c in range(len(grid[0])):
                                if int(prev_g[r][c]) != int(curr_g[r][c]):
                                    changed.append((r, c))
                        color_trans = {}
                        for r, c in changed:
                            key = (int(prev_g[r, c]), int(curr_g[r, c]))
                            color_trans[key] = color_trans.get(key, 0) + 1
                        corridor_colors = self._probe_corridor_colors
                        change_type = 'unknown'
                        for (old, new), cnt in color_trans.items():
                            if old not in corridor_colors and new in corridor_colors:
                                change_type = 'wall_opened'; break
                            if old in corridor_colors and new not in corridor_colors:
                                change_type = 'wall_closed'; break
                        color_under = 0
                        if self._ctrl_pos:
                            pr, pc = self._ctrl_pos
                            if 0 <= pr < 64 and 0 <= pc < 64:
                                color_under = int(prev_g[pr, pc])
                        self._cross_space.record_reaction(
                            self._frame, self._ctrl_pos or (32, 32), self._last_aidx,
                            change_type, color_trans, changed, color_under
                        )
                    except Exception:
                        pass
            if self._model.is_click_game and self._last_click is not None:
                had_effect = snap.diff is not None and snap.diff.has_changes
                self._click.record(self._last_click, had_effect)
                # Learn clickable positions from probe results
                if self._phase == 'click_probe' and self.prev_grid is not None:
                    try:
                        prev_g = np.array(self.prev_grid)
                        curr_g = np.array(grid)
                        cells_changed = int(np.sum(prev_g[:60] != curr_g[:60]))
                        self._click.record_click_result(
                            self._last_click, cells_changed, prev_g, curr_g)
                    except Exception:
                        pass
                # CrossSpace: record click result
                if self._cross_space and had_effect:
                    try:
                        cp = self._last_click
                        colors = set()
                        if self.prev_grid:
                            prev_g = np.array(self.prev_grid)
                            curr_g = np.array(grid)
                            for r in range(max(0,cp[0]-3), min(60,cp[0]+4)):
                                for c in range(max(0,cp[1]-3), min(64,cp[1]+4)):
                                    if int(prev_g[r,c]) != int(curr_g[r,c]):
                                        colors.add(int(prev_g[r,c]))
                                        colors.add(int(curr_g[r,c]))
                        self._cross_space.record_reaction(
                            self._frame, cp, 5, 'click_response',
                            {}, [], int(curr_g[cp[0],cp[1]]) if cp[0]<64 and cp[1]<64 else 0
                        )
                        # Remember effective click position
                        if not hasattr(self._click, '_effective_positions'):
                            self._click._effective_positions = []
                        self._click._effective_positions.append(cp)
                        print(f"CLICK_EFFECT: pos={cp} colors={colors}", file=sys.stderr)
                    except Exception:
                        pass
                self._last_click = None

        # OBSERVE
        if self._phase == 'observe':
            available = latest_frame.available_actions or ALL_ACTIONS[:4]
            # Handle both GameAction objects and raw ints
            indices = []
            for a in available:
                if a in ACTION_IDS:
                    indices.append(ACTION_IDS[a])
                elif isinstance(a, int) and 1 <= a <= 7:
                    indices.append(a - 1)  # ACTION1=0, ACTION6=5, etc.
            if not indices:
                indices = list(range(4))
            self._model.set_available(indices)
            self._monitor = DiffMonitor(snap)
            # v25: initial structure analysis
            if HAS_V25:
                try:
                    self._v25_smap = StructureAnalyzer().analyze(grid)
                    self._v25_monitor = LiveMonitor(grid, self._v25_smap.cross_signature)
                    print(f"V26_STRUCT: {self._v25_smap.cross_signature}", file=sys.stderr)
                except Exception as e:
                    print(f"V26_STRUCT_ERR: {e}", file=sys.stderr)
            print(f"OBSERVE: objs={len(snap.objects)} colors={snap.panorama.get('unique_colors',0)} "
                  f"rare={snap.panorama.get('rare_colors',[])} anom={len(snap.anomalies)} budget={budget}", file=sys.stderr)
            # Click-only games: try pattern match first, fall back to probe
            if self._model.is_click_game:
                try:
                    pattern_plan = self._click.plan_pattern_match(grid)
                    if pattern_plan:
                        print(f"CLICK_GAME: pattern match found, skipping probe", file=sys.stderr)
                        self._phase = 'plan'
                    else:
                        # Use v25 StructureAnalyzer interactive positions first (0 probe cost)
                        if HAS_V25 and self._v25_smap and self._v25_smap.interactive_objects:
                            struct_clicks = [(int(r.centroid[0]), int(r.centroid[1])) 
                                           for r in self._v25_smap.interactive_objects
                                           if r.cell_count >= 10]
                            if struct_clicks:
                                self._click._queue = struct_clicks
                                print(f"CLICK_GAME: {len(struct_clicks)} positions from StructureAnalyzer, no probe needed", file=sys.stderr)
                                self._phase = 'plan'
                            else:
                                print(f"CLICK_GAME: starting click probe", file=sys.stderr)
                                self._click.plan_click_probe(grid)
                                self._phase = 'click_probe'
                        else:
                            print(f"CLICK_GAME: starting click probe", file=sys.stderr)
                            self._click.plan_click_probe(grid)
                            self._phase = 'click_probe'
                except Exception:
                    print(f"CLICK_GAME: starting click probe", file=sys.stderr)
                    self._click.plan_click_probe(grid)
                    self._phase = 'click_probe'
            else:
                self._probe_queue = list(indices); self._phase = 'probe'

        # PROBE
        if self._phase == 'probe':
            if self._probe_queue:
                idx = self._probe_queue.pop(0); self._last_aidx = idx
                self._prev_ctrl = self._ctrl_pos; self._actions += 1
                self.prev_grid = [row[:] for row in grid]
                if self._model.is_click_game and idx == 5:
                    self._click.plan(snap, self.sensor)
                    cp = self._click.next()
                    if cp:
                        self._last_click = cp; a = GameAction.ACTION6; a.set_data({"x": cp[1], "y": cp[0]})
                        a.reasoning = f"probe click={cp}"; return a
                a = ALL_ACTIONS[idx] if idx < len(ALL_ACTIONS) else GameAction.ACTION1
                a.reasoning = f"probe idx={idx}"; return a
            else:
                print(f"PROBE_DONE: {self._model.summary()}", file=sys.stderr)
                # v26: promote corridor colors to full map
                if self._probe_corridor_colors and self._smap:
                    for color in self._probe_corridor_colors:
                        promoted = 0
                        for r in range(min(60, self._smap.rows)):
                            for c in range(self._smap.cols):
                                if int(self._smap.g[r, c]) == color:
                                    self._smap.mark_passable(r, c)
                                    promoted += 1
                        if promoted > 0:
                            print(f"V26_PROMOTE: color={color} → {promoted} cells", file=sys.stderr)
                # v26: also promote from v25 structure analyzer
                if HAS_V25 and self._v25_smap:
                    try:
                        for reg in self._v25_smap.regions:
                            if reg.role == 'corridor' and self._smap:
                                color = reg.color
                                if color not in self._probe_corridor_colors:
                                    promoted = 0
                                    for r in range(min(60, self._smap.rows)):
                                        for c in range(self._smap.cols):
                                            if int(self._smap.g[r, c]) == color:
                                                self._smap.mark_passable(r, c)
                                                promoted += 1
                                    if promoted > 0:
                                        print(f"V26_PROMOTE_V25: color={color} → {promoted} cells", file=sys.stderr)
                    except Exception:
                        pass
                self._phase = 'plan'

        # CLICK_PROBE: discover clickable positions
        if self._phase == 'click_probe':
            cp = self._click.next()
            if cp:
                self._last_click = cp
                self._last_aidx = 5
                self._actions += 1
                self.prev_grid = [row[:] for row in grid]
                a = GameAction.ACTION6
                a.set_data({"x": cp[1], "y": cp[0]})
                a.reasoning = f"click_probe pos={cp}"
                return a
            else:
                # Probe done, check results
                if self._click._clickable_positions:
                    print(f"CLICK_PROBE_DONE: found {len(self._click._clickable_positions)} clickable positions", file=sys.stderr)
                    self._click.plan_from_discovered(grid)
                else:
                    print(f"CLICK_PROBE_DONE: no clickable positions found, fallback", file=sys.stderr)
                    self._click.plan(snap, self.sensor)
                self._phase = 'execute'

        # PLAN
        if self._phase == 'plan':
            if self._model.is_click_game:
                # For click games: build a grid scan plan
                # Click every non-background block systematically
                try:
                    g = np.array(grid)
                    bg_color = int(Counter(int(v) for v in g[:60].flatten()).most_common(1)[0][0])
                    # Find all distinct color blocks (non-bg, 3x3 grid scan)
                    click_targets = []
                    visited_blocks = set()
                    for r in range(2, 58, 3):
                        for c in range(2, 62, 3):
                            color = int(g[r, c])
                            if color != bg_color:
                                # Check if this is a new block (not too close to existing)
                                too_close = False
                                for pr, pc in visited_blocks:
                                    if abs(r-pr) < 4 and abs(c-pc) < 4:
                                        too_close = True; break
                                if not too_close:
                                    click_targets.append((r, c))
                                    visited_blocks.add((r, c))
                    self._click._queue = click_targets
                    print(f"CLICK_SCAN: {len(click_targets)} positions to probe", file=sys.stderr)
                except Exception as e:
                    print(f"CLICK_SCAN_ERR: {e}", file=sys.stderr)
                    self._click_targets = self._click.plan(snap, self.sensor)
            else:
                route = self._make_plan(grid, snap, budget)
                self.action_queue = route
            self._phase = 'execute'

        # EXECUTE — queue empty → replan
        if self._phase == 'execute' and not self.action_queue and not self._model.is_click_game:
            if self._replan_cooldown <= 0:
                goals, detours = self._classify_targets(snap)
                if self._stuck_counter >= 10 and self._cross_space:
                    try:
                        if self._ctrl_pos:
                            wall_colors = set()
                            g = np.array(grid)
                            for aidx, (dr, dc) in self._model.get_mv_actions().items():
                                wr, wc = self._ctrl_pos[0] + dr, self._ctrl_pos[1] + dc
                                if 0 <= wr < 64 and 0 <= wc < 64:
                                    wall_colors.add(int(g[wr, wc]))
                            targets = self._cross_space.get_seek_targets(grid, self._ctrl_pos, wall_colors)
                            if targets:
                                for t in targets[:3]:
                                    if t not in goals:
                                        detours.insert(0, t)
                                print(f"CROSS_STUCK: {len(targets)} targets from resonance", file=sys.stderr)
                                self._stuck_counter = 0  # reset
                    except Exception:
                        pass
                route = self._make_plan(grid, snap, budget)
                self.action_queue = route
                self._replan_cooldown = max(len(route), 3)

        self._prev_ctrl = self._ctrl_pos
        self.prev_grid = [row[:] for row in grid]; self._actions += 1

        if self._model.is_click_game:
            cp = self._click.next()
            if cp:
                self._last_click = cp; self._last_aidx = 5
                a = GameAction.ACTION6; a.set_data({"x": cp[1], "y": cp[0]})
                a.reasoning = f"exec click={cp} left={self._click.remaining}"; return a
            else:
                # Pattern match mode: re-scan for remaining mismatches
                if self._click._pattern_match_mode:
                    try:
                        new_plan = self._click.plan_pattern_match(grid)
                        if new_plan:
                            cp = self._click.next()
                            if cp:
                                self._last_click = cp; self._last_aidx = 5
                                a = GameAction.ACTION6; a.set_data({"x": cp[1], "y": cp[0]})
                                a.reasoning = f"exec click rescan={cp}"; return a
                    except Exception:
                        pass
                # Replan: prioritize known effective positions
                effective = getattr(self._click, '_effective_positions', [])
                if effective:
                    # Re-click effective positions (they toggle colors)
                    self._click._queue = list(effective)
                    cp = self._click.next()
                    if cp:
                        self._last_click = cp; self._last_aidx = 5
                        a = GameAction.ACTION6; a.set_data({"x": cp[1], "y": cp[0]})
                        a.reasoning = f"exec click effective={cp}"; return a
                # Try grid scan for new positions
                try:
                    g = np.array(grid)
                    bg_color = int(Counter(int(v) for v in g[:60].flatten()).most_common(1)[0][0])
                    scan_targets = []
                    for r in range(2, 58, 3):
                        for c in range(2, 62, 3):
                            if int(g[r, c]) != bg_color:
                                if (r,c) not in set(effective):
                                    scan_targets.append((r, c))
                    if scan_targets:
                        self._click._queue = scan_targets[:20]
                        cp = self._click.next()
                        if cp:
                            self._last_click = cp; self._last_aidx = 5
                            a = GameAction.ACTION6; a.set_data({"x": cp[1], "y": cp[0]})
                            a.reasoning = f"exec click scan"; return a
                except Exception:
                    pass
                a = GameAction.ACTION6; a.set_data({"x": 32, "y": 32}); a.reasoning = "fallback"; return a
        else:
            if self.action_queue:
                aidx = self.action_queue.pop(0)
            else:
                mvs = self._model.get_mv_actions()
                aidx = list(mvs.keys())[0] if mvs else 0
            self._last_aidx = aidx
            a = ALL_ACTIONS[aidx] if aidx < len(ALL_ACTIONS) else GameAction.ACTION1
            a.reasoning = f"exec q={len(self.action_queue)} act={self._actions} ctrl={self._ctrl_pos}"
            return a
