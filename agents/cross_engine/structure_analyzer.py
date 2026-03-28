"""
structure_analyzer.py — Cross Structural Analyzer v25

Converts a raw 64x64 ARC-AGI-3 grid into a StructuralMap via:
  1. Flood-fill connected components (4-connectivity)
  2. Enclosure detection (囲み構造) — BFS from region boundary
  3. Continuity-based role assignment (wall / corridor / player / target / etc.)
  4. Compact cross_signature for frame-to-frame diffing

No LLM. No game-specific logic. All roles emerge from structure.
"""

from __future__ import annotations

import numpy as np
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any


# ─── Data classes ────────────────────────────────────────────────────────────

ROLES = (
    'wall', 'corridor', 'player', 'target',
    'interactive', 'timer', 'pattern', 'lock', 'unknown',
)


@dataclass
class Region:
    """One connected component of the grid."""
    region_id: int
    color: int
    cells: Set[Tuple[int, int]]
    role: str = 'unknown'

    # Derived geometry (filled by StructureAnalyzer)
    cell_count: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)   # r_min, c_min, r_max, c_max
    centroid: Tuple[float, float] = (0.0, 0.0)
    perimeter_cells: Set[Tuple[int, int]] = field(default_factory=set)
    interior_cells: Set[Tuple[int, int]] = field(default_factory=set)

    # Structural flags
    touches_edge: bool = False
    is_enclosed: bool = False
    enclosed_by_color: Optional[int] = None
    contains_regions: List[int] = field(default_factory=list)
    contained_by: List[int] = field(default_factory=list)

    # All colors present in this region (used for multi-color player tracking)
    colors: Set[int] = field(default_factory=set)

    # Extra properties
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.cell_count = len(self.cells)
        if self.cells:
            rs = [r for r, c in self.cells]
            cs = [c for r, c in self.cells]
            self.bbox = (min(rs), min(cs), max(rs), max(cs))
            self.centroid = (sum(rs) / len(rs), sum(cs) / len(cs))


@dataclass
class StructuralMap:
    """Full structural interpretation of one grid frame."""
    grid: np.ndarray                          # raw 64×64
    role: np.ndarray                          # role[r][c] string
    regions: List[Region]
    color_to_regions: Dict[int, List[int]]    # color → list of region_ids

    player_region: Optional[Region] = None
    goal_candidates: List[Region] = field(default_factory=list)
    interactive_objects: List[Region] = field(default_factory=list)
    timer_region: Optional[Region] = None

    cross_signature: Dict[str, Any] = field(default_factory=dict)

    def region_at(self, r: int, c: int) -> Optional[Region]:
        """Return the Region that contains cell (r, c), or None."""
        color = int(self.grid[r, c])
        for rid in self.color_to_regions.get(color, []):
            if (r, c) in self.regions[rid].cells:
                return self.regions[rid]
        return None

    def is_passable(self, r: int, c: int) -> bool:
        if r < 0 or r >= 64 or c < 0 or c >= 64:
            return False
        role = str(self.role[r, c])
        return role not in ('wall', 'timer')

    def mark_wall(self, r: int, c: int) -> None:
        if 0 <= r < 64 and 0 <= c < 64:
            self.role[r, c] = 'wall'

    def mark_corridor(self, r: int, c: int) -> None:
        if 0 <= r < 64 and 0 <= c < 64:
            if str(self.role[r, c]) not in ('timer',):
                self.role[r, c] = 'corridor'

    def promote_color_to_corridor(self, color: int) -> int:
        """Mark ALL cells of given color (in play area) as corridor. Returns count."""
        count = 0
        rows, cols = self.grid.shape
        for r in range(min(rows, 60)):
            for c in range(cols):
                if int(self.grid[r, c]) == color and str(self.role[r, c]) in ('wall', 'unknown', 'lock', 'pattern', 'border'):
                    self.role[r, c] = 'corridor'
                    count += 1
        return count


# ─── Main analyzer ───────────────────────────────────────────────────────────

class StructureAnalyzer:
    """
    Converts a raw 64×64 grid into a StructuralMap.

    Usage:
        analyzer = StructureAnalyzer()
        smap = analyzer.analyze(grid)
    """

    GRID_SIZE = 64
    TIMER_ROW_START = 60   # rows 60-63 are typically the timer bar

    def analyze(self, grid) -> StructuralMap:
        g = np.array(grid, dtype=np.int32)
        rows, cols = g.shape

        # 1. Flood-fill → regions
        regions = self._flood_fill(g, rows, cols)

        # 2. Geometry
        for reg in regions:
            self._compute_geometry(reg, rows, cols)

        # 3. Global color stats
        total_cells = rows * cols
        color_freq: Dict[int, float] = {}
        for color, count in Counter(int(v) for v in g.flatten()).items():
            color_freq[color] = count / total_cells

        # 4. Enclosure detection per region
        self._detect_enclosures(regions, g, rows, cols)

        # 5. Role assignment
        self._assign_roles(regions, color_freq, rows, cols)

        # 6. Build role grid
        role_grid = np.full((rows, cols), 'unknown', dtype=object)
        color_to_regions: Dict[int, List[int]] = {}
        for reg in regions:
            for r, c in reg.cells:
                role_grid[r, c] = reg.role
            color_to_regions.setdefault(reg.color, []).append(reg.region_id)

        # 7. Classify structural slots
        player_region = self._find_player(regions, color_freq)
        goal_candidates = self._find_goals(regions)
        interactive_objects = self._find_interactives(regions)
        timer_region = self._find_timer(regions)

        if player_region:
            player_region.role = 'player'
            for r, c in player_region.cells:
                role_grid[r, c] = 'player'

        # 8. Cross signature
        sig = self._build_signature(regions, player_region, color_freq)

        smap = StructuralMap(
            grid=g,
            role=role_grid,
            regions=regions,
            color_to_regions=color_to_regions,
            player_region=player_region,
            goal_candidates=goal_candidates,
            interactive_objects=interactive_objects,
            timer_region=timer_region,
            cross_signature=sig,
        )
        return smap

    # ── Phase 1: Flood-fill ──────────────────────────────────────────────────

    def _flood_fill(self, g: np.ndarray, rows: int, cols: int) -> List[Region]:
        visited = np.zeros((rows, cols), dtype=bool)
        regions: List[Region] = []
        rid = 0

        for r in range(rows):
            for c in range(cols):
                if visited[r, c]:
                    continue
                color = int(g[r, c])
                cells: Set[Tuple[int, int]] = set()
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr, cc]:
                        continue
                    if int(g[cr, cc]) != color:
                        continue
                    visited[cr, cc] = True
                    cells.add((cr, cc))
                    stack.extend([(cr - 1, cc), (cr + 1, cc),
                                   (cr, cc - 1), (cr, cc + 1)])

                if cells:
                    regions.append(Region(region_id=rid, color=color, cells=cells))
                    rid += 1

        return regions

    # ── Phase 2: Geometry ────────────────────────────────────────────────────

    def _compute_geometry(self, reg: Region, rows: int, cols: int) -> None:
        cells = reg.cells
        reg.cell_count = len(cells)

        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        reg.bbox = (min(rs), min(cs), max(rs), max(cs))
        reg.centroid = (sum(rs) / len(rs), sum(cs) / len(cs))

        # Edge touch
        reg.touches_edge = any(
            r == 0 or r == rows - 1 or c == 0 or c == cols - 1
            for r, c in cells
        )

        # Perimeter vs interior
        perimeter: Set[Tuple[int, int]] = set()
        interior: Set[Tuple[int, int]] = set()
        for r, c in cells:
            is_border = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cells:
                    is_border = True
                    break
            if is_border:
                perimeter.add((r, c))
            else:
                interior.add((r, c))
        reg.perimeter_cells = perimeter
        reg.interior_cells = interior

    # ── Phase 3: Enclosure detection ─────────────────────────────────────────

    def _detect_enclosures(self, regions: List[Region], g: np.ndarray,
                            rows: int, cols: int) -> None:
        """For each region, test if it's enclosed by a different color."""
        for reg in regions:
            if reg.touches_edge:
                reg.is_enclosed = False
                continue

            # Try each adjacent color as potential encloser
            adj_colors: Set[int] = set()
            for r, c in reg.perimeter_cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        adj_color = int(g[nr, nc])
                        if adj_color != reg.color:
                            adj_colors.add(adj_color)

            for enc_color in adj_colors:
                if _is_enclosed(reg.cells, enc_color, g, rows, cols):
                    reg.is_enclosed = True
                    reg.enclosed_by_color = enc_color
                    break

    # ── Phase 4: Role assignment ──────────────────────────────────────────────

    def _assign_roles(self, regions: List[Region], color_freq: Dict[int, float],
                      rows: int, cols: int) -> None:
        """Assign roles based on structure, not game knowledge."""
        for reg in regions:
            reg.role = self._infer_role(reg, color_freq, rows, cols)

    def _infer_role(self, reg: Region, color_freq: Dict[int, float],
                    rows: int, cols: int) -> str:
        freq = color_freq.get(reg.color, 0)
        n = reg.cell_count
        r_min, c_min, r_max, c_max = reg.bbox
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        bbox_area = height * width
        fill_ratio = n / bbox_area if bbox_area > 0 else 1.0
        aspect = width / height if height > 0 else 1.0

        # Timer area heuristic: rows 60-63
        centroid_r = reg.centroid[0]
        if centroid_r >= self.TIMER_ROW_START:
            return 'timer'

        # Large region touching edges → wall/background
        if reg.touches_edge and freq > 0.08:
            return 'wall'

        # Thin linear region (not touching edge, not timer) → border/timer
        is_thin = (height <= 3 and width > 20) or (width <= 3 and height > 20)
        if is_thin and not reg.is_enclosed:
            return 'timer' if centroid_r > rows * 0.7 else 'wall'

        # Enclosed regions
        if reg.is_enclosed:
            if n <= 30:
                # Small enclosed → interactive, marker, target
                if freq < 0.005:
                    return 'player'    # very rare color, small, enclosed
                return 'interactive'
            elif n <= 300:
                return 'pattern'
            else:
                return 'corridor'

        # Large non-edge, not enclosed → wall or corridor
        if freq > 0.05:
            return 'wall'

        # Medium non-enclosed, rare color → lock or interactive
        if freq < 0.01 and n <= 200:
            return 'lock'

        # Contained by bbox of another large region → pattern
        return 'unknown'

    # ── Phase 5: Find structural slots ───────────────────────────────────────

    def _find_player(self, regions: List[Region],
                     color_freq: Dict[int, float]) -> Optional[Region]:
        """
        Player detection using v24-style logic:
        1. Find rare-color, small-medium regions NOT in timer area
        2. Prefer multi-color blocks (two rare regions adjacent to each other)
        3. Return the combined block or the best single candidate
        """
        # Build adjacency: which regions share border cells?
        region_adjacency: Dict[int, Set[int]] = {i: set() for i in range(len(regions))}
        cell_to_region: Dict[Tuple[int, int], int] = {}
        for idx, reg in enumerate(regions):
            for r, c in reg.cells:
                cell_to_region[(r, c)] = idx

        for idx, reg in enumerate(regions):
            for r, c in reg.cells:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2)]:
                    nr, nc = r + dr, c + dc
                    nidx = cell_to_region.get((nr, nc))
                    if nidx is not None and nidx != idx:
                        region_adjacency[idx].add(nidx)

        # Phase 1: Find rare, small-medium, not-timer regions
        rare_candidates = []
        for idx, reg in enumerate(regions):
            if reg.role == 'timer':
                continue
            # Check if centroid is in timer area
            cr = reg.centroid[0] if reg.centroid else 32
            if cr >= 58:
                continue
            freq = color_freq.get(reg.color, 0)
            n = reg.cell_count
            if freq < 0.03 and 3 <= n <= 200 and not reg.touches_edge:
                rare_candidates.append((idx, reg, freq))

        if not rare_candidates:
            return None

        # Phase 2: Find pairs of adjacent rare regions (multi-color player block)
        best_pair = None
        best_pair_score = -1
        for i, (idx_a, reg_a, freq_a) in enumerate(rare_candidates):
            for idx_b, reg_b, freq_b in rare_candidates[i+1:]:
                if idx_b in region_adjacency.get(idx_a, set()):
                    # Adjacent pair of rare regions → likely player
                    combined_size = reg_a.cell_count + reg_b.cell_count
                    if 5 <= combined_size <= 200:
                        # Improved scoring: prefer mid-bottom, compact, enclosed
                        cr = (reg_a.centroid[0] + reg_b.centroid[0]) / 2
                        position_score = 1.0 if 20 < cr < 55 else 0.3  # penalize extreme top/bottom
                        r_min, c_min, r_max, c_max = self._merge_bbox(reg_a.bbox, reg_b.bbox)
                        bbox_w = c_max - c_min + 1
                        bbox_h = r_max - r_min + 1
                        compactness = min(bbox_w, bbox_h) / max(bbox_w, bbox_h, 1)  # 1.0 = square
                        enclosure_bonus = 2.0 if (reg_a.is_enclosed or reg_b.is_enclosed) else 1.0
                        rarity_score = (1.0 / max(freq_a, 0.001) + 1.0 / max(freq_b, 0.001))
                        score = rarity_score * position_score * compactness * enclosure_bonus / combined_size
                        if score > best_pair_score:
                            best_pair_score = score
                            # Merge into the larger region
                            if reg_a.cell_count >= reg_b.cell_count:
                                merged = Region(region_id=-1, 
                                    color=reg_a.color,
                                    cells=reg_a.cells | reg_b.cells,
                                    cell_count=combined_size,
                                    bbox=self._merge_bbox(reg_a.bbox, reg_b.bbox),
                                    centroid=self._merge_centroid(reg_a, reg_b),
                                    perimeter_cells=reg_a.perimeter_cells | reg_b.perimeter_cells,
                                    interior_cells=reg_a.interior_cells | reg_b.interior_cells,
                                    touches_edge=False,
                                    is_enclosed=reg_a.is_enclosed or reg_b.is_enclosed,
                                    enclosed_by_color=reg_a.enclosed_by_color,
                                    role='player',
                                )
                            else:
                                merged = Region(region_id=-1, 
                                    color=reg_b.color,
                                    cells=reg_a.cells | reg_b.cells,
                                    cell_count=combined_size,
                                    bbox=self._merge_bbox(reg_a.bbox, reg_b.bbox),
                                    centroid=self._merge_centroid(reg_a, reg_b),
                                    perimeter_cells=reg_a.perimeter_cells | reg_b.perimeter_cells,
                                    interior_cells=reg_a.interior_cells | reg_b.interior_cells,
                                    touches_edge=False,
                                    is_enclosed=reg_b.is_enclosed or reg_a.is_enclosed,
                                    enclosed_by_color=reg_b.enclosed_by_color,
                                    role='player',
                                )
                            # Store both colors for multi-color player tracking
                            merged.colors = {reg_a.color, reg_b.color}
                            best_pair = merged

        if best_pair:
            return best_pair

        # Phase 3: Single best rare region (fallback)
        rare_candidates.sort(key=lambda x: (-x[1].cell_count, x[2]))
        for idx, reg, freq in rare_candidates:
            if reg.cell_count >= 5:
                return reg

        # Phase 4: Any rare region
        if rare_candidates:
            return rare_candidates[0][1]
        return None

    def _merge_bbox(self, bbox_a, bbox_b):
        return (
            min(bbox_a[0], bbox_b[0]),
            min(bbox_a[1], bbox_b[1]),
            max(bbox_a[2], bbox_b[2]),
            max(bbox_a[3], bbox_b[3]),
        )

    def _merge_centroid(self, reg_a, reg_b):
        all_cells = list(reg_a.cells) + list(reg_b.cells)
        rs = [r for r, c in all_cells]
        cs = [c for r, c in all_cells]
        return (sum(rs) / len(rs), sum(cs) / len(cs))

    def _find_goals(self, regions: List[Region]) -> List[Region]:
        """Goals: locked regions, patterns, large enclosed areas."""
        goals = []
        for reg in regions:
            if reg.role in ('lock', 'pattern', 'target'):
                goals.append(reg)
            elif reg.role == 'corridor' and reg.cell_count > 200:
                goals.append(reg)
        # Sort by structural importance (lock > pattern > corridor)
        priority = {'lock': 3, 'target': 2, 'pattern': 1, 'corridor': 0}
        goals.sort(key=lambda r: -priority.get(r.role, 0))
        return goals

    def _find_interactives(self, regions: List[Region]) -> List[Region]:
        """Interactive objects: small enclosed regions, rare colors."""
        return [r for r in regions if r.role == 'interactive']

    def _find_timer(self, regions: List[Region]) -> Optional[Region]:
        """Timer: thin horizontal bar in bottom rows."""
        timers = [r for r in regions if r.role == 'timer']
        if timers:
            # Largest timer region
            return max(timers, key=lambda r: r.cell_count)
        return None

    # ── Phase 6: Cross signature ──────────────────────────────────────────────

    def _build_signature(self, regions: List[Region],
                         player_region: Optional[Region],
                         color_freq: Dict[int, float]) -> Dict[str, Any]:
        role_counts: Dict[str, int] = Counter(r.role for r in regions)
        player_pos = None
        if player_region:
            player_pos = (int(player_region.centroid[0]),
                          int(player_region.centroid[1]))

        top_colors = sorted(color_freq.items(), key=lambda x: -x[1])[:10]
        return {
            'total_regions': len(regions),
            'role_counts': dict(role_counts),
            'player_pos': player_pos,
            'top_colors': [(c, round(f, 4)) for c, f in top_colors],
            'color_count': len(color_freq),
        }


# ─── Standalone enclosure helper ─────────────────────────────────────────────

def _is_enclosed(region_cells: Set[Tuple[int, int]],
                  enclosing_color: int,
                  grid: np.ndarray,
                  rows: int, cols: int) -> bool:
    """
    BFS from cells adjacent to region_cells.
    If we reach a grid edge without crossing enclosing_color → NOT enclosed.
    If BFS exhausts → enclosed.
    """
    visited = set(region_cells)
    queue: deque = deque()

    # Seed from cells adjacent to the region (that aren't the enclosing color)
    for r, c in region_cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and 0 <= nr < rows and 0 <= nc < cols:
                if int(grid[nr, nc]) != enclosing_color:
                    queue.append((nr, nc))
                    visited.add((nr, nc))

    while queue:
        r, c = queue.popleft()
        # Reached edge → not enclosed
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            return False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and 0 <= nr < rows and 0 <= nc < cols:
                if int(grid[nr, nc]) != enclosing_color:
                    queue.append((nr, nc))
                    visited.add((nr, nc))

    return True  # couldn't reach edge → enclosed
