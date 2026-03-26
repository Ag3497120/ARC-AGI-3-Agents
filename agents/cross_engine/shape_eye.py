"""
shape_eye.py — Multi-Scale Shape Recognition

Looks at the grid at multiple scales and finds meaningful shapes.
Used during Phase 2 of CrossResonanceAgent to detect patterns, similar shapes,
and key-lock relationships before rule discovery.
"""

from __future__ import annotations
from collections import deque
from typing import List, Tuple, Dict, Any, Optional


class ShapeEye:
    """Looks at the grid at multiple scales and finds meaningful shapes."""

    def __init__(self, grid: list):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0

    # ── Scale Scanning ─────────────────────────────────────────────────────────

    def scan_at_scale(self, window_size: int) -> List[Tuple[int, int, tuple]]:
        """Slide a window_size × window_size window across the grid.
        Returns (row, col, pattern) tuples for windows that are NOT all one color.
        pattern is a flat tuple of colors (row-major) for the window."""
        results = []
        half = window_size // 2
        for r in range(0, self.rows - window_size + 1, max(1, window_size // 2)):
            for c in range(0, self.cols - window_size + 1, max(1, window_size // 2)):
                window = []
                for dr in range(window_size):
                    for dc in range(window_size):
                        window.append(self.grid[r + dr][c + dc])
                # Skip boring (all same color) windows
                if len(set(window)) > 1:
                    results.append((r, c, tuple(window)))
        return results

    # ── Shape Extraction ───────────────────────────────────────────────────────

    def extract_shapes(self) -> List[Dict[str, Any]]:
        """Find all distinct color-connected shapes in the grid.
        Each shape = {color, cells, bbox, normalized_pattern}
        normalized_pattern = the shape translated to (0,0) origin."""
        visited = [[False] * self.cols for _ in range(self.rows)]
        shapes = []

        for r in range(self.rows):
            for c in range(self.cols):
                color = self.grid[r][c]
                if visited[r][c]:
                    continue
                # Skip background colors and timer rows
                if color in (0, 3) or r >= 60:
                    visited[r][c] = True
                    continue

                # BFS flood-fill
                cells = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < self.rows and 0 <= nc < self.cols
                                and not visited[nr][nc]
                                and self.grid[nr][nc] == color):
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                # Skip tiny shapes (noise)
                if len(cells) < 2:
                    continue

                # Bounding box
                r_min = min(r_ for r_, _ in cells)
                r_max = max(r_ for r_, _ in cells)
                c_min = min(c_ for _, c_ in cells)
                c_max = max(c_ for _, c_ in cells)
                bbox = (r_min, c_min, r_max, c_max)

                # Normalized pattern (translate so top-left = (0,0))
                normalized = frozenset((r_ - r_min, c_ - c_min) for r_, c_ in cells)

                shapes.append({
                    'color': color,
                    'cells': cells,
                    'bbox': bbox,
                    'normalized_pattern': normalized,
                    'size': len(cells),
                    'height': r_max - r_min + 1,
                    'width': c_max - c_min + 1,
                })

        return shapes

    # ── Similarity Analysis ────────────────────────────────────────────────────

    def find_similar_shapes(self, threshold: float = 0.7) -> List[Tuple[Dict, Dict, float]]:
        """Find pairs of shapes that look similar (normalized pattern comparison).
        Returns [(shape_a, shape_b, similarity_score), ...]"""
        shapes = self.extract_shapes()
        pairs = []

        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                a = shapes[i]
                b = shapes[j]

                # Only compare shapes of same color (potential key/lock pairs)
                # or same dimensions
                same_color = a['color'] == b['color']
                same_size = abs(a['size'] - b['size']) <= max(2, a['size'] * 0.2)

                if not (same_color or same_size):
                    continue

                score = self._pattern_similarity(a['normalized_pattern'], b['normalized_pattern'])
                if score >= threshold:
                    pairs.append((a, b, score))

        return sorted(pairs, key=lambda x: -x[2])

    def _pattern_similarity(self, pat_a: frozenset, pat_b: frozenset) -> float:
        """Compute Jaccard similarity between two normalized patterns.
        Also tries rotations and reflections for the best match."""
        if not pat_a and not pat_b:
            return 1.0
        if not pat_a or not pat_b:
            return 0.0

        # Canonical form: normalize to (0,0) origin
        def normalize(pat):
            if not pat:
                return frozenset()
            r_min = min(r for r, c in pat)
            c_min = min(c for r, c in pat)
            return frozenset((r - r_min, c - c_min) for r, c in pat)

        # Try 4 rotations × 2 reflections = 8 transforms
        def transforms(pat):
            cells = list(pat)
            results = []
            # rotations
            cur = cells
            for _ in range(4):
                results.append(normalize(frozenset(cur)))
                cur = [(-c, r) for r, c in cur]
            # reflections
            cur = [(-r, c) for r, c in cells]
            for _ in range(4):
                results.append(normalize(frozenset(cur)))
                cur = [(-c, r) for r, c in cur]
            return results

        best = 0.0
        for variant in transforms(pat_a):
            intersection = len(variant & pat_b)
            union = len(variant | pat_b)
            if union > 0:
                score = intersection / union
                best = max(best, score)

        return best

    def match_pattern_to_hole(self, pattern_shape: Dict, hole_shape: Dict) -> float:
        """Check if pattern_shape fits into hole_shape.
        Like a key fitting into a lock.
        Returns 0.0–1.0 match score."""
        pat = pattern_shape.get('normalized_pattern', frozenset())
        hole = hole_shape.get('normalized_pattern', frozenset())
        return self._pattern_similarity(pat, hole)

    # ── Multi-Scale Analysis ───────────────────────────────────────────────────

    def multi_scale_analysis(self) -> Dict[int, List]:
        """Run analysis at scales 4, 8, 16, 32, 64.
        Returns {scale: [shapes_or_windows]} for each scale."""
        results = {}
        for scale in [4, 8, 16, 32, 64]:
            if scale < min(self.rows, self.cols):
                results[scale] = self.scan_at_scale(scale)
            else:
                results[scale] = []
        return results

    # ── Convenience Helpers ────────────────────────────────────────────────────

    def detect_player_and_lock(self) -> Dict[str, Optional[Dict]]:
        """High-level helper: find likely player and lock shapes.
        Player = shape containing color 12.
        Lock = enclosed region bordered by color 5 cells."""
        shapes = self.extract_shapes()

        player_shape = None
        lock_shape = None

        # Player: color 12 shape (top of player block)
        c12_shapes = [s for s in shapes if s['color'] == 12]
        if c12_shapes:
            player_shape = max(c12_shapes, key=lambda s: s['size'])

        # Lock: shape enclosed by color 5 borders
        c5_shapes = [s for s in shapes if s['color'] == 5]
        if c5_shapes:
            # The lock is typically in the upper region of the grid
            upper_c5 = [s for s in c5_shapes if s['bbox'][0] < self.rows // 2]
            if upper_c5:
                lock_shape = max(upper_c5, key=lambda s: s['size'])

        return {'player': player_shape, 'lock': lock_shape}

    def find_key_lock_pairs(self) -> List[Tuple[Dict, Dict, float]]:
        """Find shape pairs that could be key-lock relationships.
        Returns [(key_shape, lock_shape, confidence), ...]"""
        info = self.detect_player_and_lock()
        player = info.get('player')
        lock = info.get('lock')

        if player is None or lock is None:
            return []

        # Look for color 9 shapes (pattern cells) in both player and lock areas
        shapes = self.extract_shapes()
        player_bbox = player['bbox']
        lock_bbox = lock['bbox']

        player_c9 = [s for s in shapes if s['color'] == 9
                     and s['bbox'][0] >= player_bbox[0] - 2
                     and s['bbox'][0] <= player_bbox[2] + 2]

        lock_c9 = [s for s in shapes if s['color'] == 9
                   and s['bbox'][0] >= lock_bbox[0] - 2
                   and s['bbox'][0] <= lock_bbox[2] + 2]

        pairs = []
        for ps in player_c9:
            for ls in lock_c9:
                score = self.match_pattern_to_hole(ps, ls)
                pairs.append((ps, ls, score))

        return sorted(pairs, key=lambda x: -x[2])
