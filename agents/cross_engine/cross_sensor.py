"""
cross_sensor.py — Panoramic Cross Sensor

Converts the entire 64x64 game board into Cross structure descriptors.
No hardcoded labels. Roles emerge from structure.

6-axis Cross descriptor per object:
  1. Color axis — constituent colors, ratios, rarity in the grid
  2. Shape axis — geometry (rectangularity, aspect ratio, symmetry, type classification)
  3. Position axis — centroid, grid quadrant, relative position
  4. Scale axis — cell count, bounding box area, fill ratio
  5. Relation axis — adjacency to other objects, containment, proximity
  6. Temporal axis — movement, color change, appear/disappear across frames
"""

from __future__ import annotations
from collections import deque, Counter
from typing import List, Dict, Tuple, Optional, Set, FrozenSet, Any
import numpy as np


class CrossDescriptor:
    """6-axis Cross descriptor for a grid object."""
    __slots__ = (
        'color_axis', 'shape_axis', 'position_axis',
        'scale_axis', 'relation_axis', 'temporal_axis',
        'resonance_id',  # stable ID across frames
    )
    
    def __init__(self):
        self.color_axis = {}
        self.shape_axis = {}
        self.position_axis = {}
        self.scale_axis = {}
        self.relation_axis = {}
        self.temporal_axis = {}
        self.resonance_id = None
    
    def signature(self) -> tuple:
        """Compact signature for matching across frames."""
        return (
            self.color_axis.get('primary_color'),
            self.shape_axis.get('type'),
            self.scale_axis.get('cell_count'),
        )


class CrossObject:
    """A grid object with its Cross descriptor."""
    __slots__ = ('cells', 'colors', 'descriptor', 'obj_id')
    
    def __init__(self, cells: Set[Tuple[int, int]], color_map: Dict[Tuple[int, int], int]):
        self.cells = cells
        self.colors = {pos: color_map[pos] for pos in cells}
        self.descriptor = CrossDescriptor()
        self.obj_id = id(self)


class CrossSensor:
    """
    Panoramic sensor that converts an entire game grid into Cross objects.
    
    Usage:
        sensor = CrossSensor()
        snapshot = sensor.observe(grid)  # returns CrossSnapshot
        # snapshot.objects — all CrossObjects with descriptors
        # snapshot.anomalies — unusual things detected
        # snapshot.diff — changes from previous frame
    """
    
    def __init__(self):
        self._prev_snapshot: Optional[CrossSnapshot] = None
        self._frame_count = 0
        self._object_registry: Dict[int, CrossObject] = {}  # resonance_id → last known object
    
    def observe(self, grid) -> 'CrossSnapshot':
        """Main entry: observe a full grid frame, return CrossSnapshot."""
        g = np.array(grid)
        rows, cols = g.shape
        
        # Phase 1: Extract ALL connected components (multi-color objects supported)
        objects = self._extract_objects(g, rows, cols)
        
        # Phase 2: Compute color axis for each object
        color_freq = self._global_color_frequency(g, rows, cols)
        for obj in objects:
            self._compute_color_axis(obj, color_freq, rows * cols)
        
        # Phase 3: Compute shape axis
        for obj in objects:
            self._compute_shape_axis(obj)
        
        # Phase 4: Compute position axis
        for obj in objects:
            self._compute_position_axis(obj, rows, cols)
        
        # Phase 5: Compute scale axis
        for obj in objects:
            self._compute_scale_axis(obj, rows, cols)
        
        # Phase 6: Compute relation axis (requires all objects)
        self._compute_relation_axes(objects, g, rows, cols)
        
        # Phase 7: Temporal axis — match with previous frame
        diff = None
        if self._prev_snapshot is not None:
            diff = self._compute_temporal(objects, self._prev_snapshot)
        
        # Phase 8: Detect anomalies
        anomalies = self._detect_anomalies(objects, color_freq, g, rows, cols)
        
        # Phase 9: Build panoramic summary
        panorama = self._build_panorama(objects, color_freq, g, rows, cols)
        
        snapshot = CrossSnapshot(
            objects=objects,
            diff=diff,
            anomalies=anomalies,
            panorama=panorama,
            frame_number=self._frame_count,
        )
        
        self._prev_snapshot = snapshot
        self._frame_count += 1
        
        return snapshot
    
    # ── Phase 1: Object Extraction ──────────────────────────────────────────
    
    def _extract_objects(self, g: np.ndarray, rows: int, cols: int) -> List[CrossObject]:
        """Extract all connected components. Single-color flood fill.
        Includes ALL cells, ALL colors, ALL rows."""
        visited = np.zeros((rows, cols), dtype=bool)
        objects = []
        
        for r in range(rows):
            for c in range(cols):
                if visited[r, c]:
                    continue
                color = int(g[r, c])
                # Flood fill
                cells = set()
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
                    stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])
                
                if len(cells) < 1:
                    continue
                
                color_map = {pos: color for pos in cells}
                objects.append(CrossObject(cells, color_map))
        
        return objects
    
    # ── Phase 2: Color Axis ─────────────────────────────────────────────────
    
    def _global_color_frequency(self, g: np.ndarray, rows: int, cols: int) -> Dict[int, float]:
        """Color frequency across entire grid as fraction."""
        total = rows * cols
        counts = Counter(int(v) for v in g.flatten())
        return {c: n / total for c, n in counts.items()}
    
    def _compute_color_axis(self, obj: CrossObject, freq: Dict[int, float], total_cells: int):
        colors = list(obj.colors.values())
        color_counts = Counter(colors)
        primary = color_counts.most_common(1)[0][0]
        
        obj.descriptor.color_axis = {
            'primary_color': primary,
            'color_set': set(color_counts.keys()),
            'color_ratios': {c: n / len(colors) for c, n in color_counts.items()},
            'rarity': 1.0 - freq.get(primary, 0),  # rarer = higher
            'is_rare': freq.get(primary, 0) < 0.005,  # <0.5% of grid
            'is_dominant': freq.get(primary, 0) > 0.1,  # >10% of grid
        }
    
    # ── Phase 3: Shape Axis ─────────────────────────────────────────────────
    
    def _compute_shape_axis(self, obj: CrossObject):
        cells = obj.cells
        if not cells:
            obj.descriptor.shape_axis = {'type': 'empty'}
            return
        
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        r_min, r_max = min(rs), max(rs)
        c_min, c_max = min(cs), max(cs)
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        bbox_area = height * width
        fill_ratio = len(cells) / bbox_area if bbox_area > 0 else 0
        
        # Aspect ratio
        aspect = width / height if height > 0 else 1.0
        
        # Rectangularity: how close to a perfect rectangle
        rectangularity = fill_ratio
        
        # Symmetry checks (horizontal and vertical)
        center_r = (r_min + r_max) / 2
        center_c = (c_min + c_max) / 2
        normalized = frozenset((r - r_min, c - c_min) for r, c in cells)
        
        # Horizontal symmetry (flip around vertical center)
        h_flip = frozenset((r, width - 1 - c) for r, c in normalized)
        h_sym = len(normalized & h_flip) / len(normalized) if normalized else 0
        
        # Vertical symmetry (flip around horizontal center)
        v_flip = frozenset((height - 1 - r, c) for r, c in normalized)
        v_sym = len(normalized & v_flip) / len(normalized) if normalized else 0
        
        # Shape type classification (from structure, not color)
        shape_type = 'blob'
        if fill_ratio > 0.95:
            shape_type = 'rectangle'
        elif fill_ratio > 0.85 and (h_sym > 0.9 or v_sym > 0.9):
            shape_type = 'symmetric_block'
        elif aspect > 4 or aspect < 0.25:
            shape_type = 'line'
        elif fill_ratio < 0.4 and h_sym > 0.7 and v_sym > 0.7:
            shape_type = 'cross_or_ring'
        elif fill_ratio < 0.5:
            shape_type = 'sparse'
        
        # Is it a horizontal bar? (common for timers)
        is_hbar = height <= 2 and width > 10
        
        obj.descriptor.shape_axis = {
            'type': shape_type,
            'bbox': (r_min, c_min, r_max, c_max),
            'height': height,
            'width': width,
            'aspect_ratio': round(aspect, 2),
            'fill_ratio': round(fill_ratio, 3),
            'rectangularity': round(rectangularity, 3),
            'h_symmetry': round(h_sym, 2),
            'v_symmetry': round(v_sym, 2),
            'normalized_pattern': normalized,
            'is_hbar': is_hbar,
        }
    
    # ── Phase 4: Position Axis ──────────────────────────────────────────────
    
    def _compute_position_axis(self, obj: CrossObject, rows: int, cols: int):
        cells = obj.cells
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        centroid_r = sum(rs) / len(rs)
        centroid_c = sum(cs) / len(cs)
        
        # Quadrant (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
        quadrant = (0 if centroid_r < rows / 2 else 2) + (0 if centroid_c < cols / 2 else 1)
        
        # Region (more fine-grained): top/middle/bottom × left/center/right
        v_region = 'top' if centroid_r < rows * 0.33 else ('mid' if centroid_r < rows * 0.66 else 'bottom')
        h_region = 'left' if centroid_c < cols * 0.33 else ('center' if centroid_c < cols * 0.66 else 'right')
        
        # Is in timer area?
        in_timer_area = centroid_r >= 60
        
        obj.descriptor.position_axis = {
            'centroid': (round(centroid_r, 1), round(centroid_c, 1)),
            'centroid_int': (int(centroid_r), int(centroid_c)),
            'quadrant': quadrant,
            'region': f"{v_region}_{h_region}",
            'in_timer_area': in_timer_area,
            'min_r': min(rs),
            'max_r': max(rs),
            'min_c': min(cs),
            'max_c': max(cs),
        }
    
    # ── Phase 5: Scale Axis ─────────────────────────────────────────────────
    
    def _compute_scale_axis(self, obj: CrossObject, rows: int, cols: int):
        n = len(obj.cells)
        total = rows * cols
        bbox = obj.descriptor.shape_axis.get('bbox', (0, 0, 0, 0))
        bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        
        # Size category
        if n == 1:
            size_cat = 'point'
        elif n <= 5:
            size_cat = 'tiny'
        elif n <= 25:
            size_cat = 'small'
        elif n <= 100:
            size_cat = 'medium'
        elif n <= 500:
            size_cat = 'large'
        else:
            size_cat = 'massive'
        
        obj.descriptor.scale_axis = {
            'cell_count': n,
            'grid_fraction': round(n / total, 5),
            'bbox_area': bbox_area,
            'fill_ratio': round(n / bbox_area, 3) if bbox_area > 0 else 0,
            'size_category': size_cat,
        }
    
    # ── Phase 6: Relation Axis ──────────────────────────────────────────────
    
    def _compute_relation_axes(self, objects: List[CrossObject], g: np.ndarray, rows: int, cols: int):
        """Compute inter-object relationships."""
        # Build spatial index: cell → object_id
        cell_to_obj = {}
        for i, obj in enumerate(objects):
            for pos in obj.cells:
                cell_to_obj[pos] = i
        
        for i, obj in enumerate(objects):
            adjacent_ids = set()
            adjacent_colors = set()
            
            # Check 1-cell border around each cell
            for r, c in obj.cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        j = cell_to_obj.get((nr, nc))
                        if j is not None and j != i:
                            adjacent_ids.add(j)
                            adjacent_colors.add(objects[j].descriptor.color_axis.get('primary_color'))
            
            # Containment: does this object's bbox contain another?
            bbox = obj.descriptor.shape_axis.get('bbox', (0, 0, 0, 0))
            contains = []
            contained_by = []
            for j, other in enumerate(objects):
                if j == i:
                    continue
                ob = other.descriptor.shape_axis.get('bbox', (0, 0, 0, 0))
                if bbox[0] <= ob[0] and bbox[1] <= ob[1] and bbox[2] >= ob[2] and bbox[3] >= ob[3]:
                    contains.append(j)
                elif ob[0] <= bbox[0] and ob[1] <= bbox[1] and ob[2] >= bbox[2] and ob[3] >= bbox[3]:
                    contained_by.append(j)
            
            obj.descriptor.relation_axis = {
                'adjacent_count': len(adjacent_ids),
                'adjacent_obj_ids': adjacent_ids,
                'adjacent_colors': adjacent_colors,
                'contains_count': len(contains),
                'contained_by_count': len(contained_by),
                'contains_ids': contains,
                'contained_by_ids': contained_by,
            }
    
    # ── Phase 7: Temporal Axis ──────────────────────────────────────────────
    
    def _compute_temporal(self, current_objects: List[CrossObject], prev_snapshot: 'CrossSnapshot') -> 'FrameDiff':
        """Match objects across frames and compute changes."""
        prev_objects = prev_snapshot.objects
        
        # Match by: primary_color + size similarity + position proximity
        matched = {}  # current_idx → prev_idx
        used_prev = set()
        
        # Sort current by size (larger objects match more reliably)
        indexed_current = sorted(enumerate(current_objects), 
                                  key=lambda x: -x[1].descriptor.scale_axis.get('cell_count', 0))
        
        for ci, cobj in indexed_current:
            best_score = -1
            best_pi = -1
            cc = cobj.descriptor.color_axis.get('primary_color')
            cpos = cobj.descriptor.position_axis.get('centroid_int', (0, 0))
            csize = cobj.descriptor.scale_axis.get('cell_count', 0)
            
            for pi, pobj in enumerate(prev_objects):
                if pi in used_prev:
                    continue
                pc = pobj.descriptor.color_axis.get('primary_color')
                ppos = pobj.descriptor.position_axis.get('centroid_int', (0, 0))
                psize = pobj.descriptor.scale_axis.get('cell_count', 0)
                
                # Color must match for correspondence
                if pc != cc:
                    continue
                
                # Score: combination of position proximity and size similarity
                dist = abs(cpos[0] - ppos[0]) + abs(cpos[1] - ppos[1])
                size_diff = abs(csize - psize) / max(csize, psize, 1)
                
                score = 1.0 / (1 + dist * 0.1 + size_diff * 2)
                
                if score > best_score:
                    best_score = score
                    best_pi = pi
            
            if best_pi >= 0 and best_score > 0.1:
                matched[ci] = best_pi
                used_prev.add(best_pi)
                
                # Set temporal axis
                pobj = prev_objects[best_pi]
                ppos = pobj.descriptor.position_axis.get('centroid_int', (0, 0))
                psize = pobj.descriptor.scale_axis.get('cell_count', 0)
                
                movement = (cpos[0] - ppos[0], cpos[1] - ppos[1])
                size_change = csize - psize
                
                cobj.descriptor.temporal_axis = {
                    'status': 'persisted',
                    'movement': movement,
                    'moved': movement != (0, 0),
                    'size_change': size_change,
                    'size_changed': size_change != 0,
                }
            else:
                cobj.descriptor.temporal_axis = {
                    'status': 'appeared',
                    'movement': (0, 0),
                    'moved': False,
                    'size_change': 0,
                    'size_changed': False,
                }
        
        # Find disappeared objects
        disappeared = []
        for pi, pobj in enumerate(prev_objects):
            if pi not in used_prev:
                disappeared.append(pobj)
        
        # Build moved/changed/appeared/disappeared lists
        moved = [current_objects[ci] for ci in matched 
                 if current_objects[ci].descriptor.temporal_axis.get('moved')]
        appeared = [obj for obj in current_objects 
                    if obj.descriptor.temporal_axis.get('status') == 'appeared']
        size_changed = [current_objects[ci] for ci in matched
                        if current_objects[ci].descriptor.temporal_axis.get('size_changed')]
        
        return FrameDiff(
            matched_count=len(matched),
            moved=moved,
            appeared=appeared,
            disappeared=disappeared,
            size_changed=size_changed,
        )
    
    # ── Phase 8: Anomaly Detection ──────────────────────────────────────────
    
    def _detect_anomalies(self, objects: List[CrossObject], freq: Dict[int, float],
                          g: np.ndarray, rows: int, cols: int) -> List[Dict]:
        """Detect unusual things on the board."""
        anomalies = []
        
        # Anomaly 1: Rare color objects (potential triggers/keys)
        for obj in objects:
            if obj.descriptor.color_axis.get('is_rare'):
                anomalies.append({
                    'type': 'rare_color_object',
                    'color': obj.descriptor.color_axis.get('primary_color'),
                    'position': obj.descriptor.position_axis.get('centroid_int'),
                    'size': obj.descriptor.scale_axis.get('cell_count'),
                    'significance': 'high',
                })
        
        # Anomaly 2: Isolated small objects (potential switches/markers)
        for obj in objects:
            scale = obj.descriptor.scale_axis
            rel = obj.descriptor.relation_axis
            if (scale.get('size_category') in ('point', 'tiny') 
                and rel.get('adjacent_count', 0) <= 1
                and not obj.descriptor.position_axis.get('in_timer_area')):
                anomalies.append({
                    'type': 'isolated_small_object',
                    'color': obj.descriptor.color_axis.get('primary_color'),
                    'position': obj.descriptor.position_axis.get('centroid_int'),
                    'significance': 'medium',
                })
        
        # Anomaly 3: Contained objects (something inside something else)
        for obj in objects:
            if obj.descriptor.relation_axis.get('contained_by_count', 0) > 0:
                anomalies.append({
                    'type': 'contained_object',
                    'color': obj.descriptor.color_axis.get('primary_color'),
                    'position': obj.descriptor.position_axis.get('centroid_int'),
                    'container_colors': [
                        objects[j].descriptor.color_axis.get('primary_color')
                        for j in obj.descriptor.relation_axis.get('contained_by_ids', [])
                    ],
                    'significance': 'high',
                })
        
        # Anomaly 4: Moving objects (from temporal)
        for obj in objects:
            if obj.descriptor.temporal_axis.get('moved'):
                anomalies.append({
                    'type': 'moving_object',
                    'color': obj.descriptor.color_axis.get('primary_color'),
                    'movement': obj.descriptor.temporal_axis.get('movement'),
                    'position': obj.descriptor.position_axis.get('centroid_int'),
                    'significance': 'high',
                })
        
        # Anomaly 5: Shrinking objects (potential countdown/timer/resource)
        for obj in objects:
            sc = obj.descriptor.temporal_axis.get('size_change', 0)
            if sc < -2:  # shrinking by more than 2 cells
                anomalies.append({
                    'type': 'shrinking_object',
                    'color': obj.descriptor.color_axis.get('primary_color'),
                    'size_change': sc,
                    'position': obj.descriptor.position_axis.get('centroid_int'),
                    'significance': 'high',
                })
        
        return anomalies
    
    # ── Phase 9: Panoramic Summary ──────────────────────────────────────────
    
    def _build_panorama(self, objects: List[CrossObject], freq: Dict[int, float],
                        g: np.ndarray, rows: int, cols: int) -> Dict:
        """High-level summary of the entire board."""
        # Color distribution
        color_dist = {c: round(f, 4) for c, f in sorted(freq.items(), key=lambda x: -x[1])}
        
        # Object count by size category
        size_counts = Counter(obj.descriptor.scale_axis.get('size_category') for obj in objects)
        
        # Object count by shape type
        shape_counts = Counter(obj.descriptor.shape_axis.get('type') for obj in objects)
        
        # Region density: how many objects per region
        region_counts = Counter(obj.descriptor.position_axis.get('region') for obj in objects)
        
        # Find the largest non-background objects
        sorted_by_size = sorted(objects, 
                                 key=lambda o: -o.descriptor.scale_axis.get('cell_count', 0))
        top_objects = []
        for obj in sorted_by_size[:10]:
            top_objects.append({
                'color': obj.descriptor.color_axis.get('primary_color'),
                'size': obj.descriptor.scale_axis.get('cell_count'),
                'type': obj.descriptor.shape_axis.get('type'),
                'region': obj.descriptor.position_axis.get('region'),
                'position': obj.descriptor.position_axis.get('centroid_int'),
            })
        
        return {
            'total_objects': len(objects),
            'color_distribution': color_dist,
            'size_counts': dict(size_counts),
            'shape_counts': dict(shape_counts),
            'region_density': dict(region_counts),
            'top_objects': top_objects,
            'unique_colors': len(freq),
            'rare_colors': [c for c, f in freq.items() if f < 0.005],
            'dominant_colors': [c for c, f in freq.items() if f > 0.1],
        }
    
    # ── Convenience Queries ─────────────────────────────────────────────────
    
    def find_by_role_hint(self, snapshot: 'CrossSnapshot', role_hint: str) -> List[CrossObject]:
        """Find objects that structurally match a role description.
        role_hint can be: 'mover', 'trigger', 'barrier', 'countdown', 'container', 'pattern'
        Roles are inferred from Cross structure, NOT from color."""
        results = []
        for obj in snapshot.objects:
            if self._matches_role(obj, role_hint):
                results.append(obj)
        return results
    
    def _matches_role(self, obj: CrossObject, role: str) -> bool:
        t = obj.descriptor.temporal_axis
        s = obj.descriptor.shape_axis
        sc = obj.descriptor.scale_axis
        r = obj.descriptor.relation_axis
        c = obj.descriptor.color_axis
        p = obj.descriptor.position_axis
        
        if role == 'mover':
            return t.get('moved', False)
        elif role == 'trigger':
            return (c.get('is_rare', False) 
                    and sc.get('size_category') in ('point', 'tiny', 'small'))
        elif role == 'barrier':
            return (s.get('type') in ('rectangle', 'line') 
                    and sc.get('size_category') in ('medium', 'large')
                    and not t.get('moved', False))
        elif role == 'countdown':
            return (s.get('is_hbar', False) 
                    and t.get('size_changed', False)
                    and t.get('size_change', 0) < 0)
        elif role == 'container':
            return r.get('contains_count', 0) > 0
        elif role == 'pattern':
            return (sc.get('size_category') in ('small', 'medium')
                    and s.get('fill_ratio', 1) < 0.7
                    and not t.get('moved', False))
        return False


class FrameDiff:
    """Changes between consecutive frames."""
    __slots__ = ('matched_count', 'moved', 'appeared', 'disappeared', 'size_changed')
    
    def __init__(self, matched_count=0, moved=None, appeared=None, 
                 disappeared=None, size_changed=None):
        self.matched_count = matched_count
        self.moved = moved or []
        self.appeared = appeared or []
        self.disappeared = disappeared or []
        self.size_changed = size_changed or []
    
    @property
    def has_changes(self):
        return bool(self.moved or self.appeared or self.disappeared or self.size_changed)
    
    def summary(self) -> str:
        parts = []
        if self.moved: parts.append(f"{len(self.moved)} moved")
        if self.appeared: parts.append(f"{len(self.appeared)} appeared")
        if self.disappeared: parts.append(f"{len(self.disappeared)} disappeared")
        if self.size_changed: parts.append(f"{len(self.size_changed)} resized")
        return ", ".join(parts) if parts else "no changes"


class CrossSnapshot:
    """Complete observation of one frame."""
    __slots__ = ('objects', 'diff', 'anomalies', 'panorama', 'frame_number')
    
    def __init__(self, objects=None, diff=None, anomalies=None, 
                 panorama=None, frame_number=0):
        self.objects = objects or []
        self.diff = diff
        self.anomalies = anomalies or []
        self.panorama = panorama or {}
        self.frame_number = frame_number
    
    def summary(self) -> str:
        """Quick text summary of the snapshot."""
        p = self.panorama
        lines = [
            f"Frame {self.frame_number}: {p.get('total_objects', 0)} objects, "
            f"{p.get('unique_colors', 0)} colors",
        ]
        if self.anomalies:
            lines.append(f"  Anomalies: {len(self.anomalies)}")
            for a in self.anomalies[:5]:
                lines.append(f"    - {a['type']}: color={a.get('color')} pos={a.get('position')} [{a.get('significance')}]")
        if self.diff and self.diff.has_changes:
            lines.append(f"  Changes: {self.diff.summary()}")
        return "\n".join(lines)
