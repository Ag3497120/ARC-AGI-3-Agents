"""
cross_space.py — Experiential Memory System for ARC-AGI-3 Agent

Replaces top-down rule definitions with bottom-up experience resonance.
Experiences are placed in a multi-dimensional space indexed by colors,
positions, and events. When a new experience occurs, it resonates with
past experiences sharing the same axes. The collision of experiences
generates action impulses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class Ripple:
    """A disturbance in CrossSpace caused by grid changes."""
    frame: int
    position: Tuple[int, int]                          # center of change
    colors: Set[int]                                   # colors involved in the change
    intensity: float                                   # how big the change was (0.0 - 1.0)
    color_transitions: Dict[Tuple[int, int], int]      # (old, new) → count


@dataclass
class Experience:
    frame: int
    position: Tuple[int, int]       # where
    colors_involved: Set[int]       # which colors were part of this
    event_type: str                 # 'blocked', 'moved', 'opened', 'closed', 'clicked', 'collected', 'level_up'
    action_taken: int               # what action index triggered this
    details: Dict[str, Any] = field(default_factory=dict)
    # details can hold:
    #   'color_under_player': int
    #   'wall_color': int
    #   'opened_region': (r_min, c_min, r_max, c_max)
    #   'closed_region': (r_min, c_min, r_max, c_max)
    #   'changes': List[Tuple[int,int]]       # changed cells
    #   'color_transitions': Dict[Tuple[int,int], int]


@dataclass
class Impulse:
    """An action impulse born from experience collision."""
    action_type: str        # 'seek_color', 'go_to', 'avoid', 'click_at', 'explore_unknown', 'retry_with_timing'
    priority: float         # how strongly this impulse resonates
    target: Optional[Any] = None   # color to seek, position to go to, etc.
    source_experiences: List[int] = field(default_factory=list)  # which experiences produced this
    reason: str = ''        # human-readable explanation


# ── CrossSpace ────────────────────────────────────────────────────────────────

class CrossSpace:
    """
    Multi-dimensional experiential memory space.

    Experiences are indexed along three axes:
      - color_index:    color → [experience indices]
      - event_index:    event_type → [experience indices]
      - position_grid:  (r//8, c//8) bucket → [experience indices]

    Resonance finds past experiences that share colors, complementary events,
    and spatial proximity. Collision of resonating experiences generates Impulses.
    """

    def __init__(self):
        self.experiences: List[Experience] = []
        self.color_index: Dict[int, List[int]] = defaultdict(list)              # color → experience indices
        self.event_index: Dict[str, List[int]] = defaultdict(list)              # event_type → experience indices
        self.position_grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)  # (r//8, c//8) → indices

        # ── Ripple system ──────────────────────────────────────────────────────
        self.ripples: List[Ripple] = []
        self.color_energy: Dict[int, float] = defaultdict(float)               # accumulated energy per color
        self.position_energy: Dict[Tuple[int, int], float] = defaultdict(float)  # energy per spatial bucket
        self.ripple_threshold: float = 3.0   # energy threshold for impulse generation
        self.energy_decay: float = 0.85      # energy decays each frame (prevents infinite buildup)

    # ── Core API ──────────────────────────────────────────────────────────────

    def record(self, exp: Experience) -> None:
        """Place an experience in the space. Index by all its axes."""
        idx = len(self.experiences)
        self.experiences.append(exp)
        for color in exp.colors_involved:
            self.color_index[color].append(idx)
        self.event_index[exp.event_type].append(idx)
        bucket = (exp.position[0] // 8, exp.position[1] // 8)
        self.position_grid[bucket].append(idx)

    def resonate(self, current: Experience) -> List[Experience]:
        """Find past experiences that resonate with the current one.

        Resonance = shared colors × complementary events × spatial proximity.
        Returns up to 10 most strongly resonating past experiences.
        """
        # Gather candidates from color index (fast lookup)
        candidate_indices: set = set()
        for color in current.colors_involved:
            candidate_indices.update(self.color_index.get(color, []))

        # Also check nearby spatial buckets (5×5 neighbourhood)
        cr, cc = current.position[0] // 8, current.position[1] // 8
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                candidate_indices.update(self.position_grid.get((cr + dr, cc + dc), []))

        # Score each candidate
        scored = []
        for idx in candidate_indices:
            past = self.experiences[idx]
            if past.frame == current.frame:
                continue  # skip self (same-frame experiences)

            # Dimension 1: Color overlap
            color_overlap = len(current.colors_involved & past.colors_involved)
            if color_overlap == 0:
                continue

            # Dimension 2: Event complementarity
            complement = self._event_complement(current.event_type, past.event_type)

            # Dimension 3: Spatial proximity (closer = stronger, but not required)
            dist = abs(current.position[0] - past.position[0]) + abs(current.position[1] - past.position[1])
            proximity = 1.0 / (1.0 + dist / 20.0)

            # Dimension 4: Recency (more recent = slightly stronger)
            recency = 1.0 / (1.0 + (current.frame - past.frame) / 50.0)

            score = color_overlap * complement * (0.3 + 0.7 * proximity) * (0.5 + 0.5 * recency)
            if score > 0.1:
                scored.append((score, past))

        scored.sort(key=lambda x: -x[0])
        return [exp for _, exp in scored[:10]]  # top 10 resonating experiences

    def _event_complement(self, current_event: str, past_event: str) -> float:
        """How strongly two event types complement each other."""
        COMPLEMENT_MAP = {
            ('blocked', 'opened'): 10.0,    # "I'm blocked, but I once opened something like this"
            ('blocked', 'moved'):   3.0,    # "I'm blocked, but I moved through here before"
            ('blocked', 'clicked'): 5.0,   # "I'm blocked, maybe clicking helps"
            ('opened', 'blocked'):  5.0,   # "I opened this, it was blocked before"
            ('opened', 'opened'):   2.0,   # "This keeps opening — pattern?"
            ('closed', 'opened'):   8.0,   # "It closed, but I know how to open it"
            ('moved', 'blocked'):   2.0,   # "I moved here, was it blocked before?"
            ('clicked', 'opened'):  6.0,   # "Clicking opened something"
            ('changed', 'opened'):  4.0,   # "something changed, and we've seen openings with this color"
            ('changed', 'closed'):  3.0,   # "something changed, and we've seen closings"
            ('changed', 'moved'):   1.0,   # "something changed where we moved"
            ('changed', 'blocked'): 2.0,   # "something changed near a blockage"
        }
        return COMPLEMENT_MAP.get((current_event, past_event), 0.5)

    def collide(self, current: Experience, resonating: List[Experience]) -> List[Impulse]:
        """Collide current experience with resonating past experiences.

        Collisions produce Impulses — action suggestions born from experience overlap.
        Similar impulses are merged (reinforcement); result is sorted by priority.
        """
        impulses = []
        for past in resonating:
            impulse = self._generate_impulse(current, past)
            if impulse:
                impulses.append(impulse)

        # Merge similar impulses (same action_type + same target)
        merged = self._merge_impulses(impulses)

        # Sort by priority (highest first)
        merged.sort(key=lambda i: -i.priority)
        return merged

    def _generate_impulse(self, current: Experience, past: Experience) -> Optional[Impulse]:
        """Generate an action impulse from the collision of two experiences."""

        # Pattern 1: BLOCKED now + OPENED before → seek the trigger
        if current.event_type == 'blocked' and past.event_type == 'opened':
            trigger_colors = past.colors_involved - current.colors_involved
            color_under = past.details.get('color_under_player')
            if color_under is not None:
                trigger_colors = set(trigger_colors)  # ensure mutable copy
                trigger_colors.add(color_under)

            if trigger_colors:
                return Impulse(
                    action_type='seek_color',
                    priority=10.0,
                    target=trigger_colors,
                    source_experiences=[current.frame, past.frame],
                    reason=(
                        f"blocked by colors {current.colors_involved}, "
                        f"but opened before at frame {past.frame} with colors {trigger_colors}"
                    ),
                )
            else:
                # Same colors — go to where it opened
                return Impulse(
                    action_type='go_to',
                    priority=7.0,
                    target=past.position,
                    source_experiences=[current.frame, past.frame],
                    reason=f"blocked, but opened at {past.position} before",
                )

        # Pattern 2: BLOCKED now + MOVED through same area before
        if current.event_type == 'blocked' and past.event_type == 'moved':
            # The path was open before — something closed it, or we need a different approach
            return Impulse(
                action_type='explore_nearby',
                priority=4.0,
                target=past.position,
                source_experiences=[current.frame, past.frame],
                reason=f"blocked at {current.position}, but moved through {past.position} before",
            )

        # Pattern 3: Something OPENED → go through it NOW
        if current.event_type == 'opened':
            opened_region = current.details.get('opened_region')
            if opened_region:
                center = (
                    (opened_region[0] + opened_region[2]) // 2,
                    (opened_region[1] + opened_region[3]) // 2,
                )
                return Impulse(
                    action_type='go_to',
                    priority=15.0,  # urgent — it might close
                    target=center,
                    source_experiences=[current.frame],
                    reason=f"wall opened at {center}, go NOW",
                )

        # Pattern 4: Something CLOSED → remember what we were doing when it was open
        if current.event_type == 'closed' and past.event_type == 'opened':
            return Impulse(
                action_type='retry_trigger',
                priority=6.0,
                target=past.details.get('color_under_player'),
                source_experiences=[current.frame, past.frame],
                reason=f"path closed, try to trigger again like at frame {past.frame}",
            )

        # Pattern 5: Explore unknown areas when stuck
        if current.event_type == 'stuck':
            return Impulse(
                action_type='explore_unknown',
                priority=3.0,
                target=None,
                source_experiences=[current.frame],
                reason="stuck, explore unvisited areas",
            )

        # Pattern 6: investigate_color — ripple energy exceeded threshold, no past experience
        if current.event_type == 'changed' and past.event_type == 'opened':
            shared = current.colors_involved & past.colors_involved
            if shared:
                return Impulse(
                    action_type='investigate_color',
                    priority=4.0,
                    target=shared,
                    source_experiences=[current.frame, past.frame],
                    reason=f"colors {shared} changed and were previously involved in opening",
                )

        # Pattern 7: avoid_color — color associated with closing
        if current.event_type == 'changed' and past.event_type == 'closed':
            shared = current.colors_involved & past.colors_involved
            if shared:
                return Impulse(
                    action_type='avoid_color',
                    priority=3.0,
                    target=shared,
                    source_experiences=[current.frame, past.frame],
                    reason=f"colors {shared} changed and were previously involved in closing",
                )

        return None

    def _merge_impulses(self, impulses: List[Impulse]) -> List[Impulse]:
        """Merge similar impulses — boost priority when multiple experiences agree."""
        if not impulses:
            return []

        merged: List[Impulse] = []
        for imp in impulses:
            found = False
            for existing in merged:
                if existing.action_type == imp.action_type and existing.target == imp.target:
                    existing.priority += imp.priority * 0.5  # reinforcement
                    existing.source_experiences.extend(imp.source_experiences)
                    found = True
                    break
            if not found:
                merged.append(imp)
        return merged

    # ── Convenience: batch record from agent state ────────────────────────────

    def record_movement(self, frame: int, pos: Tuple[int, int], action_idx: int, color_under: int) -> None:
        """Record a successful movement."""
        self.record(Experience(
            frame=frame,
            position=pos,
            colors_involved={color_under},
            event_type='moved',
            action_taken=action_idx,
            details={'color_under_player': color_under},
        ))

    def record_blocked(self, frame: int, pos: Tuple[int, int], action_idx: int, wall_colors: Set[int]) -> None:
        """Record a blocked movement attempt."""
        self.record(Experience(
            frame=frame,
            position=pos,
            colors_involved=wall_colors,
            event_type='blocked',
            action_taken=action_idx,
            details={'wall_colors': wall_colors},
        ))

    def record_reaction(
        self,
        frame: int,
        player_pos: Tuple[int, int],
        action_idx: int,
        change_type: str,
        color_transitions: Dict[Tuple[int, int], int],
        changed_cells: List[Tuple[int, int]],
        color_under_player: int,
    ) -> None:
        """Record an environmental reaction (wall opened/closed, click response, etc.)."""
        colors: Set[int] = set()
        for (old, new), _cnt in color_transitions.items():
            colors.add(old)
            colors.add(new)

        bbox = None
        if changed_cells:
            rs = [r for r, c in changed_cells]
            cs = [c for r, c in changed_cells]
            bbox = (min(rs), min(cs), max(rs), max(cs))

        event = (
            'opened' if change_type == 'wall_opened' else
            'closed' if change_type == 'wall_closed' else
            'clicked' if change_type == 'click_response' else
            'changed'
        )

        self.record(Experience(
            frame=frame,
            position=player_pos,
            colors_involved=colors,
            event_type=event,
            action_taken=action_idx,
            details={
                'color_under_player': color_under_player,
                'color_transitions': dict(color_transitions),
                'opened_region': bbox if event == 'opened' else None,
                'closed_region': bbox if event == 'closed' else None,
                'changed_cells_count': len(changed_cells),
            },
        ))

    def record_stuck(self, frame: int, pos: Tuple[int, int], visited_positions) -> None:
        """Record that the agent is stuck (no progress for N frames)."""
        self.record(Experience(
            frame=frame,
            position=pos,
            colors_involved=set(),
            event_type='stuck',
            action_taken=-1,
            details={'visited_count': len(visited_positions)},
        ))

    # ── Query: find colors/positions to seek ──────────────────────────────────

    def get_seek_targets(
        self,
        grid,
        current_pos: Tuple[int, int],
        wall_colors: Set[int],
    ) -> List[Tuple[int, int]]:
        """Given current blocked state, find positions worth visiting.

        Uses resonance to find colors that previously opened walls.
        Returns up to 5 target positions on the grid.
        """
        blocked_exp = Experience(
            frame=len(self.experiences),
            position=current_pos,
            colors_involved=wall_colors,
            event_type='blocked',
            action_taken=-1,
        )

        resonating = self.resonate(blocked_exp)
        impulses = self.collide(blocked_exp, resonating)

        targets: List[Tuple[int, int]] = []
        for imp in impulses:
            if imp.action_type == 'seek_color' and imp.target:
                # Find cells with the target colors on the grid
                for r in range(min(60, len(grid))):
                    for c in range(len(grid[0]) if grid else 0):
                        if int(grid[r][c]) in imp.target:
                            targets.append((r, c))
                break  # use highest priority impulse only

            elif imp.action_type == 'go_to' and imp.target:
                targets.append(imp.target)

        return targets[:5]  # return top 5 target positions

    def get_urgent_impulses(self) -> List[Impulse]:
        """Get impulses from the most recent experience that need immediate action."""
        if not self.experiences:
            return []

        latest = self.experiences[-1]
        resonating = self.resonate(latest)
        return self.collide(latest, resonating)

    # ── Ripple System ─────────────────────────────────────────────────────────

    def ripple(self, frame: int, prev_grid, curr_grid, player_pos: Tuple[int, int],
               action_idx: int) -> List[Impulse]:
        """Process frame diff as ripples. Returns impulses if energy exceeds threshold.

        Called EVERY frame, not just on detected reactions.
        """
        import numpy as np

        prev = np.array(prev_grid) if not isinstance(prev_grid, np.ndarray) else prev_grid
        curr = np.array(curr_grid) if not isinstance(curr_grid, np.ndarray) else curr_grid

        # 1. Find all changed cells (exclude timer rows 60+)
        limit = min(60, prev.shape[0])
        changed = []
        for r in range(limit):
            for c in range(prev.shape[1] if len(prev.shape) > 1 else 64):
                if int(prev[r, c]) != int(curr[r, c]):
                    changed.append((r, c))

        if not changed:
            # Decay existing energy even with no changes
            self._decay_energy()
            return []

        # 2. Exclude player footprint (player moving is not a ripple)
        pr, pc = player_pos
        non_player = [(r, c) for r, c in changed if abs(r - pr) > 6 or abs(c - pc) > 6]

        # If all changes were player movement, still create a tiny ripple
        if not non_player:
            non_player = changed[:3]  # use a few cells as weak signal

        # 3. Compute ripple
        colors: Set[int] = set()
        transitions: Dict[Tuple[int, int], int] = {}
        for r, c in non_player:
            old = int(prev[r, c])
            new = int(curr[r, c])
            colors.add(old)
            colors.add(new)
            key = (old, new)
            transitions[key] = transitions.get(key, 0) + 1

        # Intensity = proportion of play area that changed (0-1)
        intensity = min(1.0, len(non_player) / 100.0)

        # Center of change
        if non_player:
            center_r = sum(r for r, c in non_player) // len(non_player)
            center_c = sum(c for r, c in non_player) // len(non_player)
        else:
            center_r, center_c = pr, pc

        rpl = Ripple(
            frame=frame,
            position=(center_r, center_c),
            colors=colors,
            intensity=intensity,
            color_transitions=transitions,
        )
        self.ripples.append(rpl)

        # 4. Add energy to color and position axes
        for color in colors:
            self.color_energy[color] += intensity

        bucket = (center_r // 8, center_c // 8)
        self.position_energy[bucket] += intensity

        # 5. Decay all energy
        self._decay_energy()

        # 6. Check thresholds — generate impulses from accumulated energy
        impulses = self._check_energy_thresholds(frame, player_pos, action_idx)

        return impulses

    def _decay_energy(self) -> None:
        """Decay all accumulated energy."""
        for color in list(self.color_energy):
            self.color_energy[color] *= self.energy_decay
            if self.color_energy[color] < 0.01:
                del self.color_energy[color]

        for pos in list(self.position_energy):
            self.position_energy[pos] *= self.energy_decay
            if self.position_energy[pos] < 0.01:
                del self.position_energy[pos]

    def _check_energy_thresholds(self, frame: int, player_pos: Tuple[int, int],
                                  action_idx: int) -> List[Impulse]:
        """When energy on a color axis exceeds threshold, generate an impulse."""
        impulses: List[Impulse] = []

        # Colors with high energy = something keeps happening to them
        hot_colors = [(c, e) for c, e in self.color_energy.items() if e > self.ripple_threshold]

        if hot_colors:
            hot_colors.sort(key=lambda x: -x[1])

            for color, energy in hot_colors[:3]:
                # Find past experiences involving this color
                past_exps = [self.experiences[i] for i in self.color_index.get(color, [])]

                if not past_exps:
                    # No past experience with this color — it's new and active → explore it
                    impulses.append(Impulse(
                        action_type='investigate_color',
                        priority=energy * 2,
                        target={color},
                        source_experiences=[frame],
                        reason=f"color {color} has high energy ({energy:.1f}) but no past experience — investigate",
                    ))
                else:
                    # Past experience exists — check what happened
                    opened_exps = [e for e in past_exps if e.event_type == 'opened']
                    closed_exps = [e for e in past_exps if e.event_type == 'closed']

                    if opened_exps:
                        # This color was involved in opening something before
                        latest = opened_exps[-1]
                        impulses.append(Impulse(
                            action_type='seek_color',
                            priority=energy * 3,
                            target={color},
                            source_experiences=[frame, latest.frame],
                            reason=f"color {color} energy={energy:.1f}, previously opened at frame {latest.frame}",
                        ))
                    elif closed_exps:
                        # This color was involved in closing — avoid or reverse
                        impulses.append(Impulse(
                            action_type='avoid_color',
                            priority=energy * 1.5,
                            target={color},
                            source_experiences=[frame],
                            reason=f"color {color} energy={energy:.1f}, associated with closing",
                        ))

        # Positions with high energy = something keeps happening there
        hot_positions = [(p, e) for p, e in self.position_energy.items() if e > self.ripple_threshold]
        for (br, bc), energy in sorted(hot_positions, key=lambda x: -x[1])[:2]:
            center = (br * 8 + 4, bc * 8 + 4)
            impulses.append(Impulse(
                action_type='go_to',
                priority=energy * 2,
                target=center,
                source_experiences=[frame],
                reason=f"position bucket ({br},{bc}) has high energy ({energy:.1f}) — investigate",
            ))

        return impulses

    def process_frame(self, frame: int, prev_grid, curr_grid, player_pos: Tuple[int, int],
                      action_idx: int) -> List[Impulse]:
        """Main entry point: process a single frame's changes.

        Returns any impulses generated by ripples exceeding thresholds.
        Called every frame by the agent.

        Integration example (v26 choose_action — DO NOT implement here):
            if self._cross_space and self.prev_grid is not None:
                ripple_impulses = self._cross_space.process_frame(
                    self._frame, self.prev_grid, grid,
                    self._ctrl_pos or (32, 32), self._last_aidx
                )
                for imp in ripple_impulses:
                    # Add to detour queue or action queue based on impulse type
        """
        return self.ripple(frame, prev_grid, curr_grid, player_pos, action_idx)

    def get_hot_colors(self) -> List[Tuple[int, float]]:
        """Return colors sorted by current energy level."""
        return sorted(self.color_energy.items(), key=lambda x: -x[1])

    def get_hot_positions(self) -> List[Tuple[Tuple[int, int], float]]:
        """Return position buckets sorted by current energy level."""
        return sorted(self.position_energy.items(), key=lambda x: -x[1])

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a brief diagnostic summary of the experience space."""
        event_counts = {k: len(v) for k, v in self.event_index.items()}
        color_count = len(self.color_index)
        bucket_count = len(self.position_grid)
        return (
            f"CrossSpace({len(self.experiences)} experiences | "
            f"events={event_counts} | "
            f"colors={color_count} | "
            f"spatial_buckets={bucket_count})"
        )
