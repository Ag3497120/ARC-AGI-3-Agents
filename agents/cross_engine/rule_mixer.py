"""
rule_mixer.py — Rule Combination Explorer

Explores combinations of rule primitives to find the game's actual rules.
All exploration happens on CrossWorld simulator — zero API cost.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
from collections import deque

from .simulator import CrossWorld
from .primitives import RulePrimitive, MazeRule


class RuleMixer:
    """Explores combinations of rule primitives to find the game's actual rules.
    All exploration happens on CrossWorld simulator — zero API cost."""

    # Weight combinations to try for pairs of rules
    WEIGHT_PAIRS = [
        (1.0, 0.0),
        (0.7, 0.3),
        (0.5, 0.5),
        (0.3, 0.7),
        (0.0, 1.0),
    ]

    def __init__(self, world: CrossWorld, primitives: List[RulePrimitive], excluded_rules: Optional[List[str]] = None):
        self.world = world
        self.primitives = primitives
        self.tried_combinations: List[Dict] = []
        self.successful_rules: List[Dict] = []
        self.excluded_rules: List[str] = excluded_rules or []

    # ── Internal BFS on modified world ────────────────────────────────────────

    def _bfs_path(self, world: CrossWorld) -> Optional[List[int]]:
        """Run BFS on a (possibly modified) world. Returns action list or None."""
        if world.player_pos is None or world.lock_pos is None:
            return None
        return world.find_optimal_path()

    def _score_path(self, path: Optional[List[int]], world: CrossWorld) -> float:
        """Score a path: shorter = better. None path = 0."""
        if path is None or len(path) == 0:
            return 0.0
        # Shorter paths score higher (max useful path ~200 steps)
        max_steps = 200
        return max(0.0, 1.0 - len(path) / max_steps)

    # ── Single Rule Testing ────────────────────────────────────────────────────

    def try_single_rules(self) -> List[Dict]:
        """Try each primitive alone. Returns list of result dicts sorted by score."""
        results = []
        for prim in self.primitives:
            if prim.name in self.excluded_rules:
                continue
            if not prim.can_apply(self.world):
                continue

            modified = prim.apply(self.world)
            path = self._bfs_path(modified)
            path_score = self._score_path(path, modified)
            prior_score = prim.score(self.world)
            total = 0.6 * path_score + 0.4 * prior_score

            result = {
                'type': 'single',
                'rule': prim,
                'path': path,
                'path_length': len(path) if path else -1,
                'path_score': path_score,
                'prior_score': prior_score,
                'total_score': total,
                'world': modified,
            }
            results.append(result)
            self.tried_combinations.append(result)

            if path:
                self.successful_rules.append(result)

        return sorted(results, key=lambda x: -x['total_score'])

    # ── Combination Testing ────────────────────────────────────────────────────

    def try_combinations(self, max_depth: int = 3) -> List[Dict]:
        """Try combinations of 2–3 rules with different weights.
        For each combination:
          1. Apply rules sequentially to a cloned world
          2. Run BFS pathfinding on the modified world
          3. Check if a path to goal exists
          4. Score the combination by path length
        """
        results = []

        applicable = [p for p in self.primitives if p.can_apply(self.world) and p.name not in self.excluded_rules]

        # Pairs
        if max_depth >= 2:
            for i in range(len(applicable)):
                for j in range(i + 1, len(applicable)):
                    p1, p2 = applicable[i], applicable[j]
                    for w1, w2 in self.WEIGHT_PAIRS:
                        # Apply rules in sequence (weights indicate priority/order)
                        cloned = self.world.clone()
                        if w1 >= w2:
                            cloned = p1.apply(cloned)
                            cloned = p2.apply(cloned)
                        else:
                            cloned = p2.apply(cloned)
                            cloned = p1.apply(cloned)

                        path = self._bfs_path(cloned)
                        path_score = self._score_path(path, cloned)

                        # Combined prior score (weighted)
                        prior = w1 * p1.score(self.world) + w2 * p2.score(self.world)
                        total = 0.6 * path_score + 0.4 * prior

                        result = {
                            'type': 'pair',
                            'rules': [p1, p2],
                            'weights': [w1, w2],
                            'path': path,
                            'path_length': len(path) if path else -1,
                            'path_score': path_score,
                            'prior_score': prior,
                            'total_score': total,
                            'world': cloned,
                        }
                        results.append(result)
                        self.tried_combinations.append(result)
                        if path:
                            self.successful_rules.append(result)

        # Triples (only if max_depth allows and we have candidates)
        if max_depth >= 3 and len(applicable) >= 3:
            # Only try the top-3 applicable by prior score to avoid explosion
            top_prims = sorted(applicable, key=lambda p: -p.score(self.world))[:4]
            for i in range(len(top_prims)):
                for j in range(i + 1, len(top_prims)):
                    for k in range(j + 1, len(top_prims)):
                        p1, p2, p3 = top_prims[i], top_prims[j], top_prims[k]
                        cloned = self.world.clone()
                        cloned = p1.apply(cloned)
                        cloned = p2.apply(cloned)
                        cloned = p3.apply(cloned)

                        path = self._bfs_path(cloned)
                        path_score = self._score_path(path, cloned)
                        prior = (p1.score(self.world) + p2.score(self.world) + p3.score(self.world)) / 3
                        total = 0.6 * path_score + 0.4 * prior

                        result = {
                            'type': 'triple',
                            'rules': [p1, p2, p3],
                            'weights': [1/3, 1/3, 1/3],
                            'path': path,
                            'path_length': len(path) if path else -1,
                            'path_score': path_score,
                            'prior_score': prior,
                            'total_score': total,
                            'world': cloned,
                        }
                        results.append(result)
                        self.tried_combinations.append(result)
                        if path:
                            self.successful_rules.append(result)

        return sorted(results, key=lambda x: -x['total_score'])

    # ── Game Explanation ───────────────────────────────────────────────────────

    def explain_game(self) -> Dict[str, Any]:
        """Return the discovered game rules as a structured description.

        Returns:
            {
                'primary_rule': RulePrimitive,
                'secondary_rules': [RulePrimitive, ...],
                'weights': [float, ...],
                'confidence': float,
                'optimal_path': [int, ...],
                'modified_world': CrossWorld,
            }
        """
        # Step 1: Try single rules
        single_results = self.try_single_rules()

        # Step 2: Try combinations
        combo_results = self.try_combinations(max_depth=3)

        # Step 3: Merge and find best
        all_results = single_results + combo_results
        all_results.sort(key=lambda x: -x['total_score'])

        if not all_results:
            # Absolute fallback: return empty path, maze rule
            return {
                'primary_rule': MazeRule(),
                'secondary_rules': [],
                'weights': [1.0],
                'confidence': 0.0,
                'optimal_path': [],
                'modified_world': self.world.clone(),
            }

        best = all_results[0]

        # Extract primary / secondary rules
        if best['type'] == 'single':
            primary = best['rule']
            secondary = []
            weights = [1.0]
        elif best['type'] == 'pair':
            rules = best['rules']
            wts = best['weights']
            primary = rules[0]
            secondary = rules[1:]
            weights = wts
        else:  # triple
            rules = best['rules']
            wts = best['weights']
            primary = rules[0]
            secondary = rules[1:]
            weights = wts

        # Confidence: based on whether we found a path and total score
        path = best.get('path') or []
        confidence = best['total_score'] if path else best['total_score'] * 0.5

        # Fallback: if no combination found a path, try brute force open-all-locks
        if not path:
            fallback_world = self.world.clone()
            # Open every color-5 cell in the entire grid
            for r in range(fallback_world.rows):
                for c in range(fallback_world.cols):
                    if fallback_world.raw[r][c] == 5:
                        fallback_world.set_color(r, c, 3, 'corridor')
            path = fallback_world.find_optimal_path() or []
            best['world'] = fallback_world
            confidence = 0.1 if path else 0.0

        return {
            'primary_rule': primary,
            'secondary_rules': secondary,
            'weights': weights,
            'confidence': confidence,
            'optimal_path': path,
            'modified_world': best.get('world', self.world.clone()),
        }
