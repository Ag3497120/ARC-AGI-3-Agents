"""
cross_axiom.py — Cross6軸による事象捕捉 → jcross等価式の動的生成

kofdai型の核心:
- フレーム差分をCross6軸で記述する
- 6軸の変化パターンから等価式（$=）を生成する
- 等価式をsoul.jcrossに書き込む → Cross構造が成長する
- 時間ではなく空間で推論する
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, Counter
import numpy as np
import sys


@dataclass
class CrossEvent:
    """Cross6軸で記述された事象"""
    frame: int

    # 6軸の変化ベクトル
    color_delta: Dict[str, Any] = field(default_factory=dict)    # 色の変化
    shape_delta: Dict[str, Any] = field(default_factory=dict)    # 形状の変化
    position_delta: Dict[str, Any] = field(default_factory=dict) # 位置の変化
    scale_delta: Dict[str, Any] = field(default_factory=dict)    # サイズの変化
    relation_delta: Dict[str, Any] = field(default_factory=dict) # 関係の変化
    temporal_delta: Dict[str, Any] = field(default_factory=dict) # 時間的変化

    # トリガー情報
    trigger_action: int = -1                    # 何の行動でこの変化が起きたか
    trigger_position: Tuple[int, int] = (0, 0)  # プレイヤー位置

    # 6軸のシグネチャ（空間的ハッシュ）
    def signature(self) -> tuple:
        """この事象の空間的な「場所」を返す"""
        return (
            self.color_delta.get('transition_type', 'none'),
            self.shape_delta.get('type_change', 'none'),
            self.position_delta.get('region', 'none'),
            self.scale_delta.get('size_change', 'none'),
            self.relation_delta.get('adjacency_change', 'none'),
        )


@dataclass
class CrossAxiom:
    """Cross空間上で発見された等価式"""
    axiom_id: int

    # 等価式: trigger $= effect
    trigger_sig: tuple  # トリガーの6軸シグネチャ
    effect_sig: tuple   # 効果の6軸シグネチャ

    # 確信度
    observations: int = 1   # 何回観測されたか
    confirmed: bool = False  # 3回以上で確定

    # jcross形式の等価式
    jcross_rule: str = ""  # 生成されたjcrossコード

    def confidence(self) -> float:
        return min(self.observations / 3.0, 1.0)


@dataclass
class CausalAxiom:
    """action×context $= effect の因果的等価式"""
    axiom_id: int

    # 因果の左辺: action × context
    action_idx: int                    # どの行動
    context_sig: tuple                 # 行動時の6軸コンテキスト (player_region, color_under, nearby_colors...)

    # 因果の右辺: effect
    effect_sig: tuple                  # 効果の6軸シグネチャ
    effect_details: Dict[str, Any] = field(default_factory=dict)
    # effect_details例:
    #   {'type': 'wall_open', 'region': (28,32,34,38), 'color_from': 5, 'color_to': 3}
    #   {'type': 'color_cycle', 'colors': [3,5,7], 'block_pos': (10,40)}
    #   {'type': 'player_move', 'delta': (0, 5)}
    #   {'type': 'no_effect'}

    # 確信度
    observations: int = 1
    confirmed: bool = False            # 3回で確定
    contradictions: int = 0            # 矛盾回数（同じ因果で違う効果）

    # jcross形式
    jcross_equiv: str = ""             # 等価式のjcross表現

    def match(self, action_idx: int, context_sig: tuple) -> bool:
        """この等価式が適用可能か"""
        return self.action_idx == action_idx and self.context_sig == context_sig


class CrossAxiomEngine:
    """
    フレーム差分 → 6軸CrossEvent → 等価式(CrossAxiom) → jcross動的書き換え

    kofdai型の実装:
    - 時間を排除: イベントはCross空間上の「場所」を持つ
    - 空間的整合性: 同じ「場所」のイベントが共鳴して等価式を生成
    - 変化がトリガー: diffがないフレームは何も起きない
    """

    def __init__(self):
        self.events: List[CrossEvent] = []
        self.axioms: List[CrossAxiom] = []
        self._axiom_counter = 0

        # 空間インデックス: signature → [event indices]
        self._sig_index: Dict[tuple, List[int]] = defaultdict(list)

        # 発見されたゲームルール
        self.discovered_rules: Dict[str, str] = {}
        # 例: 'click_color_cycle' → 'クリック $= 色サイクル(+1 mod N)'
        #     'move_asymmetric' → '左右移動 $= 28セル, 上下移動 $= 1セル'
        #     'pattern_match' → '左パターン $= 右パターン → クリア'

        # 因果的等価式 (action × context $= effect)
        self.causal_axioms: List['CausalAxiom'] = []
        self._causal_counter = 0

        # 色サイクル検出用
        self._color_edges: Dict[Tuple[int, int], int] = defaultdict(int)
        self._detected_cycles: List[List[int]] = []

    def process_frame(self, frame: int, prev_grid, curr_grid,
                      player_pos: Tuple[int, int], action_idx: int,
                      prev_objects=None, curr_objects=None) -> List[CrossAxiom]:
        """フレーム差分をCross6軸で分析し、新しい等価式を返す"""

        if prev_grid is None:
            return []

        prev_g = np.array(prev_grid) if not isinstance(prev_grid, np.ndarray) else prev_grid
        curr_g = np.array(curr_grid) if not isinstance(curr_grid, np.ndarray) else curr_grid

        # グリッドサイズを合わせる
        min_rows = min(prev_g.shape[0], curr_g.shape[0], 60)

        # 差分を計算（timer除外: 上60行のみ）
        diff_mask = prev_g[:min_rows] != curr_g[:min_rows]
        if not np.any(diff_mask):
            return []  # 変化なし → 何も起きない

        changed_cells = list(zip(*np.where(diff_mask)))
        if not changed_cells:
            return []

        # 色の遷移を分析
        color_transitions: Dict[Tuple[int, int], int] = defaultdict(int)
        changed_colors_from: Set[int] = set()
        changed_colors_to: Set[int] = set()
        for r, c in changed_cells:
            old = int(prev_g[r, c])
            new = int(curr_g[r, c])
            color_transitions[(old, new)] += 1
            changed_colors_from.add(old)
            changed_colors_to.add(new)

        # 変化の空間的範囲
        rows = [r for r, c in changed_cells]
        cols = [c for r, c in changed_cells]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        center = ((r_min + r_max) // 2, (c_min + c_max) // 2)

        # === 6軸で事象を記述 ===

        event = CrossEvent(frame=frame, trigger_action=action_idx, trigger_position=player_pos)

        # 1. color_delta: 色の変化パターン
        dominant_transition = max(color_transitions.items(), key=lambda x: x[1])
        is_cycle = False
        cycle_colors: List[int] = []
        # 色サイクル検出: A→B, B→C, C→A のパターン
        for (old, new), count in color_transitions.items():
            if (new, old) in color_transitions or any(
                (new, third) in color_transitions for third in changed_colors_to
            ):
                is_cycle = True
                cycle_colors = sorted(changed_colors_from | changed_colors_to)
                break

        event.color_delta = {
            'transition_type': 'cycle' if is_cycle else ('swap' if len(color_transitions) == 1 else 'multi'),
            'dominant': dominant_transition[0],
            'count': len(changed_cells),
            'transitions': dict(color_transitions),
            'cycle_colors': cycle_colors,
        }

        # 2. shape_delta: 変化した領域の形状
        width = c_max - c_min + 1
        height = r_max - r_min + 1
        fill_ratio = len(changed_cells) / (width * height) if width * height > 0 else 0
        event.shape_delta = {
            'type_change': 'block' if fill_ratio > 0.7 else ('line' if width == 1 or height == 1 else 'scattered'),
            'bbox': (r_min, c_min, r_max, c_max),
            'width': width,
            'height': height,
            'fill_ratio': round(fill_ratio, 2),
        }

        # 3. position_delta: 変化の位置（プレイヤーからの相対）
        dr = center[0] - player_pos[0]
        dc = center[1] - player_pos[1]
        distance = (dr * dr + dc * dc) ** 0.5
        event.position_delta = {
            'region': 'near' if distance < 10 else ('mid' if distance < 25 else 'far'),
            'direction': 'up' if dr < -3 else ('down' if dr > 3 else ('left' if dc < -3 else ('right' if dc > 3 else 'here'))),
            'center': center,
            'distance': round(distance, 1),
        }

        # 4. scale_delta: 変化のスケール
        total_cells = min_rows * (curr_g.shape[1] if len(curr_g.shape) > 1 else 64)
        change_ratio = len(changed_cells) / total_cells if total_cells > 0 else 0
        event.scale_delta = {
            'size_change': 'tiny' if change_ratio < 0.005 else ('small' if change_ratio < 0.02 else ('medium' if change_ratio < 0.1 else 'large')),
            'cell_count': len(changed_cells),
            'ratio': round(change_ratio, 4),
        }

        # 5. relation_delta: 変化した領域と他の構造の関係
        # プレイヤーが変化に接触しているか
        touching_player = any(
            abs(r - player_pos[0]) <= 5 and abs(c - player_pos[1]) <= 5
            for r, c in changed_cells[:10]  # 最初の10セルだけチェック
        )
        event.relation_delta = {
            'adjacency_change': 'player_contact' if touching_player else 'remote',
            'player_caused': touching_player and distance < 10,
        }

        # 6. temporal_delta: 変化の時間的パターン
        # 前回の同じシグネチャのイベントからの間隔
        sig = event.signature()
        prev_frames = [self.events[i].frame for i in self._sig_index.get(sig, [])]
        interval = frame - prev_frames[-1] if prev_frames else -1
        event.temporal_delta = {
            'interval': interval,
            'periodic': (
                interval > 0 and len(prev_frames) >= 2 and all(
                    abs((prev_frames[i + 1] - prev_frames[i]) - interval) <= 2
                    for i in range(len(prev_frames) - 1)
                )
            ) if len(prev_frames) >= 2 else False,
        }

        # イベントを登録
        idx = len(self.events)
        self.events.append(event)
        self._sig_index[sig].append(idx)

        # === 共鳴 → 等価式の生成 ===
        new_axioms = self._resonate(event, idx)

        return new_axioms

    def _resonate(self, event: CrossEvent, event_idx: int) -> List[CrossAxiom]:
        """同じシグネチャのイベントが共鳴して等価式を生成"""
        sig = event.signature()
        same_sig_indices = self._sig_index[sig]

        new_axioms = []

        # 同じシグネチャが2回以上 → 等価式候補
        if len(same_sig_indices) >= 2:
            # 既存の等価式を探す
            existing = None
            for ax in self.axioms:
                if ax.trigger_sig == sig:
                    existing = ax
                    break

            if existing:
                existing.observations += 1
                if existing.observations >= 3 and not existing.confirmed:
                    existing.confirmed = True
                    existing.jcross_rule = self._generate_jcross_rule(existing, event)
                    new_axioms.append(existing)
            else:
                # 新しい等価式を生成
                axiom = CrossAxiom(
                    axiom_id=self._axiom_counter,
                    trigger_sig=sig,
                    effect_sig=event.signature(),
                    observations=len(same_sig_indices),
                )
                self._axiom_counter += 1

                # パターンからゲームルールを推論
                rule_type = self._infer_rule_type(event, same_sig_indices)
                if rule_type:
                    axiom.jcross_rule = self._generate_jcross_rule_from_type(rule_type, event)
                    self.discovered_rules[rule_type] = axiom.jcross_rule
                    if len(same_sig_indices) >= 3:
                        axiom.confirmed = True
                    new_axioms.append(axiom)

                self.axioms.append(axiom)

        return new_axioms

    def _infer_rule_type(self, event: CrossEvent, indices: List[int]) -> Optional[str]:
        """事象パターンからゲームルールの種類を推論"""

        # パターン1: 色サイクル（クリックで色が変わる）
        if event.color_delta.get('transition_type') == 'cycle':
            if event.relation_delta.get('player_caused'):
                return 'click_color_cycle'

        # パターン2: 壁の開閉（特定の色が消える/現れる）
        if event.color_delta.get('transition_type') == 'swap':
            dom = event.color_delta.get('dominant', (0, 0))
            if isinstance(dom, tuple) and len(dom) == 2:
                if event.scale_delta.get('size_change') in ('medium', 'large'):
                    return 'wall_toggle'

        # パターン3: ブロック状の変化（パターンマッチのヒント）
        if event.shape_delta.get('type_change') == 'block':
            if event.scale_delta.get('size_change') in ('small', 'medium'):
                return 'block_state_change'

        # パターン4: 遠隔変化（プレイヤーから離れた場所の変化）
        if event.position_delta.get('region') == 'far':
            if event.relation_delta.get('adjacency_change') == 'remote':
                return 'remote_effect'

        # パターン5: 周期的変化（タイマーやアニメーション）
        if event.temporal_delta.get('periodic'):
            return 'periodic_animation'

        return None

    def _generate_jcross_rule_from_type(self, rule_type: str, event: CrossEvent) -> str:
        """ゲームルールの種類からjcross等価式を生成"""

        if rule_type == 'click_color_cycle':
            colors = event.color_delta.get('cycle_colors', [])
            return (
                f"// 発見: クリック $= 色サイクル\n"
                f"// サイクル色: {colors}\n"
                f"// [RULE:色サイクル START]\n"
                f"関数 色サイクル判定(現在色, 目標色, サイクル長) {{\n"
                f"  もし 現在色 == 目標色 {{\n"
                f"    返す 0\n"
                f"  }}\n"
                f"  // クリック回数 = 目標までの距離\n"
                f"  返す 1\n"
                f"}}\n"
                f"// [RULE:色サイクル END]"
            )

        elif rule_type == 'wall_toggle':
            bbox = event.shape_delta.get('bbox', (0, 0, 0, 0))
            return (
                f"// 発見: 壁開閉 $= 接触トリガー\n"
                f"// 領域: {bbox}\n"
                f"// [RULE:壁開閉 START]\n"
                f"関数 壁開閉を検知() {{\n"
                f"  もし 差分あり == 真 {{\n"
                f"    返す 壁開放方向\n"
                f"  }}\n"
                f"  返す -1\n"
                f"}}\n"
                f"// [RULE:壁開閉 END]"
            )

        elif rule_type == 'block_state_change':
            return (
                f"// 発見: ブロック状態変化 $= 行動の効果\n"
                f"// [RULE:状態変化 START]\n"
                f"// ブロックの色変化を追跡中\n"
                f"// [RULE:状態変化 END]"
            )

        elif rule_type == 'remote_effect':
            return (
                f"// 発見: 遠隔効果 $= プレイヤー行動の間接効果\n"
                f"// [RULE:遠隔効果 START]\n"
                f"// 遠隔変化を追跡中\n"
                f"// [RULE:遠隔効果 END]"
            )

        elif rule_type == 'periodic_animation':
            interval = event.temporal_delta.get('interval', 0)
            return (
                f"// 発見: 周期的変化 $= 自動アニメーション (間隔={interval}フレーム)\n"
                f"// → この変化は無視してよい\n"
            )

        return ""

    def _generate_jcross_rule(self, axiom: CrossAxiom, event: CrossEvent) -> str:
        """確定した等価式からjcrossルールを生成"""
        sig = axiom.trigger_sig
        return (
            f"// 確定等価式: {sig[0]}×{sig[1]}×{sig[2]} $= 行動{event.trigger_action}の効果\n"
            f"// 観測回数: {axiom.observations}\n"
        )

    def get_jcross_rules(self) -> str:
        """確定した全等価式をjcross形式で返す"""
        rules = []
        for axiom in self.axioms:
            if axiom.confirmed and axiom.jcross_rule:
                rules.append(axiom.jcross_rule)
        return "\n".join(rules)

    def get_summary(self) -> Dict:
        """現在の状態サマリー"""
        return {
            'events': len(self.events),
            'axioms': len(self.axioms),
            'confirmed': sum(1 for a in self.axioms if a.confirmed),
            'discovered_rules': list(self.discovered_rules.keys()),
            'causal_axioms': len(self.causal_axioms),
            'causal_confirmed': sum(1 for a in self.causal_axioms if a.confirmed),
        }

    # ========================================
    # 因果的等価式: action × context $= effect
    # ========================================

    def process_frame_causal(self, frame, prev_grid, curr_grid,
                              player_pos, prev_player_pos, action_idx,
                              corridor_colors=None) -> Optional['CausalAxiom']:
        """action→diffの因果関係を6軸で捕捉 — v2: 粒度改善版"""

        prev_g = np.array(prev_grid) if not isinstance(prev_grid, np.ndarray) else prev_grid
        curr_g = np.array(curr_grid) if not isinstance(curr_grid, np.ndarray) else curr_grid

        pr, pc = player_pos if player_pos else (32, 32)
        ppr, ppc = prev_player_pos if prev_player_pos else (pr, pc)

        # === コンテキスト: action方向を見る ===

        # action方向の移動ベクトル（probe結果から推定）
        move_vectors = {0: (-5, 0), 1: (5, 0), 2: (0, -5), 3: (0, 5)}
        dv = move_vectors.get(action_idx, (0, 0))

        # action方向の5セル先の色（壁か通路か）
        front_r = max(0, min(59, pr + dv[0]))
        front_c = max(0, min(63, pc + dv[1]))
        color_ahead = int(prev_g[front_r, front_c]) if 0 <= front_r < 60 and 0 <= front_c < 64 else -1

        # action方向が通路か壁か
        is_corridor_ahead = color_ahead in corridor_colors if corridor_colors else False

        # プレイヤー足元の色
        color_under = int(prev_g[pr, pc]) if 0 <= pr < 64 and 0 <= pc < 64 else -1

        # 移動したか
        moved = (pr != ppr or pc != ppc)
        move_delta = (pr - ppr, pc - ppc)

        # コンテキスト（v2: action方向ベース）
        context_sig = (
            action_idx,                                          # どの行動
            'corridor' if is_corridor_ahead else 'wall',        # 前方が通路か壁か
            color_ahead,                                         # 前方の色
            color_under,                                         # 足元の色
        )

        # === 効果: 色遷移パターンで分類 ===

        diff_mask = prev_g[:60] != curr_g[:60]
        changed_cells = list(zip(*np.where(diff_mask))) if np.any(diff_mask) else []

        if not changed_cells and not moved:
            effect_type = 'blocked'
            effect_sig = ('blocked', color_ahead)  # 何色でブロックされたか
            effect_details: Dict[str, Any] = {'type': 'blocked', 'blocking_color': color_ahead}

        elif not changed_cells and moved:
            effect_type = 'moved'
            effect_sig = ('moved', move_delta[0], move_delta[1])
            effect_details = {'type': 'moved', 'delta': move_delta}

        else:
            # 色遷移を分析
            transitions: Dict[Tuple[int, int], int] = defaultdict(int)
            for r, c in changed_cells:
                transitions[(int(prev_g[r, c]), int(curr_g[r, c]))] += 1

            n_changed = len(changed_cells)

            # 色サイクル検出: 同一色→別色の一括変化
            unique_transitions = set(transitions.keys())
            is_single_swap = len(unique_transitions) == 1
            is_color_cycle = len(unique_transitions) >= 2 and all(
                cnt >= 4 for cnt in transitions.values()  # 各遷移が4セル以上
            )

            # 壁開放/閉鎖の検出
            wall_opened = False
            wall_closed = False
            if corridor_colors:
                for (old, new), cnt in transitions.items():
                    if old not in corridor_colors and new in corridor_colors:
                        wall_opened = True
                    elif old in corridor_colors and new not in corridor_colors:
                        wall_closed = True

            # 効果の分類
            if wall_opened:
                # 壁開放: 主要な遷移の色を記録
                main_trans = max(transitions.items(), key=lambda x: x[1])
                effect_type = 'wall_open'
                effect_sig = ('wall_open', main_trans[0][0], main_trans[0][1])
                effect_details = {'type': 'wall_open', 'transitions': dict(transitions), 'n': n_changed}

            elif wall_closed:
                main_trans = max(transitions.items(), key=lambda x: x[1])
                effect_type = 'wall_close'
                effect_sig = ('wall_close', main_trans[0][0], main_trans[0][1])
                effect_details = {'type': 'wall_close', 'transitions': dict(transitions), 'n': n_changed}

            elif is_color_cycle or is_single_swap:
                # 色変化: 遷移パターンを記録
                sorted_trans = tuple(sorted(unique_transitions))
                effect_type = 'color_change'
                effect_sig = ('color_change', sorted_trans)
                effect_details = {
                    'type': 'color_change',
                    'transitions': dict(transitions),
                    'is_cycle': is_color_cycle,
                    'n': n_changed,
                }
                # 色サイクルの蓄積
                self._accumulate_color_cycle(transitions)

            else:
                # その他の変化: 主要遷移で分類
                main_trans = max(transitions.items(), key=lambda x: x[1])
                effect_type = 'grid_change'
                effect_sig = ('grid_change', main_trans[0][0], main_trans[0][1])
                effect_details = {'type': 'grid_change', 'transitions': dict(transitions), 'n': n_changed}

        # === 因果等価式の検索/生成 ===

        existing = None
        for ax in self.causal_axioms:
            if ax.context_sig == context_sig:
                if ax.effect_sig == effect_sig:
                    existing = ax
                    break
                else:
                    ax.contradictions += 1

        new_axiom = None
        if existing:
            existing.observations += 1
            if existing.observations >= 3 and not existing.confirmed:
                existing.confirmed = True
                existing.jcross_equiv = self._generate_causal_jcross(existing)
                new_axiom = existing
        else:
            axiom = CausalAxiom(
                axiom_id=self._causal_counter,
                action_idx=action_idx,
                context_sig=context_sig,
                effect_sig=effect_sig,
                effect_details=effect_details,
            )
            self._causal_counter += 1
            self.causal_axioms.append(axiom)

        return new_axiom

    def _accumulate_color_cycle(self, transitions: Dict[Tuple[int, int], int]) -> None:
        """色遷移からサイクルを自動構築
        (3→5), (5→7) を蓄積 → [3,5,7] のサイクルを検出
        """
        for (old, new), cnt in transitions.items():
            if cnt >= 4:  # 4セル以上の一括変化のみ
                self._color_edges[(old, new)] += 1

        # サイクル検出: A→B→C→A を探す
        colors_seen: Set[int] = set()
        for (a, b) in self._color_edges:
            colors_seen.add(a)
            colors_seen.add(b)

        for start in colors_seen:
            cycle = [start]
            current = start
            visited = {start}
            while True:
                # currentから出る辺を探す
                next_color = None
                for (a, b), cnt in self._color_edges.items():
                    if a == current and b not in visited and cnt >= 2:
                        next_color = b
                        break
                if next_color is None:
                    # currentからstartへの辺があればサイクル完成
                    if (current, start) in self._color_edges and len(cycle) >= 2:
                        if sorted(cycle) not in [sorted(c) for c in self._detected_cycles]:
                            self._detected_cycles.append(cycle)
                            print(f"COLOR_CYCLE_DETECTED: {cycle}", file=sys.stderr)
                    break
                cycle.append(next_color)
                visited.add(next_color)
                current = next_color
                if len(cycle) > 10:
                    break

    def get_detected_cycles(self) -> List[List[int]]:
        """検出された色サイクルを返す"""
        return self._detected_cycles

    def detect_game_type(self, is_click_game, action_model=None) -> str:
        """因果等価式のパターンからゲーム種類を判定

        Returns:
            'maze' — 移動メインのゲーム（ls20, m0r0）
            'click_cycle' — 色サイクルクリックゲーム（ft09）
            'click_toggle' — トグルクリックゲーム
            'unknown'
        """
        if not self.causal_axioms:
            return 'unknown'

        # 統計
        n_blocked = sum(1 for a in self.causal_axioms if a.effect_sig[0] == 'blocked')
        n_moved = sum(1 for a in self.causal_axioms if a.effect_sig[0] == 'moved')
        n_wall_open = sum(1 for a in self.causal_axioms if a.effect_sig[0] == 'wall_open')
        n_color_change = sum(1 for a in self.causal_axioms if a.effect_sig[0] == 'color_change')
        n_total = len(self.causal_axioms)

        # 色サイクルが検出されていればclick_cycle
        if self.get_detected_cycles():
            if is_click_game:
                return 'click_cycle'

        # clickゲームで色変化が多ければclick_toggle
        if is_click_game and n_color_change > n_total * 0.3:
            return 'click_toggle'

        # keyboard(非click)でblocked+movedが多ければmaze
        if not is_click_game:
            return 'maze'

        # clickゲームだがパターン不明
        if is_click_game:
            return 'click_toggle'

        return 'unknown'

    def simulate(self, action_idx: int, player_pos, grid,
                 corridor_colors=None) -> Optional[Dict[str, Any]]:
        """確定等価式から行動の結果を予測 — v2"""
        g = np.array(grid) if not isinstance(grid, np.ndarray) else grid
        pr, pc = player_pos if player_pos else (32, 32)

        # action方向
        move_vectors = {0: (-5, 0), 1: (5, 0), 2: (0, -5), 3: (0, 5)}
        dv = move_vectors.get(action_idx, (0, 0))
        front_r = max(0, min(59, pr + dv[0]))
        front_c = max(0, min(63, pc + dv[1]))
        color_ahead = int(g[front_r, front_c]) if 0 <= front_r < 60 and 0 <= front_c < 64 else -1
        is_corridor = color_ahead in corridor_colors if corridor_colors else False
        color_under = int(g[pr, pc]) if 0 <= pr < 64 and 0 <= pc < 64 else -1

        context_sig = (
            action_idx,
            'corridor' if is_corridor else 'wall',
            color_ahead,
            color_under,
        )

        # 完全一致を探す
        for ax in self.causal_axioms:
            if ax.confirmed and ax.context_sig == context_sig:
                return {
                    'predicted_effect': ax.effect_sig[0],
                    'confidence': min(ax.observations / 5.0, 1.0),
                    'details': ax.effect_details,
                    'axiom_id': ax.axiom_id,
                }

        # 部分一致: 同じaction + 同じ前方状態（色は違っても壁/通路が同じ）
        for ax in self.causal_axioms:
            if ax.confirmed and ax.context_sig[0] == action_idx and ax.context_sig[1] == context_sig[1]:
                return {
                    'predicted_effect': ax.effect_sig[0],
                    'confidence': min(ax.observations / 10.0, 0.5),
                    'details': ax.effect_details,
                    'axiom_id': ax.axiom_id,
                }

        return None

    def get_best_action(self, player_pos, grid, corridor_colors=None,
                        available_actions=None) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        """全行動シミュレーション → blocked/no_effectを避ける最良行動"""
        if not available_actions:
            available_actions = [0, 1, 2, 3]

        predictions = []
        for aidx in available_actions:
            pred = self.simulate(aidx, player_pos, grid, corridor_colors)
            if pred:
                predictions.append((aidx, pred))

        if not predictions:
            return None, None

        # blocked/no_effectは除外
        good = [(a, p) for a, p in predictions
                if p['predicted_effect'] not in ('blocked', 'no_effect')]
        if not good:
            return None, None

        priority = {'wall_open': 100, 'color_change': 80, 'moved': 50,
                    'grid_change': 30, 'wall_close': 10}

        best = max(good, key=lambda x: priority.get(x[1]['predicted_effect'], 0) * x[1]['confidence'])
        return best

    def _generate_causal_jcross(self, axiom: 'CausalAxiom') -> str:
        """因果等価式から実行可能なjcrossコードを生成（テンプレートエンジン）"""
        action_names = {0: '上', 1: '下', 2: '左', 3: '右', 4: '行動5', 5: 'クリック'}
        action_name = action_names.get(axiom.action_idx, f'行動{axiom.action_idx}')
        effect = axiom.effect_details.get('type', 'unknown')

        # コンテキストから条件を抽出
        # context_sig = (action_idx, 'corridor'/'wall', color_ahead, color_under)
        ahead_type = axiom.context_sig[1]  # 'corridor' or 'wall'
        color_ahead = axiom.context_sig[2]
        color_under = axiom.context_sig[3]

        lines = []
        lines.append(f"// 因果等価式: {action_name} × {ahead_type} × 色{color_ahead}")
        lines.append(f"//   $= {effect} (観測{axiom.observations}回)")

        if effect == 'blocked':
            # ブロック: この色の方向を避ける
            alt_action = (axiom.action_idx + 2) % 4  # 反対方向
            alt2 = (axiom.action_idx + 1) % 4  # 90度
            lines.append(f"関数 回避_{axiom.axiom_id}(方向, 前方色) {{")
            lines.append(f"  もし 方向 == {axiom.action_idx} {{")
            lines.append(f"    もし 前方色 == {color_ahead} {{")
            lines.append(f"      返す {alt_action}")
            lines.append(f"    }}")
            lines.append(f"  }}")
            lines.append(f"  返す -1")
            lines.append(f"}}")

        elif effect == 'wall_open':
            # 壁開放: この方向+色の組み合わせで壁が開く → この方向を優先
            lines.append(f"関数 壁開放_{axiom.axiom_id}(方向, 前方色) {{")
            lines.append(f"  もし 方向 == {axiom.action_idx} {{")
            lines.append(f"    もし 前方色 == {color_ahead} {{")
            lines.append(f"      返す {axiom.action_idx}")
            lines.append(f"    }}")
            lines.append(f"  }}")
            lines.append(f"  返す -1")
            lines.append(f"}}")

        elif effect == 'color_change':
            # 色変化: この方向で色が変わる（クリックゲームの手がかり）
            transitions = axiom.effect_details.get('transitions', {})
            trans_str = str(list(transitions.keys())[:3])
            lines.append(f"// 色変化パターン: {trans_str}")
            lines.append(f"関数 色変化_{axiom.axiom_id}(方向, 前方色) {{")
            lines.append(f"  もし 方向 == {axiom.action_idx} {{")
            lines.append(f"    返す {axiom.action_idx}")
            lines.append(f"  }}")
            lines.append(f"  返す -1")
            lines.append(f"}}")

        elif effect == 'moved':
            # 移動成功: この方向+色は安全
            delta = axiom.effect_details.get('delta', (0, 0))
            lines.append(f"// 移動量: ({delta[0]},{delta[1]})")
            lines.append(f"関数 安全移動_{axiom.axiom_id}(方向, 前方色) {{")
            lines.append(f"  もし 方向 == {axiom.action_idx} {{")
            lines.append(f"    もし 前方色 == {color_ahead} {{")
            lines.append(f"      返す {axiom.action_idx}")
            lines.append(f"    }}")
            lines.append(f"  }}")
            lines.append(f"  返す -1")
            lines.append(f"}}")

        else:
            # その他: コメントのみ
            lines.append(f"// 未分類の効果: {effect}")

        return '\n'.join(lines)
