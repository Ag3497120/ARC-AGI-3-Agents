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
        """action→diffの因果関係を6軸で捕捉"""

        prev_g = np.array(prev_grid) if not isinstance(prev_grid, np.ndarray) else prev_grid
        curr_g = np.array(curr_grid) if not isinstance(curr_grid, np.ndarray) else curr_grid

        # === 行動のコンテキストを6軸で記述 ===

        # プレイヤーの位置コンテキスト
        pr, pc = player_pos if player_pos else (32, 32)
        player_region = ('top' if pr < 20 else ('mid' if pr < 40 else 'bottom'),
                         'left' if pc < 21 else ('center' if pc < 43 else 'right'))

        # プレイヤー足元の色
        color_under = int(prev_g[pr, pc]) if 0 <= pr < 64 and 0 <= pc < 64 else -1

        # プレイヤー周囲の色セット（5セル以内）
        nearby_colors = set()
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < 60 and 0 <= nc < 64:
                    nearby_colors.add(int(prev_g[nr, nc]))
        nearby_sig = tuple(sorted(nearby_colors))[:5]  # 上位5色

        # 移動したか
        moved = prev_player_pos and player_pos and (prev_player_pos != player_pos)
        move_delta = (player_pos[0] - prev_player_pos[0], player_pos[1] - prev_player_pos[1]) if moved else (0, 0)

        # コンテキストシグネチャ（因果の左辺の条件部分）
        context_sig = (
            player_region,     # どの領域にいた時
            color_under,       # 何色の上にいた時
            nearby_sig[:3],    # 周囲にどんな色があった時
        )

        # === 効果を6軸で記述 ===

        diff_mask = prev_g[:60] != curr_g[:60]
        changed_cells = list(zip(*np.where(diff_mask))) if np.any(diff_mask) else []

        if not changed_cells and not moved:
            # 何も起きなかった（ブロックされた）
            effect_sig = ('no_effect', 'blocked', action_idx)
            effect_details: Dict[str, Any] = {'type': 'blocked', 'action': action_idx, 'position': (pr, pc)}

        elif not changed_cells and moved:
            # 移動のみ（グリッド変化なし）
            effect_sig = ('move_only', f'dr={move_delta[0]}', f'dc={move_delta[1]}')
            effect_details = {'type': 'player_move', 'delta': move_delta}

        else:
            # グリッドが変化した
            color_transitions: Dict[Tuple[int, int], int] = defaultdict(int)
            for r, c in changed_cells:
                color_transitions[(int(prev_g[r, c]), int(curr_g[r, c]))] += 1

            # 変化の領域
            rows = [r for r, c in changed_cells]
            cols = [c for r, c in changed_cells]
            change_region = (min(rows), min(cols), max(rows), max(cols))

            # 変化の種類を判定
            n_changed = len(changed_cells)

            # 壁開放: 壁色→通路色の変化
            wall_opened = False
            wall_closed = False
            if corridor_colors:
                for (old, new), cnt in color_transitions.items():
                    if old not in corridor_colors and new in corridor_colors:
                        wall_opened = True
                    elif old in corridor_colors and new not in corridor_colors:
                        wall_closed = True

            if wall_opened:
                change_type = 'wall_open'
            elif wall_closed:
                change_type = 'wall_close'
            elif n_changed < 50:
                change_type = 'block_change'
            else:
                change_type = 'large_change'

            effect_sig = (change_type, f'n={n_changed}', f'region={change_region[:2]}')
            effect_details = {
                'type': change_type,
                'region': change_region,
                'transitions': dict(color_transitions),
                'n_changed': n_changed,
                'player_moved': moved,
                'move_delta': move_delta,
            }

        # === 因果等価式の検索/生成 ===

        # 既存の等価式を探す
        existing = None
        for ax in self.causal_axioms:
            if ax.action_idx == action_idx and ax.context_sig == context_sig:
                # 同じ効果か？
                if ax.effect_sig == effect_sig:
                    existing = ax
                    break
                else:
                    # 矛盾（同じ行動×コンテキストで違う効果）
                    ax.contradictions += 1
                    existing = None
                    break

        new_axiom = None
        if existing:
            existing.observations += 1
            if existing.observations >= 3 and not existing.confirmed:
                existing.confirmed = True
                existing.jcross_equiv = self._generate_causal_jcross(existing)
                new_axiom = existing
        else:
            # 新しい因果等価式
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

    def simulate(self, action_idx: int, player_pos, grid,
                 corridor_colors=None) -> Optional[Dict[str, Any]]:
        """確定した因果等価式を使って、行動の結果を予測する

        Returns:
            {'predicted_effect': str, 'confidence': float, 'details': dict}
            None if no matching axiom
        """
        # 現在のコンテキストを計算
        g = np.array(grid) if not isinstance(grid, np.ndarray) else grid
        pr, pc = player_pos if player_pos else (32, 32)
        player_region = ('top' if pr < 20 else ('mid' if pr < 40 else 'bottom'),
                         'left' if pc < 21 else ('center' if pc < 43 else 'right'))
        color_under = int(g[pr, pc]) if 0 <= pr < 64 and 0 <= pc < 64 else -1
        nearby: Set[int] = set()
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < 60 and 0 <= nc < 64:
                    nearby.add(int(g[nr, nc]))
        nearby_sig = tuple(sorted(nearby))[:5]

        context_sig = (player_region, color_under, nearby_sig[:3])

        # 確定等価式を検索
        for ax in self.causal_axioms:
            if ax.confirmed and ax.action_idx == action_idx:
                # コンテキストが完全一致
                if ax.context_sig == context_sig:
                    return {
                        'predicted_effect': ax.effect_sig[0],
                        'confidence': min(ax.observations / 5.0, 1.0),
                        'details': ax.effect_details,
                        'axiom_id': ax.axiom_id,
                    }
                # コンテキストが部分一致（同じ領域、色は違う）
                if ax.context_sig[0] == context_sig[0]:
                    return {
                        'predicted_effect': ax.effect_sig[0],
                        'confidence': min(ax.observations / 10.0, 0.5),
                        'details': ax.effect_details,
                        'axiom_id': ax.axiom_id,
                    }

        return None

    def get_best_action(self, player_pos, grid, corridor_colors=None,
                        available_actions=None) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        """全行動をシミュレーションして最良の行動を選ぶ

        Returns:
            (best_action_idx, prediction) or (None, None)
        """
        if not available_actions:
            available_actions = [0, 1, 2, 3]

        predictions = []
        for aidx in available_actions:
            pred = self.simulate(aidx, player_pos, grid, corridor_colors)
            if pred:
                predictions.append((aidx, pred))

        if not predictions:
            return None, None

        # 優先度: wall_open > player_move > block_change > no_effect > blocked
        priority = {'wall_open': 100, 'player_move': 50, 'block_change': 30,
                    'large_change': 20, 'wall_close': 10, 'no_effect': -1, 'blocked': -10}

        # no_effect/blockedは除外（避けるべき行動）
        positive = [(a, p) for a, p in predictions 
                    if priority.get(p['predicted_effect'], 0) > 0]
        if not positive:
            return None, None
        best = max(positive,
                   key=lambda x: priority.get(x[1]['predicted_effect'], 0) * x[1]['confidence'])
        return best

    def _generate_causal_jcross(self, axiom: 'CausalAxiom') -> str:
        """因果等価式をjcross形式に変換"""
        action_names = {0: '上', 1: '下', 2: '左', 3: '右', 4: '行動5', 5: 'クリック'}
        action_name = action_names.get(axiom.action_idx, f'行動{axiom.action_idx}')

        region = axiom.context_sig[0]
        color = axiom.context_sig[1]
        effect = axiom.effect_details.get('type', 'unknown')

        lines = []
        lines.append(f"// 因果等価式: {action_name} × 領域{region} × 色{color}")
        lines.append(f"//   $= {effect}")
        lines.append(f"// 観測: {axiom.observations}回 確定: {axiom.confirmed}")

        if effect == 'wall_open':
            r = axiom.effect_details.get('region', (0, 0, 0, 0))
            lines.append(f"// 壁開放領域: ({r[0]},{r[1]})-({r[2]},{r[3]})")
        elif effect == 'player_move':
            d = axiom.effect_details.get('delta', (0, 0))
            lines.append(f"// 移動量: ({d[0]},{d[1]})")
        elif effect == 'blocked':
            lines.append(f"// この方向はブロックされる")

        return '\n'.join(lines)
