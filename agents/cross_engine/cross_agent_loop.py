"""
cross_agent_loop.py — SLMがCross空間を能動的に操作するReActエージェント

kofdai型:
- LLMは候補を提案
- Cross Simulatorで検証
- 最良のプログラムを選択

SLMは5つのツールを使って自律的にCross空間を操作:
1. cross_observe — グリッドを6軸Crossで観察
2. cross_query — Cross空間内の体験を検索
3. jcross_write — soul.jcrossに新しいルールを書き込む
4. jcross_simulate — jcrossコードをテスト実行する
5. cross_plan — 行動計画を生成する
"""

import json
import sys
import urllib.request
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from collections import Counter, defaultdict


class CrossAgentLoop:
    """SLMがCross空間を能動的に操作するエージェントループ"""

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "qwen2.5:1.5b"
    MAX_STEPS = 3  # 1ゲームあたり最大3ステップ（レイテンシ制限）

    def __init__(self):
        self._available = self._check_ollama()
        self._cache = {}
        self._discoveries = {}  # ゲーム中に発見したルール
        self._jcross_parser = None
        self._init_parser()

    def _check_ollama(self):
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def _init_parser(self):
        try:
            import os
            sys.path.insert(0, os.path.expanduser('~/verantyx_v6/bootstrap'))
            from jcross_japanese_parser import JCrossJapaneseParser
            self._jcross_parser = JCrossJapaneseParser
        except ImportError:
            self._jcross_parser = None

    @property
    def is_available(self):
        return self._available

    def _call_slm(self, prompt, max_tokens=200):
        if not self._available:
            return None
        key = prompt[:300]
        if key in self._cache:
            return self._cache[key]
        try:
            data = json.dumps({
                "model": self.MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1}
            }).encode()
            req = urllib.request.Request(
                self.OLLAMA_URL, data=data,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                ans = json.loads(r.read()).get("response", "").strip()
                self._cache[key] = ans
                return ans
        except Exception:
            return None

    # ========================================
    # 5つのツール
    # ========================================

    def tool_cross_observe(self, grid, player_pos, prev_grid=None):
        """ツール1: グリッドを6軸Crossで観察"""
        g = np.array(grid) if not isinstance(grid, np.ndarray) else grid
        rows, cols = g.shape

        # 色の分布
        color_counts = Counter(int(v) for v in g[:min(60, rows)].flatten())
        bg_color = color_counts.most_common(1)[0][0]
        rare_colors = [c for c, n in color_counts.items() if n < rows * cols * 0.01 and c != bg_color]

        # ブロック検出（非背景色の連結領域）
        blocks = []
        visited = set()
        for r in range(min(60, rows)):
            for c in range(cols):
                color = int(g[r, c])
                if color != bg_color and (r, c) not in visited:
                    # BFS for connected component
                    block_cells = []
                    queue = [(r, c)]
                    visited.add((r, c))
                    while queue:
                        cr, cc = queue.pop(0)
                        block_cells.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < min(60, rows) and 0 <= nc < cols
                                    and (nr, nc) not in visited
                                    and int(g[nr, nc]) == color):
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                    if 4 <= len(block_cells) <= 200:  # 意味のあるサイズ
                        rr = [r for r, c in block_cells]
                        cc = [c for r, c in block_cells]
                        blocks.append({
                            'color': color,
                            'center': ((min(rr) + max(rr)) // 2, (min(cc) + max(cc)) // 2),
                            'size': len(block_cells),
                            'bbox': (min(rr), min(cc), max(rr), max(cc)),
                        })

        # 左右の対称性検出（ft09型パターンマッチ検出）
        mid_col = cols // 2
        left_blocks = [b for b in blocks if b['center'][1] < mid_col - 2]
        right_blocks = [b for b in blocks if b['center'][1] > mid_col + 2]
        has_lr_symmetry = (
            len(left_blocks) > 2 and len(right_blocks) > 2
            and abs(len(left_blocks) - len(right_blocks)) <= 2
        )

        # 差分（前フレームとの比較）
        diff_info = {}
        if prev_grid is not None:
            prev_g = np.array(prev_grid) if not isinstance(prev_grid, np.ndarray) else prev_grid
            if prev_g.shape == g.shape:
                diff = prev_g[:min(60, rows)] != g[:min(60, rows)]
                changed = int(np.sum(diff))
                if changed > 0:
                    transitions = defaultdict(int)
                    for r, c in zip(*np.where(diff)):
                        transitions[(int(prev_g[r, c]), int(g[r, c]))] += 1
                    diff_info = {
                        'changed_cells': changed,
                        'transitions': dict(transitions),
                    }

        observation = {
            'grid_size': f'{rows}x{cols}',
            'bg_color': bg_color,
            'num_blocks': len(blocks),
            'rare_colors': rare_colors,
            'has_lr_symmetry': has_lr_symmetry,
            'left_blocks': len(left_blocks),
            'right_blocks': len(right_blocks),
            'player_pos': list(player_pos) if player_pos else [32, 32],
            'blocks': blocks[:20],  # 上位20個
            'diff': diff_info,
        }

        return observation

    def tool_cross_query(self, axiom_engine, query_type):
        """ツール2: Cross空間内の体験を検索"""
        if not axiom_engine:
            return {'events': 0, 'axioms': 0, 'rules': {}}

        result = axiom_engine.get_summary()

        if query_type == 'color_cycles':
            cycles = []
            for event in axiom_engine.events:
                if event.color_delta.get('transition_type') == 'cycle':
                    cycles.append({
                        'colors': event.color_delta.get('cycle_colors', []),
                        'frame': event.frame,
                        'action': event.trigger_action,
                    })
            result['color_cycles'] = cycles

        elif query_type == 'block_changes':
            changes = []
            for event in axiom_engine.events:
                if event.shape_delta.get('type_change') == 'block':
                    changes.append({
                        'bbox': event.shape_delta.get('bbox'),
                        'frame': event.frame,
                        'transitions': event.color_delta.get('transitions', {}),
                    })
            result['block_changes'] = changes

        return result

    def tool_jcross_write(self, jcross_runtime, rule_name, rule_code):
        """ツール3: soul.jcrossに新しいルールを書き込む"""
        if not jcross_runtime:
            return {'success': False, 'reason': 'no runtime'}

        try:
            success = jcross_runtime.rewrite_rule(rule_name, rule_code)
            return {'success': success, 'rule_name': rule_name, 'code_len': len(rule_code)}
        except Exception as e:
            return {'success': False, 'reason': str(e)}

    def tool_jcross_simulate(self, jcross_code, test_vars=None):
        """ツール4: jcrossコードをテスト実行する"""
        if not self._jcross_parser:
            return {'success': False, 'reason': 'no parser'}

        try:
            parser = self._jcross_parser()
            if test_vars:
                for k, v in test_vars.items():
                    parser.globals[k] = v
            parser.execute(jcross_code)
            return {
                'success': True,
                'globals': {
                    k: v for k, v in parser.globals.items()
                    if not k.startswith('_') and not callable(v)
                },
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def tool_cross_plan(self, observation, axioms, game_type):
        """ツール5: 行動計画を生成する"""
        plan = {'actions': [], 'game_type': game_type}

        if game_type == 'click_pattern_match':
            obs = observation
            if obs.get('has_lr_symmetry'):
                left = sorted(
                    [b for b in obs.get('blocks', []) if b['center'][1] < 32],
                    key=lambda b: (b['center'][0], b['center'][1])
                )
                right = sorted(
                    [b for b in obs.get('blocks', []) if b['center'][1] > 32],
                    key=lambda b: (b['center'][0], b['center'][1])
                )

                # 色サイクル情報を取得
                cycle_colors = []
                for rule_type, info in axioms.items():
                    if 'cycle' in rule_type:
                        cycle_colors = info.get('cycle_colors', [])
                        break

                if cycle_colors and left and right:
                    cycle_len = len(cycle_colors)
                    for lb, rb in zip(left, right):
                        if lb['color'] != rb['color']:
                            if lb['color'] in cycle_colors and rb['color'] in cycle_colors:
                                target_idx = cycle_colors.index(lb['color'])
                                current_idx = cycle_colors.index(rb['color'])
                                clicks = (target_idx - current_idx + cycle_len) % cycle_len
                                if clicks > 0:
                                    plan['actions'].append({
                                        'type': 'click',
                                        'pos': rb['center'],
                                        'clicks': clicks,
                                        'reason': f'color {rb["color"]}→{lb["color"]} ({clicks} clicks)',
                                    })

        elif game_type == 'maze':
            plan['actions'].append({'type': 'delegate_to_python', 'reason': 'maze solver'})

        return plan

    # ========================================
    # ReActループ
    # ========================================

    def run(self, grid, player_pos, prev_grid, axiom_engine, jcross_runtime,
            frame_num, is_click_game):
        """
        SLMが自律的にツールを使ってCross空間を操作するReActループ。

        呼ばれるタイミング:
        - ゲーム開始時（observe直後、1回のみ）
        - axiomが確定した時

        Returns:
            {
                'jcross_rules': [{'name': str, 'code': str}],  # 書き込むべきルール
                'click_plan': [{'pos': (r,c), 'clicks': n}],   # クリック計画
                'game_type': str,                               # 推定されたゲーム種類
                'discoveries': dict,                            # 発見事項
            }
        """
        if not self._available:
            return None

        result = {
            'jcross_rules': [],
            'click_plan': [],
            'game_type': 'unknown',
            'discoveries': {},
        }

        # === Step 1: Observe ===
        obs = self.tool_cross_observe(grid, player_pos, prev_grid)

        # === Step 2: SLMにゲームの種類を推定させる ===
        game_type_prompt = f"""You are analyzing a puzzle game grid.
Observation:
- Grid: {obs['grid_size']}, Background color: {obs['bg_color']}
- Blocks found: {obs['num_blocks']}
- Left-right symmetry: {obs['has_lr_symmetry']} (left={obs['left_blocks']}, right={obs['right_blocks']})
- Rare colors: {obs['rare_colors']}
- Is click game: {is_click_game}

What type of game is this? Answer with ONE word:
- pattern_match (left-right matching)
- maze (navigate to goal)
- toggle (click to toggle states)
- cycle (click to cycle colors)
- unknown

Answer:"""

        game_type_response = self._call_slm(game_type_prompt, max_tokens=10)
        if game_type_response:
            for gtype in ['pattern_match', 'maze', 'toggle', 'cycle', 'unknown']:
                if gtype in game_type_response.lower():
                    result['game_type'] = gtype
                    break

        # 左右対称性があればpattern_matchに上書き
        if obs.get('has_lr_symmetry') and is_click_game:
            result['game_type'] = 'click_pattern_match'

        print(
            f"AGENT_LOOP: game_type={result['game_type']} "
            f"blocks={obs['num_blocks']} lr_sym={obs['has_lr_symmetry']}",
            file=sys.stderr
        )

        # === Step 3: Cross空間をクエリ ===
        if axiom_engine:
            query_result = self.tool_cross_query(axiom_engine, 'color_cycles')

            if query_result.get('color_cycles'):
                all_cycle_colors = set()
                for cycle_info in query_result['color_cycles']:
                    all_cycle_colors.update(cycle_info.get('colors', []))
                if all_cycle_colors:
                    result['discoveries']['cycle_colors'] = sorted(all_cycle_colors)
                    print(
                        f"AGENT_LOOP: discovered cycle_colors={result['discoveries']['cycle_colors']}",
                        file=sys.stderr
                    )

        # === Step 4: ゲーム種類に応じた行動計画 ===
        if result['game_type'] == 'click_pattern_match':
            axiom_info = {}
            if result['discoveries'].get('cycle_colors'):
                axiom_info['cycle_colors'] = result['discoveries']['cycle_colors']

            plan = self.tool_cross_plan(
                obs,
                {'color_cycle': axiom_info} if axiom_info else {},
                'click_pattern_match'
            )

            if plan.get('actions'):
                for action in plan['actions']:
                    if action['type'] == 'click':
                        result['click_plan'].append({
                            'pos': action['pos'],
                            'clicks': action['clicks'],
                        })
                print(f"AGENT_LOOP: click_plan={len(result['click_plan'])} actions", file=sys.stderr)

        # === Step 5: jcrossルールを生成 ===
        if result['game_type'] not in ('unknown', 'maze'):
            jcross_prompt = f"""Write a jcross rule as a comment describing this game:
Game type: {result['game_type']}
Discoveries: {json.dumps(result['discoveries'], default=str)[:200]}

Use ONLY Japanese comments (// lines).
Under 5 lines."""

            slm_rule = self._call_slm(jcross_prompt, max_tokens=100)
            if slm_rule:
                clean_lines = []
                for line in slm_rule.split('\n'):
                    s = line.strip()
                    if (s.startswith('//') or s.startswith('関数')
                            or s.startswith('もし') or s.startswith('返す')):
                        clean_lines.append(s)
                if clean_lines:
                    rule_code = '\n'.join(clean_lines)
                    result['jcross_rules'].append({
                        'name': f'ゲーム分析_{result["game_type"]}',
                        'code': rule_code,
                    })

                    # jcross_simulateで検証
                    if self._jcross_parser:
                        sim_result = self.tool_jcross_simulate(rule_code)
                        if sim_result.get('success'):
                            print("AGENT_LOOP: jcross rule verified ✓", file=sys.stderr)
                        else:
                            print(
                                f"AGENT_LOOP: jcross rule failed: {sim_result.get('error', '')[:50]}",
                                file=sys.stderr
                            )

        self._discoveries = result['discoveries']
        return result

    def get_click_plan(self):
        """現在のクリック計画を返す（v26から呼ばれる）"""
        # run()の戻り値から取得する設計
        return None
