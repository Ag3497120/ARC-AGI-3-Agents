"""
slm_bridge.py — SLM(2B)をcross_axiomに接続する橋
axiomが発見した等価式を具体的行動に変換する

kofdaiの方針: AIは使わない（LLMなし）が原則だが、
2Bレベルの小さなモデルは「道具」として許容。
計算の延長。推論の延長。
"""

import json
import urllib.request
import urllib.error
from typing import Optional, Dict, List, Any


class SLMBridge:
    """
    axiomの発見 → SLM → 具体的行動計画

    呼ぶタイミング:
    - axiomが確定した時（数回/ゲーム）
    - 新しいゲームルールが発見された時
    - NOT 毎フレーム（遅すぎる）
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "qwen2.5:1.5b"

    def __init__(self):
        self._available = self._check_available()
        self._cache: Dict[str, str] = {}  # プロンプトハッシュ→回答キャッシュ

    def _check_available(self) -> bool:
        """ollamaが起動しているか確認"""
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def _call(self, prompt: str, max_tokens: int = 100) -> Optional[str]:
        """ollamaにプロンプトを送って回答を得る"""
        if not self._available:
            return None

        # キャッシュ確認
        cache_key = prompt[:200]
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            data = json.dumps({
                "model": self.MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.1,  # 決定論的に近く
                }
            }).encode('utf-8')

            req = urllib.request.Request(
                self.OLLAMA_URL,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                answer = result.get("response", "").strip()
                self._cache[cache_key] = answer
                return answer
        except Exception:
            return None

    def translate_axiom_to_action(self, axiom_type: str, axiom_details: Dict,
                                   grid_context: str) -> Optional[Dict]:
        """
        axiomの等価式を具体的行動計画に変換

        Returns:
            {"action_type": "click_sequence", "targets": [(r,c), ...], "clicks": [3, 1, 2]}
            {"action_type": "move_to", "target": (r, c), "direction": 0}
            None if conversion failed
        """

        if axiom_type == "click_color_cycle":
            return self._solve_color_cycle(axiom_details, grid_context)
        elif axiom_type == "wall_toggle":
            return self._solve_wall_toggle(axiom_details, grid_context)
        elif axiom_type == "block_state_change":
            return self._solve_block_state(axiom_details, grid_context)
        elif axiom_type == "remote_effect":
            return self._solve_remote_effect(axiom_details, grid_context)

        return None

    def _solve_color_cycle(self, details: Dict, grid_context: str) -> Optional[Dict]:
        """色サイクルパズルを解く"""
        cycle_colors = details.get("cycle_colors", [])
        if not cycle_colors:
            return None

        cycle_len = len(cycle_colors)

        # SLMに解かせるプロンプト（短く直接的に）
        prompt = f"""Color cycle: {' → '.join(str(c) for c in cycle_colors)} → {cycle_colors[0]} (length {cycle_len})
{grid_context}
List clicks needed per block. ONLY output lines like: row,col:clicks
No explanation."""

        response = self._call(prompt, max_tokens=200)
        if not response:
            return None

        # パース: "10,40:2\n20,40:1"
        click_plan = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if ':' in line:
                try:
                    pos_str, clicks_str = line.split(':')
                    parts = pos_str.strip().split(',')
                    r, c = int(parts[0].strip()), int(parts[1].strip())
                    clicks = int(clicks_str.strip())
                    if 0 <= r < 64 and 0 <= c < 64 and 0 < clicks < cycle_len:
                        click_plan.append({"pos": (r, c), "clicks": clicks})
                except (ValueError, IndexError):
                    continue

        if click_plan:
            return {
                "action_type": "click_sequence",
                "plan": click_plan,
                "cycle_colors": cycle_colors,
                "cycle_len": cycle_len,
            }

        return None

    def _solve_wall_toggle(self, details: Dict, grid_context: str) -> Optional[Dict]:
        """壁開閉パズルの行動提案"""
        prompt = f"""Wall puzzle. Touching areas toggles walls open/closed.
{grid_context}
Move direction to reach toggle? Answer ONE word: up, down, left, or right"""

        response = self._call(prompt, max_tokens=20)
        if not response:
            return None

        direction_map = {"up": 0, "down": 1, "left": 2, "right": 3}
        for word, idx in direction_map.items():
            if word in response.lower():
                return {"action_type": "move_to", "direction": idx}

        return None

    def _solve_block_state(self, details: Dict, grid_context: str) -> Optional[Dict]:
        """ブロック状態変化パズル"""
        # 汎用的な状態変化 — SLMに判断を委ねる
        prompt = f"""Block puzzle. Blocks change state when interacted with.
{grid_context}
Next action? Answer: click row,col OR move up/down/left/right. No explanation."""

        response = self._call(prompt, max_tokens=30)
        if not response:
            return None

        resp_lower = response.lower()
        if "click" in resp_lower:
            try:
                # "click 10,40" or "click at 10,40"
                parts = resp_lower.replace("click at", "click").replace("click", "").strip().split(',')
                r = int(parts[0].strip())
                c = int(parts[1].strip())
                return {"action_type": "click_at", "pos": (r, c)}
            except (ValueError, IndexError):
                pass

        if "move" in resp_lower:
            direction_map = {"up": 0, "down": 1, "left": 2, "right": 3}
            for word, idx in direction_map.items():
                if word in resp_lower:
                    return {"action_type": "move_to", "direction": idx}

        return None

    def _solve_remote_effect(self, details: Dict, grid_context: str) -> Optional[Dict]:
        """遠隔効果の解析"""
        return None  # 今は未実装

    def build_grid_context(self, grid, player_pos,
                           discovered_rules: Dict[str, str],
                           left_pattern=None, right_pattern=None) -> str:
        """グリッドのコンテキスト文字列を構築"""
        import numpy as np
        g = np.array(grid) if not isinstance(grid, np.ndarray) else grid

        lines = []
        lines.append(f"Grid: {g.shape[0]}x{g.shape[1]}")
        lines.append(f"Player position: row={player_pos[0]}, col={player_pos[1]}")

        # 色の分布
        from collections import Counter
        color_counts = Counter(int(v) for v in g[:60].flatten())
        top_colors = color_counts.most_common(5)
        lines.append(f"Top colors: {top_colors}")

        # 発見されたルール
        if discovered_rules:
            lines.append("Discovered rules:")
            for rule_type, rule_desc in discovered_rules.items():
                # jcrossのコメント行だけ抽出
                for line in rule_desc.split('\n'):
                    if line.strip().startswith('//'):
                        lines.append(f"  {line.strip()}")

        # パターン比較（左右のブロック色）
        if left_pattern and right_pattern:
            lines.append(f"Left pattern: {left_pattern}")
            lines.append(f"Right pattern: {right_pattern}")
            lines.append("Goal: Make right pattern match left pattern")

        return '\n'.join(lines)
