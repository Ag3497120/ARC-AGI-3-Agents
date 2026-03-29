"""
slm_bridge.py — SLMをjcrossコード生成器として使う

kofdai型:
- SLMは問題を解かない。jcrossを書く。
- テンプレート（Cross構造のスケルトン）を提供し、SLMが穴埋め
- 生成されたjcrossはsoul.jcrossに書き込まれてCross構造が成長
"""

import json
import urllib.request
from typing import Optional, Dict, List, Any


class SLMBridge:
    """SLMをjcrossコード生成器として使う"""

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "qwen2.5:1.5b"

    def __init__(self):
        self._available = self._check()
        self._cache = {}

    def _check(self):
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    @property
    def is_available(self):
        return self._available

    def _call(self, prompt, max_tokens=150):
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

    def generate_jcross_rule(self, axiom_type, context):
        """axiomの種類に応じてjcrossルールを生成する"""

        if axiom_type == "click_color_cycle":
            return self._gen_color_cycle_rule(context)
        elif axiom_type == "wall_toggle":
            return self._gen_wall_toggle_rule(context)
        elif axiom_type == "block_state_change":
            return self._gen_block_state_rule(context)
        elif axiom_type == "move_asymmetric":
            return self._gen_asymmetric_move_rule(context)

        return self._gen_generic_rule(axiom_type, context)

    def _gen_color_cycle_rule(self, ctx):
        """色サイクルルールのjcross生成"""
        cycle = ctx.get("cycle_colors", [])
        blocks = ctx.get("block_positions", [])
        left_colors = ctx.get("left_colors", [])
        right_colors = ctx.get("right_colors", [])

        if not cycle:
            return None

        cycle_len = len(cycle)

        # 計算部分はPythonで確実に作る
        click_plan_lines = []
        for i, (lc, rc) in enumerate(zip(left_colors, right_colors)):
            if lc != rc:
                if lc in cycle and rc in cycle:
                    target_idx = cycle.index(lc)
                    current_idx = cycle.index(rc)
                    clicks = (target_idx - current_idx + cycle_len) % cycle_len
                    if clicks > 0 and i < len(blocks):
                        click_plan_lines.append(
                            f"// ブロック{i}: 現在色={rc} → 目標色={lc} → {clicks}回クリック"
                        )
                        click_plan_lines.append(
                            f"クリック計画 に 追加({{\"位置\": {list(blocks[i])}, \"回数\": {clicks}}})"
                        )

        if not click_plan_lines:
            return None

        # SLMに拡張コードを生成させる
        prompt = f"""You are writing jcross code (Japanese syntax).
Color cycle: {cycle} (length={cycle_len})
Write a jcross function that executes a click plan.

Use ONLY this syntax:
- 関数 name(params) {{ }}
- もし condition {{ }}
- 返す value
- 変数 = 値

Write a function 色サイクル実行() that returns the next click position from クリック計画.
Keep it under 10 lines. No English."""

        slm_code = self._call(prompt, max_tokens=200)

        # SLMの出力をサニタイズ
        func_code = self._sanitize_jcross(slm_code) if slm_code else ""

        # テンプレート + 計算結果 + SLM拡張を合成
        rule = f"""// 発見: 色サイクル $= クリックで色が巡回
// サイクル: {cycle} (長さ={cycle_len})
クリック計画 = []
{chr(10).join(click_plan_lines)}

{func_code if func_code else "// SLM拡張なし"}"""

        return rule

    def _gen_wall_toggle_rule(self, ctx):
        """壁開閉ルールのjcross生成"""
        trigger_pos = ctx.get("trigger_position", [])
        effect_region = ctx.get("effect_region", [])

        prompt = f"""Write a jcross function 壁開閉判定() in Japanese syntax.
It should return the direction (0=up,1=down,2=left,3=right) toward a trigger at position {trigger_pos}.
Player position is 自分の位置 (a list [row,col]).
Use もし/返す only. Under 8 lines. No English."""

        slm_code = self._call(prompt, max_tokens=150)
        func_code = self._sanitize_jcross(slm_code) if slm_code else ""

        return f"""// 発見: 壁開閉 $= 特定位置への接触
// トリガー位置: {trigger_pos}
// 効果領域: {effect_region}
{func_code if func_code else "// SLM拡張なし"}"""

    def _gen_block_state_rule(self, ctx):
        """ブロック状態変化ルールのjcross生成"""
        return f"""// 発見: ブロック状態変化
// 行動によりブロックの色/状態が変化する
// 変化パターン: {ctx.get('transitions', {})}"""

    def _gen_asymmetric_move_rule(self, ctx):
        """非対称移動ルールのjcross生成"""
        moves = ctx.get("move_vectors", {})
        return f"""// 発見: 非対称移動
// 移動ベクトル: {moves}
// 上下と左右で移動量が異なる"""

    def _gen_generic_rule(self, axiom_type, ctx):
        """汎用ルール生成 — SLMに自由に書かせる"""
        prompt = f"""Write a short jcross comment describing this game rule:
Type: {axiom_type}
Context: {json.dumps(ctx, default=str)[:200]}
Use Japanese. Format: // comment lines only. Under 5 lines."""

        slm_code = self._call(prompt, max_tokens=100)
        if slm_code:
            return self._sanitize_jcross(slm_code)
        return f"// 発見: {axiom_type}"

    def _sanitize_jcross(self, code):
        """SLM出力からjcross互換コードだけ抽出"""
        if not code:
            return ""

        lines = []
        in_code = False
        for line in code.split('\n'):
            stripped = line.strip()
            # コードブロック記号を除去
            if stripped.startswith('```'):
                in_code = not in_code
                continue
            # 空行やコメントは通す
            if not stripped:
                continue
            # 日本語キーワードまたはコメントを含む行だけ採用
            if any(kw in stripped for kw in ['関数', 'もし', '返す', '各', '表示', '//', '=', '{', '}']):
                lines.append(stripped)
            elif stripped.startswith('//'):
                lines.append(stripped)

        return '\n'.join(lines)

    def generate_click_plan_from_patterns(self, grid, cycle_colors,
                                          left_blocks, right_blocks):
        """
        左右のブロックパターンを比較してクリック計画を生成

        SLMを使わず純粋な計算。SLMはjcrossルールの拡張のみ。

        Args:
            grid: 64x64 grid
            cycle_colors: [3, 5, 7] 色サイクル
            left_blocks: [(r,c,color), ...] 左側ブロック
            right_blocks: [(r,c,color), ...] 右側ブロック

        Returns:
            [{"pos": (r,c), "clicks": n}, ...]
        """
        if not cycle_colors:
            return []

        cycle_len = len(cycle_colors)
        plan = []

        for (lr, lc, l_color), (rr, rc, r_color) in zip(left_blocks, right_blocks):
            if l_color != r_color:
                if l_color in cycle_colors and r_color in cycle_colors:
                    target_idx = cycle_colors.index(l_color)
                    current_idx = cycle_colors.index(r_color)
                    clicks = (target_idx - current_idx + cycle_len) % cycle_len
                    if clicks > 0:
                        plan.append({"pos": (rr, rc), "clicks": clicks})

        return plan
