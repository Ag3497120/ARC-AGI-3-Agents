"""
slm_only_agent.py — SLM(2B)のみでARC-AGI-3を解くベースライン
jcrossなし、Crossなし、BFSなし。純粋にSLMの推論力のみ。
"""
import sys
import json
import urllib.request
import numpy as np
from collections import Counter
from arcengine import FrameData, GameAction, GameState
from .agent import Agent

ALL_ACTIONS = [
    GameAction.ACTION1, GameAction.ACTION2,
    GameAction.ACTION3, GameAction.ACTION4,
    GameAction.ACTION5, GameAction.ACTION6,
    GameAction.ACTION7,
]

class SLMOnlyAgent(Agent):
    agent_name = "slmonlyagent"
    
    def __init__(self):
        self._frame = 0
        self._prev_grid = None
        self._action_history = []
        self._slm_cache = {}
    
    def _call_slm(self, prompt, max_tokens=30):
        cache_key = prompt[:300]
        if cache_key in self._slm_cache:
            return self._slm_cache[cache_key]
        try:
            data = json.dumps({
                "model": "qwen2.5:1.5b",
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1}
            }).encode('utf-8')
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                answer = result.get("response", "").strip()
                self._slm_cache[cache_key] = answer
                return answer
        except Exception:
            return ""
    
    def _grid_summary(self, grid):
        g = np.array(grid)
        rows, cols = g.shape
        colors = Counter(int(v) for v in g[:min(60,rows)].flatten())
        top = colors.most_common(5)
        
        # 変化した部分
        diff_info = ""
        if self._prev_grid is not None:
            prev = np.array(self._prev_grid)
            if prev.shape == g.shape:
                diff = np.sum(prev[:min(60,rows)] != g[:min(60,rows)])
                if diff > 0:
                    diff_info = f"\nChanged cells: {diff}"
        
        return f"Grid {rows}x{cols}. Colors: {top}.{diff_info}\nRecent actions: {self._action_history[-5:]}"
    
    def choose_action(self, frames, latest_frame):
        grid = latest_frame.frame[0]
        self._frame += 1
        
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._prev_grid = None
            self._action_history = []
            self._slm_cache = {}
            return GameAction.RESET
        
        available = latest_frame.available_actions or ALL_ACTIONS[:4]
        n_actions = len(available)
        
        summary = self._grid_summary(grid)
        
        # 10フレームに1回だけSLMに聞く（コスト削減）
        if self._frame % 10 == 1 or self._frame <= 5:
            prompt = f"""You are playing a puzzle game on a 64x64 grid.
{summary}
Available actions: {n_actions} (numbered 1-{n_actions})
Action 6 = click at coordinates.

What action number (1-{n_actions}) should I take next? Answer with just a number."""
            
            response = self._call_slm(prompt)
            try:
                action_num = int(response.strip().split()[0]) - 1
                if 0 <= action_num < n_actions:
                    self._last_slm_choice = action_num
                else:
                    self._last_slm_choice = self._frame % n_actions
            except (ValueError, IndexError):
                self._last_slm_choice = self._frame % n_actions
        
        choice = getattr(self, '_last_slm_choice', self._frame % n_actions)
        self._action_history.append(choice)
        self._prev_grid = [row[:] for row in grid]
        
        a = available[min(choice, len(available)-1)]
        return a

