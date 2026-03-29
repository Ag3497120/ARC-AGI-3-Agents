#!/usr/bin/env python3
"""SLM-onlyエージェントの5ゲームベースライン"""
import sys, os, time
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

from arc_agi import Arcade
from arcengine import GameAction
from agents.slm_only_agent import SLMOnlyAgent

ROOT_URL = "https://three.arcprize.org"
test_games = ["ls20", "lp85", "m0r0", "ft09", "tr87"]

arcade = Arcade(ROOT_URL, os.environ.get("ARC_API_KEY",""))
scorecard = arcade.create_scorecard()
print(f"Scorecard: {scorecard}")

total_levels = 0
for game_name in test_games:
    agent = SLMOnlyAgent()
    env = arcade.get_environment(game_name, scorecard_id=scorecard)
    frame = env.reset()
    actions = 0
    max_actions = 150  # 低い上限
    
    while actions < max_actions:
        action = agent.choose_action([frame], frame)
        frame = env.step(action)
        actions += 1
        if frame.state == "GAME_OVER":
            break
    
    levels = frame.levels_completed if hasattr(frame, 'levels_completed') else 0
    total_levels += levels
    print(f"{game_name}: levels={levels} actions={actions}")

print(f"\nSLM-only total: {total_levels} levels from {len(test_games)} games")
