#!/usr/bin/env python3
"""
run_all_sequential.py — Run CrossResonanceAgent on all games SEQUENTIALLY.

Avoids 429 rate limits by:
  1. Running one game at a time with delays between games
  2. Rate-limiting API calls (min 0.5s between step() calls)
  3. Exponential backoff retry on 429 / None responses

Usage:
    .venv/bin/python run_all_sequential.py
    .venv/bin/python run_all_sequential.py --games ls01,ls02,ls03
    .venv/bin/python run_all_sequential.py --delay 5
"""

import json
import logging
import os
import sys
import time
import threading
import traceback
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# Load env before anything else
load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

from arc_agi import Arcade
from arcengine import GameAction
from agents.cross_resonance_v26 import CrossResonanceV26 as CrossResonanceAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_URL = "https://three.arcprize.org"
AGENT_NAME = "crossresonanceagent"
BETWEEN_GAME_DELAY = 3.0        # seconds between games
MIN_API_INTERVAL = 0.5           # minimum seconds between API calls
MAX_RETRIES = 6                  # max retries on 429
INITIAL_BACKOFF = 2.0            # initial backoff seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate Limiter — enforces minimum interval between API calls
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                time.sleep(sleep_time)
            self._last_call = time.time()

_rate_limiter = RateLimiter(MIN_API_INTERVAL)

# ---------------------------------------------------------------------------
# Monkey-patch: wrap do_action_request with retry + rate limiting
# ---------------------------------------------------------------------------
_original_do_action_request = CrossResonanceAgent.do_action_request

def _retrying_do_action_request(self, action: GameAction):
    """Wraps do_action_request with rate limiting and 429 retry logic."""
    for attempt in range(MAX_RETRIES + 1):
        _rate_limiter.wait()
        try:
            result = _original_do_action_request(self, action)
            return result
        except ValueError as e:
            # "Received None frame data from environment" → likely a 429
            if "None" in str(e) and attempt < MAX_RETRIES:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"[{self.game_id}] Got None response (likely 429), "
                    f"retry {attempt + 1}/{MAX_RETRIES} in {backoff:.1f}s"
                )
                time.sleep(backoff)
                continue
            raise
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate" in err_str) and attempt < MAX_RETRIES:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"[{self.game_id}] Rate limited (429), "
                    f"retry {attempt + 1}/{MAX_RETRIES} in {backoff:.1f}s"
                )
                time.sleep(backoff)
                continue
            raise
    # All retries exhausted — raise
    raise RuntimeError(f"[{self.game_id}] Exhausted {MAX_RETRIES} retries on API call")

CrossResonanceAgent.do_action_request = _retrying_do_action_request

# ---------------------------------------------------------------------------
# Fetch game list from API
# ---------------------------------------------------------------------------
def fetch_game_list() -> list[str]:
    headers = {
        "X-API-Key": os.getenv("ARC_API_KEY", ""),
        "Accept": "application/json",
    }
    try:
        r = requests.get(f"{ROOT_URL}/api/games", headers=headers, timeout=15)
        if r.status_code == 200:
            games = [g["game_id"] for g in r.json()]
            return games
        else:
            logger.error(f"Failed to fetch games: HTTP {r.status_code}")
            return []
    except Exception as e:
        logger.error(f"Failed to fetch games: {e}")
        return []

# ---------------------------------------------------------------------------
# Run a single game
# ---------------------------------------------------------------------------
def run_single_game(game_id: str, arcade: Arcade, card_id: str, tags: list[str]) -> dict:
    """Run CrossResonanceAgent on one game, return result dict."""
    result = {
        "game_id": game_id,
        "levels_completed": 0,
        "actions_taken": 0,
        "status": "error",
        "errors": [],
        "elapsed_seconds": 0.0,
    }
    start = time.time()
    try:
        env = arcade.make(game_id, scorecard_id=card_id)
        agent = CrossResonanceAgent(
            card_id=card_id,
            game_id=game_id,
            agent_name=AGENT_NAME,
            ROOT_URL=ROOT_URL,
            record=True,
            arc_env=env,
            tags=tags,
        )
        agent.main()

        result["levels_completed"] = agent.levels_completed
        result["actions_taken"] = agent.action_counter
        result["status"] = "completed"

    except KeyboardInterrupt:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        result["errors"].append(str(e))
        logger.error(f"[{game_id}] Error: {e}\n{tb}")
        # Still capture partial progress
        try:
            result["levels_completed"] = agent.levels_completed
            result["actions_taken"] = agent.action_counter
        except Exception:
            pass
        result["status"] = "error"

    result["elapsed_seconds"] = round(time.time() - start, 2)
    return result

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
def print_summary(results: list[dict]):
    print("\n" + "=" * 80)
    print("SEQUENTIAL RUN SUMMARY")
    print("=" * 80)
    print(f"{'Game':<20} {'Status':<12} {'Levels':<8} {'Actions':<10} {'Time(s)':<10} {'Errors'}")
    print("-" * 80)

    total_levels = 0
    total_actions = 0
    total_errors = 0

    for r in results:
        errors_str = "; ".join(r["errors"][:1]) if r["errors"] else ""
        if len(errors_str) > 30:
            errors_str = errors_str[:27] + "..."
        print(
            f"{r['game_id']:<20} {r['status']:<12} {r['levels_completed']:<8} "
            f"{r['actions_taken']:<10} {r['elapsed_seconds']:<10.1f} {errors_str}"
        )
        total_levels += r["levels_completed"]
        total_actions += r["actions_taken"]
        if r["errors"]:
            total_errors += 1

    print("-" * 80)
    print(
        f"{'TOTAL':<20} {'':12} {total_levels:<8} {total_actions:<10} "
        f"{'':10} {total_errors} games with errors"
    )
    print(f"Games: {len(results)} | Completed: {sum(1 for r in results if r['status'] == 'completed')}")
    print("=" * 80)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run CrossResonanceAgent sequentially on all games")
    parser.add_argument("--games", type=str, default=None,
                        help="Comma-separated game ID prefixes to filter (e.g. ls01,ls02)")
    parser.add_argument("--delay", type=float, default=BETWEEN_GAME_DELAY,
                        help=f"Delay in seconds between games (default: {BETWEEN_GAME_DELAY})")
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated tags for the scorecard")
    args = parser.parse_args()

    logger.info("Fetching game list from API...")
    all_games = fetch_game_list()
    if not all_games:
        logger.error("No games found. Check your ARC_API_KEY and network.")
        sys.exit(1)

    # Filter games if specified
    games = all_games
    if args.games:
        prefixes = [p.strip() for p in args.games.split(",")]
        games = [g for g in all_games if any(g.startswith(p) for p in prefixes)]
        if not games:
            logger.error(f"No games match filter: {args.games}")
            logger.info(f"Available games: {all_games}")
            sys.exit(1)

    logger.info(f"Will run {len(games)} games sequentially: {games}")

    # Tags
    tags = ["agent", AGENT_NAME, "sequential"]
    if args.tags:
        tags.extend(t.strip() for t in args.tags.split(","))

    # Create Arcade and open scorecard
    arcade = Arcade()
    card_id = arcade.open_scorecard(tags=tags)
    logger.info(f"Opened scorecard: {card_id}")

    results = []
    try:
        for i, game_id in enumerate(games):
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i+1}/{len(games)}] Starting game: {game_id}")
            logger.info(f"{'='*60}")

            result = run_single_game(game_id, arcade, card_id, tags)
            results.append(result)

            logger.info(
                f"[{i+1}/{len(games)}] {game_id}: "
                f"status={result['status']}, levels={result['levels_completed']}, "
                f"actions={result['actions_taken']}, time={result['elapsed_seconds']}s"
            )

            # Delay between games (skip after last)
            if i < len(games) - 1:
                logger.info(f"Waiting {args.delay}s before next game...")
                time.sleep(args.delay)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Saving partial results...")
    finally:
        # Close scorecard
        try:
            scorecard = arcade.close_scorecard(card_id)
            if scorecard:
                logger.info("Scorecard closed successfully.")
                logger.info(json.dumps(scorecard.model_dump(), indent=2))
            scorecard_url = f"{ROOT_URL}/scorecards/{card_id}"
            logger.info(f"View scorecard: {scorecard_url}")
        except Exception as e:
            logger.error(f"Failed to close scorecard: {e}")

        # Print summary
        print_summary(results)

        # Save results
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"results/full_run_{timestamp}.json"
        output = {
            "timestamp": timestamp,
            "scorecard_id": card_id,
            "total_games": len(games),
            "games_run": len(results),
            "total_levels_completed": sum(r["levels_completed"] for r in results),
            "total_actions": sum(r["actions_taken"] for r in results),
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
