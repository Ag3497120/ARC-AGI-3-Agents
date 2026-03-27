"""
Test the full Cross Engine pipeline against recorded game frames.
"""
import json
import sys
import os
import numpy as np

sys.path.insert(0, '.')
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sensor_mod = load_module('cross_sensor', 'agents/cross_engine/cross_sensor.py')
sim_mod = load_module('simulator', 'agents/cross_engine/simulator.py')
CrossSensor = sensor_mod.CrossSensor
CrossWorld = sim_mod.CrossWorld

# For primitives/rule_mixer, patch the relative imports
import types
# Create a fake package so relative imports work
cross_engine_pkg = types.ModuleType('cross_engine')
cross_engine_pkg.__path__ = ['agents/cross_engine']
sys.modules['cross_engine'] = cross_engine_pkg
sys.modules['cross_engine.simulator'] = sim_mod

prim_mod = load_module('cross_engine.primitives', 'agents/cross_engine/primitives.py')
sys.modules['cross_engine.primitives'] = prim_mod

rm_mod = load_module('cross_engine.rule_mixer', 'agents/cross_engine/rule_mixer.py')

CrossSensor = sensor_mod.CrossSensor
CrossWorld = sim_mod.CrossWorld
RuleMixer = rm_mod.RuleMixer
all_primitives = prim_mod.all_primitives


def load_recording(path):
    frames = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                data = entry.get('data', entry)
                if 'frame' in data:
                    frames.append(data)
    return frames


def test_game(rec_path, game_name):
    frames = load_recording(rec_path)
    if not frames:
        print(f"  No frames in {rec_path}")
        return

    print(f"\n{'='*60}")
    print(f"Game: {game_name}  |  Frames: {len(frames)}")
    print(f"{'='*60}")

    sensor = CrossSensor()
    test_indices = [0, 1, len(frames)//4, len(frames)//2]
    test_indices = sorted(set(i for i in test_indices if i < len(frames)))

    for fi in test_indices:
        frame = frames[fi]
        grid = frame['frame'][0]
        state = frame.get('state', '?')
        levels = frame.get('levels_completed', 0)

        snap = sensor.observe(grid)

        print(f"\n--- Frame {fi} (state={state}, levels={levels}) ---")
        print(f"  Objects: {len(snap.objects)} | Colors: {snap.panorama.get('unique_colors',0)} | Anomalies: {len(snap.anomalies)}")
        print(f"  Rare: {snap.panorama.get('rare_colors',[])} | Dominant: {snap.panorama.get('dominant_colors',[])}")
        print(f"  Shapes: {snap.panorama.get('shape_counts',{})}")
        
        for a in snap.anomalies[:4]:
            print(f"    [{a.get('significance','?')}] {a['type']}: c={a.get('color')} pos={a.get('position')}")

        if snap.diff and snap.diff.has_changes:
            print(f"  Diff: {snap.diff.summary()}")

        # Roles
        roles = {}
        for role in ['mover','trigger','barrier','countdown','container','pattern']:
            found = sensor.find_by_role_hint(snap, role)
            if found:
                roles[role] = len(found)
        if roles:
            print(f"  Roles: {roles}")

        # Rule discovery on first frame only
        if fi == 0:
            print(f"\n  >>> RULE DISCOVERY <<<")
            world = CrossWorld(grid)
            print(f"  Player: {world.player_pos} | Lock: {world.lock_pos}")
            
            pp = world.get_player_pattern()
            lp = world.get_lock_pattern()
            print(f"  Player pattern({len(pp)} cells) | Lock pattern({len(lp)} cells)")

            prims = all_primitives()
            mixer = RuleMixer(world, prims)
            expl = mixer.explain_game()

            primary = expl.get('primary_rule')
            secondary = expl.get('secondary_rules', [])
            path = expl.get('optimal_path', [])
            conf = expl.get('confidence', 0.0)

            rnames = [primary.name] + [r.name for r in secondary]
            print(f"  Rules: {rnames} | Confidence: {conf:.3f}")
            print(f"  Path: {len(path)} actions | First 10: {path[:10]}")
            print(f"  Combos tried: {len(mixer.tried_combinations)} | Successful: {len(mixer.successful_rules)}")


# === Run ===
print("Cross Engine Full Pipeline Test")
print("="*60)

rec_dir = 'recordings'
tested = set()

for fname in sorted(os.listdir(rec_dir)):
    if not fname.endswith('.recording.jsonl'):
        continue
    game_prefix = fname.split('-')[0]
    if game_prefix in tested:
        continue
    tested.add(game_prefix)
    game_id = fname.split('.')[0]
    test_game(os.path.join(rec_dir, fname), game_id)

print(f"\n{'='*60}")
print(f"Tested {len(tested)} game types: {sorted(tested)}")
