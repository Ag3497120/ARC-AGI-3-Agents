"""
Test the v21 autonomous loop against recorded frames.
Simulates the agent's decision-making without the API.
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
CrossSensor = sensor_mod.CrossSensor


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


def simulate_autonomous(rec_path, game_name, max_frames=30):
    """Simulate the autonomous observe→probe→map→simulate→execute loop."""
    frames = load_recording(rec_path)
    if not frames:
        print(f"  No frames")
        return

    print(f"\n{'='*60}")
    print(f"Game: {game_name} | {len(frames)} recorded frames | Testing first {min(max_frames, len(frames))}")
    print(f"{'='*60}")

    sensor = CrossSensor()
    action_model = {}  # action_idx → [effective_bools]
    movement_map = {}  # action_idx → movement vectors
    phase = 'observe'
    available = list(range(4))  # default
    probe_queue = []
    action_queue = []
    total_probes = 0
    total_executes = 0
    last_snap = None

    for fi in range(min(max_frames, len(frames))):
        frame = frames[fi]
        grid = frame['frame'][0]
        state = frame.get('state', '?')
        levels = frame.get('levels_completed', 0)
        avail = frame.get('available_actions', None)

        # Observe
        snap = sensor.observe(grid)

        if phase == 'observe':
            # First frame analysis
            p = snap.panorama
            print(f"\n[OBSERVE] Frame {fi}")
            print(f"  Objects: {p.get('total_objects',0)} | Colors: {p.get('unique_colors',0)}")
            print(f"  Rare: {p.get('rare_colors',[])} | Dominant: {p.get('dominant_colors',[])}")
            print(f"  Shapes: {p.get('shape_counts',{})}")
            print(f"  Anomalies: {len(snap.anomalies)}")
            
            # Roles
            for role in ['mover','trigger','barrier','countdown','container','pattern']:
                found = sensor.find_by_role_hint(snap, role)
                if found:
                    print(f"  {role}: {len(found)} objects")
            
            if avail:
                # Parse available actions
                print(f"  Available actions: {avail}")
            
            # Start probing
            probe_queue = list(range(min(7, len(avail) if avail else 4)))
            phase = 'probe'
            last_snap = snap
            print(f"  → Transition to PROBE ({len(probe_queue)} actions to test)")
            continue

        elif phase == 'probe':
            # Record reaction from previous probe
            if last_snap is not None and snap.diff is not None:
                probe_idx = total_probes - 1
                if probe_idx >= 0:
                    effective = snap.diff.has_changes
                    movers = [obj for obj in snap.objects if obj.descriptor.temporal_axis.get('moved')]
                    
                    if probe_idx not in action_model:
                        action_model[probe_idx] = []
                    action_model[probe_idx].append(effective)
                    
                    if effective and movers:
                        for m in movers:
                            mv = m.descriptor.temporal_axis.get('movement', (0,0))
                            if mv != (0,0):
                                if probe_idx not in movement_map:
                                    movement_map[probe_idx] = []
                                movement_map[probe_idx].append(mv)
                    
                    status = "EFFECTIVE" if effective else "no effect"
                    mv_str = f" movement={movement_map.get(probe_idx, [])}" if probe_idx in movement_map else ""
                    diff_str = snap.diff.summary() if snap.diff.has_changes else "nothing"
                    print(f"  [PROBE result] action {probe_idx}: {status} | {diff_str}{mv_str}")

            if probe_queue:
                next_probe = probe_queue.pop(0)
                total_probes += 1
                last_snap = snap
                print(f"  [PROBE] Testing action {next_probe}")
                continue
            else:
                # Done probing
                phase = 'simulate'
                effective_actions = [k for k, v in action_model.items() if any(v)]
                useless_actions = [k for k, v in action_model.items() if not any(v)]
                print(f"\n[MAP] Probing complete:")
                print(f"  Effective: {effective_actions}")
                print(f"  Useless: {useless_actions}")
                print(f"  Movement map: { {k: v[0] if v else (0,0) for k,v in movement_map.items()} }")

        if phase == 'simulate':
            # Build a plan
            effective = [k for k, v in action_model.items() if any(v)]
            
            # Find targets: rare, small objects
            targets = []
            for obj in snap.objects:
                c = obj.descriptor.color_axis
                p = obj.descriptor.position_axis
                s = obj.descriptor.scale_axis
                if (c.get('is_rare', False)
                    and not p.get('in_timer_area', True)
                    and s.get('size_category') in ('point', 'tiny', 'small')):
                    targets.append(obj.descriptor.position_axis.get('centroid_int'))

            # Find controllable (mover with size >= 5)
            controllable_pos = None
            if snap.diff and snap.diff.moved:
                for obj in snap.diff.moved:
                    if obj.descriptor.scale_axis.get('cell_count', 0) >= 5:
                        controllable_pos = obj.descriptor.position_axis.get('centroid_int')
                        break

            print(f"\n[SIMULATE] Frame {fi}")
            print(f"  Targets: {targets[:5]}")
            print(f"  Controllable: {controllable_pos}")
            print(f"  Movement actions: {movement_map}")
            
            if effective:
                # Simple plan: cycle effective actions
                action_queue = (effective * 15)[:50]
                print(f"  Plan: {len(action_queue)} actions (cycling effective)")
            else:
                action_queue = (available * 10)[:40]
                print(f"  Plan: {len(action_queue)} actions (cycling all)")
            
            phase = 'execute'

        if phase == 'execute':
            if action_queue:
                act = action_queue.pop(0)
                total_executes += 1
                last_snap = snap
                
                # Periodic re-observe
                if total_executes % 10 == 0:
                    print(f"  [EXECUTE] Frame {fi} action={act} queue={len(action_queue)} "
                          f"objs={len(snap.objects)} anom={len(snap.anomalies)}")
                    if snap.diff and snap.diff.has_changes:
                        print(f"    diff: {snap.diff.summary()}")
            else:
                phase = 'simulate'  # re-plan

    print(f"\n[DONE] Probes: {total_probes} | Executes: {total_executes}")
    print(f"  Final model: effective={[k for k,v in action_model.items() if any(v)]} "
          f"movement={movement_map}")


# Run on all game types
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
    simulate_autonomous(os.path.join(rec_dir, fname), game_id, max_frames=25)

print(f"\n{'='*60}")
print(f"Tested {len(tested)} games: {sorted(tested)}")
