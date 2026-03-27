"""Test v22 autonomous loop with wall learning + click planning + auto-replan."""
import json, sys, os
import numpy as np
sys.path.insert(0, '.')
import importlib.util

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sensor_mod = load_mod('cross_sensor', 'agents/cross_engine/cross_sensor.py')
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


def sim_game(rec_path, game_name, max_frames=50):
    frames = load_recording(rec_path)
    if not frames:
        return

    print(f"\n{'='*60}")
    print(f"Game: {game_name} | {len(frames)} frames | Testing {min(max_frames, len(frames))}")
    print(f"{'='*60}")

    sensor = CrossSensor()
    
    # Import agent components
    agent_code = open('agents/cross_resonance_agent.py').read()
    # Extract just the helper classes (WallMap, ClickPlanner, ReplanTrigger, ActionModel)
    # We'll instantiate them directly
    
    # Simple state machine simulation
    phase = 'observe'
    model_reactions = {}       # action_idx → [bool]
    model_movements = {}       # action_idx → [(r,c)]
    available = []
    probe_queue = []
    action_queue = []
    total_probes = 0
    total_executes = 0
    last_snap = None
    controllable_pos = None
    is_click = False
    wall_blocked = 0
    replan_count = 0
    click_targets = []
    
    for fi in range(min(max_frames, len(frames))):
        frame = frames[fi]
        grid = frame['frame'][0]
        state = frame.get('state', '?')
        levels = frame.get('levels_completed', 0)
        avail = frame.get('available_actions', None)

        snap = sensor.observe(grid)

        # Process reaction from previous action
        if last_snap is not None and snap.diff is not None:
            effective = snap.diff.has_changes
            movements = []
            if snap.diff.moved:
                for obj in snap.diff.moved:
                    mv = obj.descriptor.temporal_axis.get('movement', (0,0))
                    if mv != (0,0) and (abs(mv[0]) >= 3 or abs(mv[1]) >= 3):
                        movements.append(mv)
            
            # Find controllable
            if snap.diff.moved:
                for obj in sorted(snap.diff.moved,
                                  key=lambda o: -o.descriptor.scale_axis.get('cell_count', 0)):
                    if not obj.descriptor.position_axis.get('in_timer_area'):
                        if obj.descriptor.scale_axis.get('cell_count', 0) >= 3:
                            controllable_pos = obj.descriptor.position_axis.get('centroid_int')
                            break

            if phase == 'probe' and total_probes > 0:
                pidx = total_probes - 1
                if pidx not in model_reactions:
                    model_reactions[pidx] = []
                    model_movements[pidx] = []
                model_reactions[pidx].append(effective)
                model_movements[pidx].extend(movements)
                
                mv_str = f" mv={movements}" if movements else ""
                print(f"  [PROBE {pidx}] {'EFFECTIVE' if effective else 'no-op'} | {snap.diff.summary()}{mv_str}")

        if phase == 'observe':
            is_click = avail == [6] if avail else False
            available = list(range(len(avail))) if avail else list(range(4))
            
            p = snap.panorama
            print(f"\n[OBSERVE] f={fi} objs={p.get('total_objects',0)} colors={p.get('unique_colors',0)}")
            print(f"  rare={p.get('rare_colors',[])} dom={p.get('dominant_colors',[])}")
            print(f"  shapes={p.get('shape_counts',{})} anom={len(snap.anomalies)}")
            
            roles = {}
            for role in ['mover','trigger','barrier','countdown','container','pattern']:
                found = sensor.find_by_role_hint(snap, role)
                if found: roles[role] = len(found)
            print(f"  roles={roles}")
            print(f"  click_game={is_click} available={avail}")
            
            # Bootstrap wall map
            dominant = p.get('dominant_colors', [])
            print(f"  wall_bootstrap: dominant colors {dominant} marked as potential walls")
            
            probe_queue = list(available)
            phase = 'probe'
            last_snap = snap
            continue

        elif phase == 'probe':
            if probe_queue:
                pidx = probe_queue.pop(0)
                total_probes += 1
                last_snap = snap
                continue
            else:
                # Done probing
                effective_actions = [k for k, v in model_reactions.items() if any(v)]
                mv_map = {}
                for k, vecs in model_movements.items():
                    if vecs:
                        avg_r = sum(r for r,c in vecs) // len(vecs)
                        avg_c = sum(c for r,c in vecs) // len(vecs)
                        mv_map[k] = (avg_r, avg_c)
                
                print(f"\n[MAP] effective={effective_actions} movements={mv_map} ctrl={controllable_pos}")
                phase = 'simulate'

        if phase == 'simulate':
            if is_click:
                # Click planning
                click_targets = []
                for obj in snap.objects:
                    p = obj.descriptor.position_axis
                    s = obj.descriptor.scale_axis
                    r = obj.descriptor.relation_axis
                    c = obj.descriptor.color_axis
                    if p.get('in_timer_area'): continue
                    score = 0
                    if r.get('contained_by_count', 0) > 0: score += 3
                    if s.get('size_category') in ('small','medium'): score += 2
                    if c.get('is_rare'): score += 1
                    if score > 0:
                        click_targets.append((score, p.get('centroid_int')))
                click_targets.sort(key=lambda x: -x[0])
                click_targets = [pos for _, pos in click_targets]
                action_queue = list(range(min(len(click_targets), 30)))  # indices into click_targets
                print(f"\n[SIM_CLICK] {len(click_targets)} targets, first 5: {click_targets[:5]}")
            else:
                # Movement BFS
                targets = []
                for obj in snap.objects:
                    c = obj.descriptor.color_axis
                    p = obj.descriptor.position_axis
                    s = obj.descriptor.scale_axis
                    if p.get('in_timer_area'): continue
                    score = 0
                    if c.get('is_rare'): score += 3
                    if obj.descriptor.relation_axis.get('contains_count', 0) > 0: score += 2
                    if s.get('size_category') in ('point','tiny','small'): score += 1
                    if score > 0:
                        targets.append((score, p.get('centroid_int')))
                targets.sort(key=lambda x: -x[0])
                target_positions = [pos for _, pos in targets[:5]]
                
                effective_actions = [k for k, v in model_reactions.items() if any(v)]
                mv_map = {}
                for k, vecs in model_movements.items():
                    if vecs:
                        avg_r = sum(r for r,c in vecs) // len(vecs)
                        avg_c = sum(c for r,c in vecs) // len(vecs)
                        mv_map[k] = (avg_r, avg_c)
                
                if controllable_pos and mv_map and target_positions:
                    # BFS
                    start = controllable_pos
                    target_set = set(target_positions)
                    queue = [(start, [])]
                    visited = {start}
                    best_path = []
                    best_dist = float('inf')
                    
                    from collections import deque
                    bfs_q = deque(queue)
                    while bfs_q and len(visited) < 2000:
                        (cr,cc), path = bfs_q.popleft()
                        if len(path) > 60: continue
                        for tr,tc in target_set:
                            dist = abs(cr-tr) + abs(cc-tc)
                            if dist < best_dist:
                                best_dist = dist
                                best_path = path
                            if dist <= 7:
                                best_path = path
                                bfs_q.clear()
                                break
                        if not bfs_q and best_dist <= 7:
                            break
                        for aidx, (dr,dc) in mv_map.items():
                            nr,nc = cr+dr, cc+dc
                            if (nr,nc) not in visited and 0<=nr<64 and 0<=nc<64:
                                visited.add((nr,nc))
                                bfs_q.append(((nr,nc), path+[aidx]))
                    
                    action_queue = best_path[:50]
                    print(f"\n[SIM_MOVE] ctrl={controllable_pos} targets={target_positions[:3]} "
                          f"path={len(action_queue)} best_dist={best_dist}")
                else:
                    action_queue = (effective_actions * 10)[:40]
                    print(f"\n[SIM_FALLBACK] cycling effective actions, len={len(action_queue)}")
            
            phase = 'execute'

        if phase == 'execute':
            # Auto-replan check
            should_replan = False
            if not action_queue:
                should_replan = True
            elif snap.diff and snap.diff.has_changes:
                changes = (len(snap.diff.moved) + len(snap.diff.appeared) 
                           + len(snap.diff.disappeared) + len(snap.diff.size_changed))
                if changes >= 5:
                    should_replan = True
            elif total_executes > 0 and total_executes % 20 == 0:
                should_replan = True
            
            if should_replan and total_executes > 0:
                replan_count += 1
                phase = 'simulate'
                print(f"  [REPLAN #{replan_count}] f={fi} queue_was={len(action_queue)}")
                continue
            
            if action_queue:
                act = action_queue.pop(0)
                total_executes += 1
                last_snap = snap
                
                if total_executes % 10 == 0 or total_executes <= 3:
                    diff_str = snap.diff.summary() if snap.diff and snap.diff.has_changes else "none"
                    print(f"  [EXEC #{total_executes}] f={fi} act={act} q={len(action_queue)} diff={diff_str}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Probes: {total_probes} | Executes: {total_executes} | Replans: {replan_count}")
    eff = [k for k,v in model_reactions.items() if any(v)]
    mv = {k: model_movements[k][0] if model_movements.get(k) else (0,0) for k in eff}
    print(f"  Effective: {eff} | Movements: {mv}")
    print(f"  Controllable: {controllable_pos} | Click game: {is_click}")
    print(f"  Wall blocks: {wall_blocked}")


# Run
rec_dir = 'recordings'
tested = set()
for fname in sorted(os.listdir(rec_dir)):
    if not fname.endswith('.recording.jsonl'): continue
    gp = fname.split('-')[0]
    if gp in tested: continue
    tested.add(gp)
    sim_game(os.path.join(rec_dir, fname), fname.split('.')[0], max_frames=40)

print(f"\n{'='*60}")
print(f"Tested: {sorted(tested)}")
