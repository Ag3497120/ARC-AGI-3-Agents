"""Test v23: passable map + isolated ctrl movement + BFS fix."""
import json, sys, os
import numpy as np
from collections import deque, Counter
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
                if 'frame' in data: frames.append(data)
    return frames


def sim_game(rec_path, game_name, max_frames=60):
    frames = load_recording(rec_path)
    if not frames: return

    print(f"\n{'='*60}")
    print(f"Game: {game_name} | {len(frames)} frames | Test {min(max_frames, len(frames))}")
    print(f"{'='*60}")

    sensor = CrossSensor()
    phase = 'observe'
    # Action model
    action_records = {}   # aidx → [(effective, ctrl_mv)]
    available = []
    probe_queue = []
    action_queue = []
    click_targets = []
    is_click = False
    # Controllable
    ctrl_pos = None
    ctrl_offsets = []
    prev_ctrl_pos = None
    # Passable map
    passable_colors = set()
    blocked_colors = set()
    passable_evidence = np.zeros((64, 64), dtype=float)
    # Stats
    total_probes = 0
    total_executes = 0
    replan_count = 0
    last_snap = None
    last_action_idx = 0
    grid_ref = None

    def get_ctrl_movement(snap_):
        if snap_.diff and snap_.diff.moved:
            best = None; best_size = 0
            for obj in snap_.diff.moved:
                if obj.descriptor.position_axis.get('in_timer_area'): continue
                sc = obj.descriptor.scale_axis.get('cell_count', 0)
                if sc >= 3 and sc > best_size:
                    mv = obj.descriptor.temporal_axis.get('movement', (0,0))
                    if abs(mv[0]) >= 3 or abs(mv[1]) >= 3:
                        best = mv; best_size = sc
            if best: return best
        return (0,0)

    def find_ctrl(snap_):
        nonlocal ctrl_pos, ctrl_offsets, prev_ctrl_pos
        if snap_.diff and snap_.diff.moved:
            for obj in sorted(snap_.diff.moved,
                              key=lambda o: -o.descriptor.scale_axis.get('cell_count', 0)):
                if obj.descriptor.position_axis.get('in_timer_area'): continue
                if obj.descriptor.scale_axis.get('cell_count', 0) >= 3:
                    pos = obj.descriptor.position_axis.get('centroid_int')
                    cells = list(obj.cells)
                    prev_ctrl_pos = ctrl_pos
                    ctrl_pos = pos
                    ctrl_offsets = [(r-pos[0], c-pos[1]) for r,c in cells]
                    return

    def get_movement(aidx):
        recs = action_records.get(aidx, [])
        moves = [mv for eff, mv in recs if eff and mv != (0,0)]
        if not moves: return (0,0)
        cnt = Counter(moves)
        return cnt.most_common(1)[0][0]

    def get_effective():
        return [a for a in available
                if a in action_records and any(eff for eff, _ in action_records[a])]

    def get_mv_actions():
        result = {}
        for a in get_effective():
            mv = get_movement(a)
            if mv != (0,0): result[a] = mv
        return result

    def can_pass(r, c):
        for dr, dc in ctrl_offsets:
            nr, nc = r+dr, c+dc
            if nr<0 or nr>=64 or nc<0 or nc>=64: return False
            ev = passable_evidence[nr, nc]
            if ev <= -1.0: return False
            if ev >= 0.5: continue
            if grid_ref is not None:
                color = int(grid_ref[nr, nc])
                if color in passable_colors: continue
                if color in blocked_colors: return False
        return True

    for fi in range(min(max_frames, len(frames))):
        frame = frames[fi]
        grid = frame['frame'][0]
        state = frame.get('state', '?')
        levels = frame.get('levels_completed', 0)
        avail = frame.get('available_actions', None)
        g = np.array(grid)

        snap = sensor.observe(grid)

        # Process reaction
        if last_snap is not None and snap.diff is not None and total_probes + total_executes > 0:
            effective = snap.diff.has_changes
            ctrl_mv = get_ctrl_movement(snap)
            find_ctrl(snap)

            if last_action_idx not in action_records:
                action_records[last_action_idx] = []
            action_records[last_action_idx].append((effective, ctrl_mv))

            # Update passable map
            if ctrl_pos and prev_ctrl_pos and ctrl_mv != (0,0) and grid_ref is not None:
                for dr, dc in ctrl_offsets:
                    r, c = ctrl_pos[0]+dr, ctrl_pos[1]+dc
                    if 0<=r<64 and 0<=c<64:
                        passable_evidence[r,c] = min(passable_evidence[r,c]+1.0, 3.0)
                        passable_colors.add(int(grid_ref[r,c]))
                for dr, dc in ctrl_offsets:
                    r, c = prev_ctrl_pos[0]+dr, prev_ctrl_pos[1]+dc
                    if 0<=r<64 and 0<=c<64:
                        passable_evidence[r,c] = min(passable_evidence[r,c]+0.5, 3.0)
                        passable_colors.add(int(grid_ref[r,c]))

            if phase == 'probe':
                pidx = (total_probes - 1) if total_probes > 0 else last_action_idx
                mv_str = f" ctrl_mv={ctrl_mv}" if ctrl_mv != (0,0) else " ctrl_mv=BLOCKED"
                print(f"  [PROBE {pidx}] {'EFF' if effective else 'nop'} {snap.diff.summary()}{mv_str}")

        if phase == 'observe':
            is_click = avail == [6] if avail else False
            available = list(range(len(avail))) if avail else list(range(4))
            p = snap.panorama
            print(f"\n[OBSERVE] f={fi} objs={p.get('total_objects',0)} colors={p.get('unique_colors',0)}")
            print(f"  rare={p.get('rare_colors',[])} dom={p.get('dominant_colors',[])}")
            print(f"  click={is_click} avail={avail}")
            # Bootstrap passable map
            passable_evidence[60:, :] = -2.0  # timer area = wall
            grid_ref = g.copy()
            
            probe_queue = list(available)
            phase = 'probe'
            last_snap = snap
            continue

        if phase == 'probe':
            if probe_queue:
                idx = probe_queue.pop(0)
                last_action_idx = idx
                total_probes += 1
                last_snap = snap
                grid_ref = g.copy()
                continue
            else:
                eff = get_effective()
                mvs = get_mv_actions()
                print(f"\n[MAP] effective={eff} movements={mvs}")
                print(f"  passable_colors={passable_colors} blocked_colors={blocked_colors}")
                print(f"  ctrl={ctrl_pos}")
                phase = 'simulate'

        if phase == 'simulate':
            grid_ref = g.copy()
            if is_click:
                # Click targets
                candidates = []
                for obj in snap.objects:
                    p_ = obj.descriptor.position_axis
                    s_ = obj.descriptor.scale_axis
                    rel_ = obj.descriptor.relation_axis
                    c_ = obj.descriptor.color_axis
                    if p_.get('in_timer_area'): continue
                    score = 0
                    if rel_.get('contained_by_count',0) > 0: score += 4
                    if s_.get('size_category') in ('small','medium'): score += 2
                    if not c_.get('is_dominant'): score += 1
                    if score > 0:
                        candidates.append((score, p_.get('centroid_int')))
                candidates.sort(key=lambda x: -x[0])
                click_targets = [pos for _,pos in candidates]
                action_queue = list(range(len(click_targets)))
                print(f"\n[SIM_CLICK] {len(click_targets)} targets")
            else:
                mv_actions = get_mv_actions()
                targets = []
                for obj in snap.objects:
                    c_ = obj.descriptor.color_axis
                    p_ = obj.descriptor.position_axis
                    s_ = obj.descriptor.scale_axis
                    rel_ = obj.descriptor.relation_axis
                    if p_.get('in_timer_area'): continue
                    score = 0
                    if c_.get('is_rare'): score += 3
                    if rel_.get('contains_count',0) > 0: score += 2
                    if rel_.get('contained_by_count',0) > 0: score += 2
                    if s_.get('size_category') in ('point','tiny','small'): score += 1
                    if score > 0:
                        targets.append((score, p_.get('centroid_int')))
                targets.sort(key=lambda x: -x[0])
                target_positions = [pos for _,pos in targets[:10]]

                if ctrl_pos and mv_actions and target_positions:
                    start = ctrl_pos
                    target_set = set(target_positions)
                    bfs_q = deque([(start, [])])
                    visited = {start}
                    best_path = []; best_dist = float('inf')

                    while bfs_q and len(visited) < 5000:
                        (cr,cc), path = bfs_q.popleft()
                        if len(path) > 80: continue
                        for tr,tc in target_set:
                            dist = abs(cr-tr) + abs(cc-tc)
                            if dist < best_dist:
                                best_dist = dist; best_path = path
                            if dist <= 7:
                                best_path = path; bfs_q.clear(); break
                        if not bfs_q and best_dist <= 7: break
                        for aidx, (dr,dc) in mv_actions.items():
                            nr,nc = cr+dr, cc+dc
                            if (nr,nc) in visited or nr<0 or nr>=64 or nc<0 or nc>=64: continue
                            if not can_pass(nr, nc): continue
                            visited.add((nr,nc))
                            bfs_q.append(((nr,nc), path+[aidx]))

                    action_queue = best_path[:50]
                    print(f"\n[SIM_MOVE] ctrl={ctrl_pos} targets={target_positions[:3]}")
                    print(f"  BFS: visited={len(visited)} path={len(action_queue)} best_dist={best_dist}")
                else:
                    eff = get_effective()
                    action_queue = (eff * 10)[:40] if eff else (available * 10)[:40]
                    print(f"\n[SIM_FALLBACK] len={len(action_queue)}")

            phase = 'execute'

        if phase == 'execute':
            # Auto-replan
            should_replan = len(action_queue) <= 0
            if not should_replan and snap.diff and snap.diff.has_changes:
                changes = (len(snap.diff.moved) + len(snap.diff.appeared)
                           + len(snap.diff.disappeared) + len(snap.diff.size_changed))
                if changes >= 5: should_replan = True
            if not should_replan and total_executes > 0 and total_executes % 15 == 0:
                should_replan = True

            if should_replan and total_executes > 0:
                replan_count += 1
                phase = 'simulate'
                print(f"  [REPLAN #{replan_count}] f={fi}")
                continue

            if action_queue:
                act = action_queue.pop(0)
                total_executes += 1
                last_action_idx = act if not is_click else 0
                last_snap = snap
                grid_ref = g.copy()
                prev_ctrl_pos = ctrl_pos

                if total_executes % 10 == 0 or total_executes <= 3:
                    diff_str = snap.diff.summary() if snap.diff and snap.diff.has_changes else "none"
                    print(f"  [EXEC #{total_executes}] f={fi} act={act} q={len(action_queue)} diff={diff_str} ctrl={ctrl_pos}")

    print(f"\n--- Summary ---")
    print(f"  Probes: {total_probes} | Executes: {total_executes} | Replans: {replan_count}")
    eff = get_effective()
    mvs = get_mv_actions()
    print(f"  Effective: {eff} | Movements: {mvs}")
    print(f"  Passable colors: {passable_colors} | Blocked: {blocked_colors}")
    pcells = int((passable_evidence > 0).sum())
    wcells = int((passable_evidence < -0.5).sum())
    print(f"  Passable cells: {pcells} | Wall cells: {wcells}")
    print(f"  Click: {is_click} | Ctrl: {ctrl_pos}")

rec_dir = 'recordings'
tested = set()
for fname in sorted(os.listdir(rec_dir)):
    if not fname.endswith('.recording.jsonl'): continue
    gp = fname.split('-')[0]
    if gp in tested: continue
    tested.add(gp)
    sim_game(os.path.join(rec_dir, fname), fname.split('.')[0], max_frames=50)

print(f"\n{'='*60}")
print(f"Tested: {sorted(tested)}")
