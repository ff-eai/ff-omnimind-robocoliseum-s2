# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

dt = api.task_info()["row_dt_s"]
belt = api.table_height("head")
report["belt_z"] = round(belt, 4)
SIDE = None
STEP_OUT = 900
w0 = np.array(api.robot_state()["waist"], dtype=float)
WAIST = [-0.55, 0.85, 0.30, float(w0[3]), float(w0[4])]
WAIST_DEEP = [-0.55, 0.85, 0.50, float(w0[3]), float(w0[4])]
waist_lim = Limits(joint_velocity=1.2, joint_acceleration=4.0)
fast = Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.4)
CLOSE = 18
LEAD = 20
AMBUSH_UP = [(0.40, 0.0, 0.0), (0.40, 0.35, -math.pi / 2.0), (0.35, 0.0, 0.0), (0.35, 0.35, -math.pi / 2.0), (0.30, 0.0, 0.0),
             (0.30, 0.35, -math.pi / 2.0)]
AMBUSH = []
D = None


def red_boxes():
    out = []
    for d in api.perceive("red box", max_detections=4):
        if d.points.shape[0] < 80 or d.score < 0.2:
            continue
        if abs(d.bottom_z - belt) > 0.03 or not (0.03 < float(max(d.extent[0], d.extent[1])) < 0.14):
            continue
        out.append(d)
    return out


picked = False
attempts = []


def ambush_plan(lane_x, grasp_z, first):
    errs = []
    waists = (WAIST, WAIST_DEEP) if lane_x < 0.82 else (WAIST_DEEP, WAIST)
    for w in waists:
        for y_i, tilt, toward in AMBUSH:
            best = None
            for yaw in (0.0, math.pi):
                m = api.motion()
                if not first:
                    up = m.grasp_pose(SIDE).copy()
                    up[2, 3] += 0.08
                    try:
                        m.tcp(SIDE, up, linear=True, limits=fast)
                    except MotionError:
                        pass
                target = top_down((lane_x, y_i, grasp_z), yaw, tilt=tilt, tilt_toward=toward)
                pre = target.copy()
                pre[:3, 3] = target[:3, 3] - target[:3, 2] * 0.08
                try:
                    m.tcp(SIDE, pre, gripper="open", limits=fast, waist=w, waist_limits=waist_lim)
                    m.tcp(SIDE, target, linear=True, limits=fast)
                except MotionError as error:
                    errs.append(f"w {w[2]} y {y_i} tilt {tilt} yaw {yaw:.1f}: {str(error)[:60]}")
                    continue
                if best is None or m.rows < best[0].rows:
                    best = (m, target)
            if best is not None:
                return best[0], best[1], y_i, tilt, errs
    return None, None, None, None, errs[-4:]


def strip_count(lane_x, y_i):
    cam = f"hand_{SIDE}"
    if not api.has_depth(cam):
        return -1
    P = api.points(cam, stride=2)
    sel = (np.abs(P[:, 0] - lane_x) < 0.03) & (D * (P[:, 1] - y_i) > -0.10) & (D * (P[:, 1] - y_i) < -0.015) & \
          (P[:, 2] > belt + 0.015) & (P[:, 2] < belt + 0.075)
    return int(np.count_nonzero(sel))


def strip_lead(lane_x, y_i):
    cam = f"hand_{SIDE}"
    if not api.has_depth(cam):
        return None, -1
    P = api.points(cam, stride=2)
    rel = D * (P[:, 1] - y_i)
    sel = (np.abs(P[:, 0] - lane_x) < 0.03) & (rel > -0.12) & (rel < 0.05) & (P[:, 2] > belt + 0.015) & (P[:, 2] < belt + 0.075)
    n = int(np.count_nonzero(sel))
    if n < 30:
        return None, n
    return float(np.percentile(rel[sel], 98)), n


def lane_boxes(cur, lane_x, y_i):
    return [d for d in cur if abs(float(d.mid[0]) - lane_x) < 0.08 and D * (float(d.mid[1]) - y_i) < -0.10]


used = 0
k_dir = 0
shifts = []
while D is None and k_dir < 6:
    k_dir += 1
    a0 = red_boxes()
    api.hold(8, depth_next=True, label="belt_dir")
    used += 8
    b0 = red_boxes()
    shifts += [float(db.mid[1]) - float(da.mid[1]) for da in a0 for db in b0
               if abs(float(db.mid[0]) - float(da.mid[0])) < 0.03 and 0.005 < abs(float(db.mid[1]) - float(da.mid[1])) < 0.08]
    if shifts and abs(float(np.median(shifts))) > 0.01:
        D = 1.0 if float(np.median(shifts)) > 0.0 else -1.0
report["belt_dir"] = {"D": D, "looks": k_dir, "shifts": [round(v_, 3) for v_ in shifts[:8]]}
V0 = None
if D is not None:
    SIDE = "left" if D > 0 else "right"
    AMBUSH = [(x_, D * y_, D * t_) for x_, y_, t_ in AMBUSH_UP]
boxes0 = red_boxes()
report["first"] = [[round(float(v), 3) for v in d.mid] for d in boxes0]


def arrives_late_enough(d, y_ref=0.40):
    if any(abs(float(d.mid[0]) - b_) < 0.03 for b_ in bad_lanes):
        return False
    same_lane = lane is not None and abs(float(d.mid[0]) - lane) < 0.015
    yr = y_i if (same_lane and y_i is not None) else D * y_ref
    return D * (float(d.mid[1]) - yr) < (-0.25 if same_lane else -0.45)


bad_lanes = []


first_call = True
target = None
lane = None
y_i = None
cands = sorted([d for d in boxes0 if D is not None and arrives_late_enough(d)], key=lambda d: -D * float(d.mid[1]))
target_box = cands[0] if cands else None
V = V0
sight = None
half_len = None
while D is not None and used < STEP_OUT - 120 and not picked and len(attempts) < 6 and (api.budget()["phases"][0].get("N") or 0) < 21:
    if target_box is None:
        if first_call:
            m = api.motion()
            m.waist(WAIST, limits=waist_lim)
            m.send(depth_next=True, label="crouch")
            used += m.rows
            first_call = False
        api.hold(30, depth_next=True, label="watch_belt")
        used += 30
        cands = sorted([d for d in red_boxes() if arrives_late_enough(d)], key=lambda d: -D * float(d.mid[1]))
        target_box = cands[0] if cands else None
        continue
    lane_x = float(target_box.mid[0])
    if sight is None or abs(sight[0] - lane_x) > 0.03:
        sight = (lane_x, float(target_box.mid[1]), used)
    if target is None or lane is None or abs(lane_x - lane) > 0.015:
        m, target, y_i, tilt, errs = ambush_plan(lane_x, float(target_box.top_z) - 0.02, first_call)
        rec = {"lane_x": round(lane_x, 3), "y0": round(float(target_box.mid[1]), 3), "errors": errs}
        attempts.append(rec)
        if m is None:
            bad_lanes.append(lane_x)
            ups = sorted([d for d in red_boxes() if arrives_late_enough(d)], key=lambda d: -D * float(d.mid[1]))
            target_box = ups[0] if ups else None
            continue
        m.send(depth_next=True, label="ambush")
        first_call = False
        used += m.rows
        lane = lane_x
        rec.update({"y_i": y_i, "tilt": tilt, "setup_rows": m.rows})
    else:
        rec = {"lane_x": round(lane_x, 3), "reuse": True}
        attempts.append(rec)
    cur = red_boxes()
    mine = lane_boxes(cur, lane, y_i)
    fresh = rec.get("setup_rows") is not None and target_box is not None and abs(float(target_box.mid[0]) - lane) < 0.03
    if not mine:
        if strip_count(lane, y_i) < 60 and not fresh:
            rec["result"] = "none coming"
            api.hold(20, depth_next=True, label="watch_belt")
            used += 20
            ups = sorted([d for d in red_boxes() if arrives_late_enough(d)], key=lambda d: -D * float(d.mid[1]))
            target_box = ups[0] if ups else None
            continue
        y_c = y_i - D * 0.08
    else:
        mb = max(mine, key=lambda d: D * float(d.mid[1]))
        y_c = float(mb.mid[1])
        half_len = min(max(0.5 * float(mb.extent[1]), 0.01), 0.10)
        if sight is not None and used - sight[2] >= 20 and abs(sight[0] - lane) < 0.03:
            v_ = D * (y_c - sight[1]) / ((used - sight[2]) * dt)
            if 0.05 < v_ < 1.0:
                V = v_
        sight = (lane, y_c, used)
    if V is None and mine:
        api.hold(30, depth_next=True, label="speed_look")
        used += 30
        mine2 = lane_boxes(red_boxes(), lane, y_i)
        if mine2:
            mb2 = max(mine2, key=lambda d: D * float(d.mid[1]))
            v_ = D * (float(mb2.mid[1]) - y_c) / (30 * dt)
            if 0.05 < v_ < 1.0:
                V = v_
            y_c = float(mb2.mid[1])
            sight = (lane, y_c, used)
    lead_m = D * (y_i - y_c) - 0.10 - (half_len if half_len is not None else 0.0)
    wait = int(lead_m / V / dt) - 30 if V else 0
    rec["y_c"] = round(y_c, 3)
    rec["v"] = None if V is None else round(V, 3)
    if wait > 0:
        api.hold(min(wait, 240), depth_next=True, label="approach_wait")
        used += min(wait, 240)
    fired = False
    checks = []
    prev_lead = None
    step_lead = None
    unseen = fresh and not mine
    for k in range(24 if unseen else 16):
        if api.dispatches_left() <= 3:
            break
        lead, cnt = strip_lead(lane, y_i)
        checks.append([cnt, None if lead is None else round(lead, 3)])
        if lead is not None:
            if prev_lead is not None and lead > prev_lead + 0.003:
                step_lead = lead - prev_lead
            if lead > 0.020:
                break
            if lead >= -0.0125 - 0.5 * (step_lead if step_lead is not None else 0.04):
                fired = True
                break
            prev_lead = lead
        hk = 6 if (unseen and lead is None) else 3
        api.hold(hk, depth_next=True, label="poll")
        used += hk
    rec["checks"] = checks
    if not fired:
        rec["result"] = "no trigger"
        ups = sorted([d for d in red_boxes() if arrives_late_enough(d)], key=lambda d: -D * float(d.mid[1]))
        target_box = ups[0] if ups else None
        continue
    c = api.motion()
    c.gripper(SIDE, "close", rows=CLOSE)
    c.hold(6)
    for lift_ in (0.12, 0.06):
        try:
            c.tcp(SIDE, offset(target, dz=lift_), linear=True, limits=fast)
            break
        except MotionError:
            try:
                c.tcp(SIDE, offset(target, dz=lift_), limits=fast)
                break
            except MotionError:
                pass
    c.send(depth_next=True, label="catch")
    used += c.rows
    held = api.holding(SIDE)
    rec["result"] = "held" if held else "missed"
    if held:
        picked = True
        break
    o = api.motion()
    o.gripper(SIDE, "open", rows=12)
    try:
        o.tcp(SIDE, target, linear=True, limits=fast)
    except MotionError:
        try:
            o.tcp(SIDE, target, limits=fast)
        except MotionError:
            pass
    o.send(depth_next=True, label="reopen")
    used += o.rows
    ups = sorted([d for d in red_boxes() if arrives_late_enough(d)], key=lambda d: -D * float(d.mid[1]))
    target_box = ups[0] if ups else None
report["attempts"] = attempts
report["rows_used"] = used
report["holding"] = picked


def fwd_rot(roll=0.0):
    z = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, math.cos(roll), math.sin(roll)])
    return np.column_stack([np.cross(y, z), y, z])


SHELF_Q = ("storage shelf", "metal shelf", "warehouse shelf", "shelf")
CARRY_Z0 = 1.10
CARRY_UP = ((0.75, 0.35), (0.75, 0.30), (0.70, 0.35), (0.80, 0.35))
CARRY = tuple((x_, (D if D is not None else 1.0) * y_) for x_, y_ in CARRY_UP)


def carry_pose(x, y, z):
    T = np.eye(4)
    T[:3, :3] = fwd_rot((D if D is not None else 1.0) * math.pi / 2.0)
    T[:3, 3] = (x, y, z)
    return T


def shelves_now():
    found = []
    for q in SHELF_Q:
        for d in api.perceive(q, max_detections=4):
            if d.points.shape[0] < 300 or float(d.top_z) < 1.3 or float(max(d.extent[0], d.extent[1])) < 0.4:
                continue
            found.append(d)
        if found:
            break
    return found


def face_of(det):
    yaw_, length_, width_ = det.principal_yaw()
    nrm = np.array([math.cos(yaw_ - math.pi / 2.0), math.sin(yaw_ - math.pi / 2.0)])
    if float(nrm @ np.asarray(det.mid[:2], dtype=float)) < 0.0:
        nrm = -nrm
    P = det.points
    s_ = P[:, :2] @ nrm
    along = np.array([-nrm[1], nrm[0]])
    return nrm, float(np.percentile(s_, 5)), float(np.median(P[:, :2] @ along)), P


def board_levels(P, nrm, front):
    s_ = P[:, :2] @ nrm
    Q = P[(s_ > front - 0.02) & (s_ < front + 0.15)]
    if Q.shape[0] < 200:
        return [], 0
    along = np.array([-nrm[1], nrm[0]])
    lat = Q[:, :2] @ along
    width = float(np.percentile(lat, 95) - np.percentile(lat, 5))
    edges = np.arange(0.02, float(np.max(Q[:, 2])) + 0.04, 0.02)
    h, e = np.histogram(Q[:, 2], bins=edges)
    nz = h[h > 0]
    thr = 2.0 * float(np.median(nz)) if nz.size else 1e9
    peaks = []
    for i in range(len(h)):
        if h[i] >= thr and h[i] >= (h[i - 1] if i > 0 else 0) and h[i] >= (h[i + 1] if i + 1 < len(h) else 0):
            z_ = 0.5 * (e[i] + e[i + 1])
            band = lat[np.abs(Q[:, 2] - z_) < 0.015]
            if band.size < 20 or float(np.percentile(band, 95) - np.percentile(band, 5)) < 0.6 * width:
                continue
            if peaks and z_ - peaks[-1][0] < 0.08:
                if h[i] > peaks[-1][1]:
                    peaks[-1] = (z_, int(h[i]))
            else:
                peaks.append((z_, int(h[i])))
    return [round(float(z_), 3) for z_, c_ in peaks], int(Q.shape[0])


if picked:
    api.stage_done()
    catch_start = api.base_to_start(api.grasp_pose(SIDE)[:3, 3])
    carton_half_h = None
    carried = None
    for cx_, cy_ in CARRY:
        try:
            m = api.motion()
            m.tcp(SIDE, carry_pose(cx_, cy_, CARRY_Z0), waist=[0.0, 0.0, 0.0, float(w0[3]), float(w0[4])], waist_limits=waist_lim)
            m.hold(10)
            m.send(depth_next=True, label="carry_pose")
            carried = (cx_, cy_)
            break
        except MotionError as error:
            report.setdefault("carry_errors", []).append(str(error)[:70])
    report["carry_pose"] = carried
    report["holding_after_carry"] = bool(api.holding(SIDE))
    if target_box is not None:
        carton_half_h = min(max(0.5 * float(target_box.extent[2]), 0.01), 0.10)
    report["carton_half_h"] = None if carton_half_h is None else round(carton_half_h, 4)
    seen = []
    for k in range(4):
        if k > 0:
            try:
                m = api.motion()
                m.base(0.0, 0.0, math.pi / 2.0, yaw_speed=1.0)
                m.hold(15)
                m.send(depth_next=True, label=f"look_turn{k}")
            except MotionError as error:
                report.setdefault("look_errors", []).append(str(error)[:80])
                break
        for d in shelves_now():
            p0 = api.base_to_start(np.asarray(d.mid, dtype=float))
            seen.append({"start_xy": [float(p0[0]), float(p0[1])], "view": k, "score": float(d.score), "n": int(d.points.shape[0])})
    report["shelves_seen"] = [{"xy": [round(v, 2) for v in s_["start_xy"]], "view": s_["view"], "n": s_["n"]} for s_ in seen]
    goal = None
    if seen:
        cs = np.asarray(catch_start[:2], dtype=float)
        goal = min(seen, key=lambda s_: float(np.linalg.norm(np.asarray(s_["start_xy"]) - cs)))
        report["shelf_goal"] = {"xy": [round(v, 3) for v in goal["start_xy"]], "view": goal["view"]}
    placed = False
    if goal is not None:
        try:
            gb = api.start_to_base(np.array([goal["start_xy"][0], goal["start_xy"][1], 0.0]))
            yaw_to = math.atan2(float(gb[1]), float(gb[0]))
            m = api.motion()
            m.base(0.0, 0.0, yaw_to, yaw_speed=1.0)
            m.hold(20)
            m.send(depth_next=True, label="face_shelf")
            near = [d for d in shelves_now() if float(np.linalg.norm(api.base_to_start(np.asarray(d.mid, dtype=float))[:2] - np.asarray(goal["start_xy"]))) < 0.8]
            det = max(near, key=lambda d: d.points.shape[0]) if near else None
            if det is not None:
                nrm, front, lat, P = face_of(det)
                levels, nq = board_levels(P, nrm, front)
                report["shelf_face"] = {"normal": [round(float(v), 3) for v in nrm], "front": round(front, 3), "lat": round(lat, 3),
                                        "levels": levels, "points": nq}
                if len(levels) >= 3 and carton_half_h is not None:
                    board = levels[2]
                    carry_z = board + carton_half_h + 0.04
                    gb = api.start_to_base(np.array([goal["start_xy"][0], goal["start_xy"][1], 0.0]))[:2]
                    E = P[(np.abs(P[:, 2] - board) < 0.03) & (np.hypot(P[:, 0] - gb[0], P[:, 1] - gb[1]) < 1.2)]
                    if E.shape[0] >= 50:
                        ce = np.median(E[:, :2], axis=0)
                        ev_, V_ = np.linalg.eigh(np.cov((E[:, :2] - ce).T))
                        al_ = V_[:, int(np.argmax(ev_))]
                        nrm = np.array([al_[1], -al_[0]])
                        if float(nrm @ ce) < 0.0:
                            nrm = -nrm
                        front = float(np.percentile(E[:, :2] @ nrm, 5))
                    along = np.array([-nrm[1], nrm[0]])
                    lat_g = float(gb @ along)
                    report["shelf_edge"] = {"n_pts": int(E.shape[0]), "normal": [round(float(v), 3) for v in nrm], "front": round(front, 3),
                                            "lat_goal": round(lat_g, 3)}
                    cx_, cy_ = carried if carried is not None else CARRY[0]
                    face_pt = nrm * (front + 0.12) + along * lat_g
                    heading = math.atan2(float(nrm[1]), float(nrm[0]))
                    ch, sh = math.cos(heading), math.sin(heading)
                    off = np.array([ch * cx_ - sh * cy_, sh * cx_ + ch * cy_])
                    stand = face_pt - off
                    report["stand"] = {"xy": [round(float(v), 3) for v in stand], "heading_deg": round(math.degrees(heading), 1),
                                       "board": board, "carry_z": round(carry_z, 3)}
                    m = api.motion()
                    m.tcp(SIDE, carry_pose(cx_, cy_, carry_z), linear=True, limits=Limits(linear_velocity=0.15))
                    m.hold(6)
                    m.send(depth_next=True, label="carry_height")
                    api.stage_done()
                    m = api.motion()
                    m.base(float(stand[0]), float(stand[1]), heading, speed=0.30, yaw_speed=1.0)
                    m.hold(15)
                    m.send(depth_next=True, label="to_shelf")
                    report["odom_at_shelf"] = [round(float(v), 3) for v in api.robot_state()["odom"]]
                    placed = True
        except MotionError as error:
            report["shelf_error"] = str(error)[:160]
    if placed:
        try:
            g = api.grasp_pose(SIDE)
            report["at_shelf_grasp"] = [round(float(v), 4) for v in g[:3, 3]]
            m = api.motion()
            m.gripper(SIDE, "open", rows=12)
            m.hold(8)
            back = g.copy()
            back[0, 3] -= 0.12
            m.tcp(SIDE, back, linear=True, limits=Limits(linear_velocity=0.3))
            m.hold(10)
            m.send(depth_next=True, label="release")
            report["released"] = True
        except MotionError as error:
            report["release_error"] = str(error)[:120]
report["budget"] = api.budget()["phases"]
api.end_episode()
