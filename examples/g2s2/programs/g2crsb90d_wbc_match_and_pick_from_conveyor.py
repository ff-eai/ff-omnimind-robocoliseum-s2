# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

dt = api.task_info()["row_dt_s"]
belt = api.table_height("head")
report["belt_z"] = round(belt, 4)
SIDE = "left"
STEP_OUT = 900
w0 = np.array(api.robot_state()["waist"], dtype=float)
WAIST = [-0.55, 0.85, 0.30, float(w0[3]), float(w0[4])]
WAIST_DEEP = [-0.55, 0.85, 0.50, float(w0[3]), float(w0[4])]
waist_lim = Limits(joint_velocity=1.2, joint_acceleration=4.0)
fast = Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.4)
CLOSE = 18
LEAD = 20
AMBUSH = [(0.40, 0.0, 0.0), (0.40, 0.35, -math.pi / 2.0), (0.35, 0.0, 0.0), (0.35, 0.35, -math.pi / 2.0), (0.30, 0.0, 0.0),
          (0.30, 0.35, -math.pi / 2.0)]


def red_boxes():
    out = []
    for d in api.perceive("red box", max_detections=4):
        if d.points.shape[0] < 80 or d.score < 0.2:
            continue
        if abs(d.bottom_z - belt) > 0.03 or not (0.03 < float(max(d.extent[0], d.extent[1])) < 0.14):
            continue
        out.append(d)
    return out


used = 0
picked = False
attempts = []


def ambush_plan(lane_x, grasp_z, first):
    errs = []
    waists = (WAIST, WAIST_DEEP) if lane_x < 0.82 else (WAIST_DEEP, WAIST)
    for w in waists:
        for y_i, tilt, toward in AMBUSH:
            m = api.motion()
            if not first:
                up = m.grasp_pose(SIDE).copy()
                up[2, 3] += 0.08
                try:
                    m.tcp(SIDE, up, linear=True, limits=fast)
                except MotionError:
                    pass
            target = top_down((lane_x, y_i, grasp_z), 0.0, tilt=tilt, tilt_toward=toward)
            pre = target.copy()
            pre[:3, 3] = target[:3, 3] - target[:3, 2] * 0.08
            try:
                m.tcp(SIDE, pre, gripper="open", limits=fast, waist=w, waist_limits=waist_lim)
                m.tcp(SIDE, target, linear=True, limits=fast)
                return m, target, y_i, tilt, errs
            except MotionError as error:
                errs.append(f"w {w[2]} y {y_i} tilt {tilt}: {str(error)[:60]}")
    return None, None, None, None, errs[-4:]


def strip_count(lane_x, y_i):
    cam = f"hand_{SIDE}"
    if not api.has_depth(cam):
        return -1
    P = api.points(cam, stride=2)
    sel = (np.abs(P[:, 0] - lane_x) < 0.03) & (P[:, 1] > y_i - 0.10) & (P[:, 1] < y_i - 0.015) & \
          (P[:, 2] > belt + 0.015) & (P[:, 2] < belt + 0.075)
    return int(np.count_nonzero(sel))


def lane_boxes(cur, lane_x, y_i):
    return [d for d in cur if abs(float(d.mid[0]) - lane_x) < 0.08 and float(d.mid[1]) < y_i - 0.10]


boxes0 = red_boxes()
report["first"] = [[round(float(v), 3) for v in d.mid] for d in boxes0]
cands = sorted([d for d in boxes0 if float(d.mid[1]) < 0.15], key=lambda d: -float(d.mid[1]))
target_box = cands[0] if cands else (boxes0[0] if boxes0 else None)
first_call = True
target = None
lane = None
y_i = None
V = 0.21
while used < STEP_OUT - 120 and not picked and len(attempts) < 6 and (api.budget()["phases"][0].get("N") or 0) < 21:
    if target_box is None:
        if first_call:
            m = api.motion()
            m.waist(WAIST, limits=waist_lim)
            m.send(depth_next=True, label="crouch")
            used += m.rows
            first_call = False
        api.hold(30, depth_next=True, label="watch_belt")
        used += 30
        cands = sorted([d for d in red_boxes() if float(d.mid[1]) < -0.05], key=lambda d: -float(d.mid[1]))
        target_box = cands[0] if cands else None
        continue
    lane_x = float(target_box.mid[0])
    if target is None or lane is None or abs(lane_x - lane) > 0.015:
        m, target, y_i, tilt, errs = ambush_plan(lane_x, float(target_box.top_z) - 0.02, first_call)
        rec = {"lane_x": round(lane_x, 3), "y0": round(float(target_box.mid[1]), 3), "errors": errs}
        attempts.append(rec)
        if m is None:
            break
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
    if not mine:
        if strip_count(lane, y_i) < 60:
            rec["result"] = "none coming"
            api.hold(20, depth_next=True, label="watch_belt")
            used += 20
            ups = sorted([d for d in red_boxes() if float(d.mid[1]) < y_i - 0.25], key=lambda d: -float(d.mid[1]))
            target_box = ups[0] if ups else None
            continue
        y_c = y_i - 0.08
    else:
        y_c = float(max(mine, key=lambda d: float(d.mid[1])).mid[1])
    lead_m = (y_i - 0.10) - (y_c + 0.045)
    wait = int(lead_m / V / dt) - 30
    rec["y_c"] = round(y_c, 3)
    if wait > 0:
        api.hold(min(wait, 240), depth_next=True, label="approach_wait")
        used += min(wait, 240)
    fired = False
    checks = []
    for k in range(10):
        cnt = strip_count(lane, y_i)
        checks.append(cnt)
        if cnt >= 80:
            fired = True
            break
        api.hold(3, depth_next=True, label="poll")
        used += 3
    rec["checks"] = checks
    if not fired:
        rec["result"] = "no trigger"
        ups = sorted([d for d in red_boxes() if float(d.mid[1]) < y_i - 0.25], key=lambda d: -float(d.mid[1]))
        target_box = ups[0] if ups else None
        continue
    c = api.motion()
    c.gripper(SIDE, "close", rows=CLOSE)
    c.hold(6)
    c.tcp(SIDE, offset(target, dz=0.12), linear=True, limits=fast)
    c.send(depth_next=True, label="catch")
    used += c.rows
    held = api.holding(SIDE)
    rec["result"] = "held" if held else "missed"
    if held:
        picked = True
        break
    o = api.motion()
    o.gripper(SIDE, "open", rows=12)
    o.tcp(SIDE, target, linear=True, limits=fast)
    o.send(depth_next=True, label="reopen")
    used += o.rows
    ups = sorted([d for d in red_boxes() if float(d.mid[1]) < y_i - 0.25], key=lambda d: -float(d.mid[1]))
    target_box = ups[0] if ups else None
report["attempts"] = attempts
report["rows_used"] = used
report["holding"] = picked
SHELF_START = (-1.614, 3.341)
STAND_START = (-1.614, 2.39)


def fwd_rot(roll=0.0):
    z = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, math.cos(roll), math.sin(roll)])
    return np.column_stack([np.cross(y, z), y, z])


CARRY_Z = 1.305
CARRY = ((0.75, 0.35), (0.75, 0.30), (0.70, 0.35), (0.80, 0.35))


def carry_pose(x, y, z):
    T = np.eye(4)
    T[:3, :3] = fwd_rot(math.pi / 2.0)
    T[:3, 3] = (x, y, z)
    return T


if picked:
    api.stage_done()
    carried = None
    for cx_, cy_ in CARRY:
        try:
            m = api.motion()
            m.tcp(SIDE, carry_pose(cx_, cy_, CARRY_Z), waist=[0.0, 0.0, 0.0, float(w0[3]), float(w0[4])], waist_limits=waist_lim)
            m.hold(10)
            m.send(depth_next=True, label="carry_pose")
            carried = (cx_, cy_)
            break
        except MotionError as error:
            report.setdefault("carry_errors", []).append(str(error)[:70])
    report["carry_pose"] = carried
    if carried is not None:
        g = api.grasp_pose(SIDE)
        report["carry_actual"] = [round(float(v), 4) for v in g[:3, 3]]
        if abs(float(g[2, 3]) - CARRY_Z) > 0.006 or abs(float(g[0, 3]) - carried[0]) > 0.02:
            try:
                m = api.motion()
                m.tcp(SIDE, carry_pose(carried[0], carried[1], CARRY_Z), linear=True, limits=Limits(linear_velocity=0.15))
                m.hold(6)
                m.send(depth_next=True, label="carry_fix")
                report["carry_fixed"] = [round(float(v), 4) for v in api.grasp_pose(SIDE)[:3, 3]]
            except MotionError as error:
                report["carry_fix_error"] = str(error)[:70]
    report["holding_after_carry"] = bool(api.holding(SIDE))
    try:
        odom = np.array(api.robot_state()["odom"], dtype=float)
        d = np.array(STAND_START, dtype=float) - odom[:2]
        c_, s_ = math.cos(-odom[2]), math.sin(-odom[2])
        dx, dy = c_ * d[0] - s_ * d[1], s_ * d[0] + c_ * d[1]
        dyaw = (math.pi / 2.0 - float(odom[2]) + math.pi) % (2.0 * math.pi) - math.pi
        m = api.motion()
        m.base(float(dx), float(dy), float(dyaw), speed=0.35, yaw_speed=1.0)
        m.hold(15)
        m.send(depth_next=True, label="to_shelf")
        report["odom_at_shelf"] = [round(float(v), 3) for v in api.robot_state()["odom"]]
        api.stage_done()
    except MotionError as error:
        report["drive_error"] = str(error)[:160]
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
