# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

QUERIES = ["small cardboard box", "box", "white box", "brown cardboard box"]
fast = Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30)
waist_lim = Limits(joint_velocity=1.0, joint_acceleration=3.0)
w0 = [float(v) for v in api.robot_state()["waist"]]
start_arm = {s: np.array(q, dtype=float) for s, q in api.robot_state()["arm"].items()}
PLATFORM_EDGE_X = -0.40
PLATFORM_Y = 1.80
SHELF_STOP_X = 0.33
WAIST_LOW = [-0.9, 2.2, -0.3, w0[3], w0[4]]
WAIST_DROP = [-0.7, 1.5, 0.0, w0[3], w0[4]]


def odom():
    return np.array(api.robot_state()["odom"], dtype=float)


def go_start_frame(x, y, yaw, label, speed=0.3):
    o = odom()
    d = np.array([x, y]) - o[:2]
    c, s_ = math.cos(-o[2]), math.sin(-o[2])
    dx, dy = c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]
    dyaw = (yaw - o[2] + math.pi) % (2.0 * math.pi) - math.pi
    m = api.motion()
    m.base(float(dx), float(dy), float(dyaw), speed=speed, yaw_speed=1.0)
    m.hold(25)
    m.send(depth_next=True, label=label)
    report.setdefault("legs", []).append([label] + [round(float(v), 3) for v in odom()])


def parcels_now(shelf_z):
    found = []
    for q in QUERIES:
        for d in api.plausible(api.perceive(q, max_detections=6), size=(0.03, 0.11), table_z=shelf_z, z_tolerance=0.06,
                               away_from_grippers=0.08):
            if d.points.shape[0] < 30 or not (0.03 < float(max(d.extent[0], d.extent[1])) < 0.11):
                continue
            if abs(float(d.bottom_z) - shelf_z) > 0.05:
                continue
            if float(np.min(d.extent)) < 0.032:
                continue
            hands = [api.grasp_pose(sd)[:3, 3] for sd in ("left", "right")] + [api.tcp_pose(sd)[:3, 3] for sd in ("left", "right")]
            if min(float(np.linalg.norm(d.mid - h)) for h in hands) < 0.10:
                continue
            if all(float(np.linalg.norm(d.mid[:2] - f.mid[:2])) > 0.04 for f in found):
                found.append(d)
    return found


def carry_pose():
    m = api.motion()
    targets = {}
    for side, sy in (("left", 0.22), ("right", -0.22)):
        t = top_down((0.38, sy, 1.10), 0.0)
        targets[side] = t
    try:
        m.tcp(targets, free_axis=(0.0, 0.0, 1.0), limits=fast)
    except MotionError:
        m = api.motion()
        for side in ("left", "right"):
            m.joints(side, start_arm[side], limits=fast)
    m.send(depth_next=True, label="carry")


def trip(k, shelf_z):
    got = 0
    try:
        m = api.motion()
        m.waist([-0.05, 0.05, 0.0, w0[3], w0[4]], limits=waist_lim)
        m.hold(10)
        m.send(depth_next=True, label=f"stand_tall{k}")
    except MotionError as error:
        report[f"trip{k}_stand_error"] = str(error)[:120]
    for side, lat in (("right", -0.15), ("left", 0.15)):
        seen = parcels_now(shelf_z)
        report.setdefault(f"trip{k}_seen", []).append([[round(float(v), 3) for v in d.mid] for d in seen])
        if not seen:
            continue
        ref = min(seen, key=lambda d: float(d.mid[1])) if side == "right" else max(seen, key=lambda d: float(d.mid[1]))
        o = odom()
        step = np.array([float(ref.mid[0]) - 0.62, float(ref.mid[1]) - lat])
        step[0] = min(float(step[0]), SHELF_STOP_X - float(o[0]))
        try:
            if float(np.linalg.norm(step)) > 0.03:
                m = api.motion()
                m.base(float(step[0]), float(step[1]), 0.0, speed=0.25)
                m.hold(90)
                m.send(depth_next=True, label=f"to_parcel{k}_{side}")
            else:
                step = np.zeros(2)
        except MotionError as error:
            report[f"trip{k}_{side}_step_error"] = str(error)[:120]
            continue
        exp = np.array([float(ref.mid[0]) - float(step[0]), float(ref.mid[1]) - float(step[1])])
        report.setdefault(f"trip{k}_exp", []).append([round(float(v), 3) for v in exp])
        raw = []
        for q in QUERIES:
            for d in api.perceive(q, max_detections=6):
                raw.append([q[:10], round(float(d.score), 3), int(d.points.shape[0])] + [round(float(v), 3) for v in d.mid] +
                           [round(float(v), 3) for v in d.extent] + [round(float(d.bottom_z), 3)])
        report.setdefault(f"trip{k}_{side}_raw", raw[:16])
        shelf_now = float(api.table_height("head"))
        report[f"trip{k}_{side}_shelf_now"] = round(shelf_now, 4)
        sz = shelf_now if abs(shelf_now - shelf_z) < 0.05 else shelf_z
        pk = api.pick_object(side, QUERIES,
                             find={"size": (0.03, 0.11), "table_z": sz, "z_tolerance": 0.06, "side": side, "away_from_grippers": 0.06,
                                   "region": (lambda c, e=exp: float(np.linalg.norm(np.asarray(c[:2], dtype=float) - e)) < 0.10)},
                             grasp={"depth_into": 0.03, "lift": 0.08, "above": 0.08, "settle_after_close": 10, "corner_yaw": True},
                             refine_wrist=0.10, limits=fast, label=f"pick{k}_{side}")
        report[f"trip{k}_{side}"] = {kk: pk.get(kk) for kk in ("ok", "reason", "attempts")}
        if pk.get("ok"):
            got += 1
            try:
                m = api.motion()
                up = m.grasp_pose(side).copy()
                up[2, 3] += 0.10
                up[1, 3] += -0.10 if side == "right" else 0.10
                m.tcp(side, up, limits=fast)
                m.send(depth_next=True, label=f"aside{k}_{side}")
            except MotionError:
                pass
    carry_pose()
    return got


shelf = api.table_height("head")
report["shelf_z"] = round(shelf, 4)
placed = 0
for k in range(2):
    got = trip(k, shelf)
    if got == 0:
        break
    try:
        o = odom()
        go_start_frame(float(o[0]) - 0.35, float(o[1]), 0.0, f"back{k}")
        go_start_frame(PLATFORM_EDGE_X + 0.32, PLATFORM_Y, math.pi, f"to_cart{k}")
        m = api.motion()
        m.waist(WAIST_DROP, limits=waist_lim)
        targets = {}
        for side, sy in (("left", 0.14), ("right", -0.14)):
            targets[side] = top_down((0.50, sy, 0.35), 0.0)
        m.tcp(targets, free_axis=(0.0, 0.0, 1.0), limits=fast)
        m.gripper("left", "open", rows=12).gripper("right", "open", rows=12)
        m.hold(15)
        m.tcp({sd: offset(t, dz=0.15) for sd, t in targets.items()}, free_axis=(0.0, 0.0, 1.0), limits=fast)
        m.waist(w0, limits=waist_lim)
        m.send(depth_next=True, label=f"drop{k}")
        placed += got
        report[f"placed_after_trip{k}"] = placed
        if placed >= 2 and k == 0:
            api.stage_done()
        if placed >= 4:
            api.stage_done()
    except MotionError as error:
        report[f"cart_error{k}"] = str(error)[:200]
        break
    if k == 0:
        try:
            go_start_frame(0.0, 0.0, 0.0, "back_to_shelf")
        except MotionError as error:
            report["return_error"] = str(error)[:200]
            break


HANDLE_X, HANDLE_Y = -0.694, 1.25
try:
    o = odom()
    go_start_frame(float(o[0]) + 0.10 if float(o[1]) > 1.0 else float(o[0]), float(o[1]), float(o[2]), "clear_cart")
    go_start_frame(0.05, 0.69, math.pi / 2.0, "beside_handle")
    go_start_frame(HANDLE_X, HANDLE_Y - 0.55, math.pi / 2.0, "at_handle")
    P = api.points("head", stride=2)
    P = P[np.all(np.isfinite(P), axis=1)]
    sel = P[(P[:, 0] > 0.30) & (P[:, 0] < 0.90) & (np.abs(P[:, 1]) < 0.35) & (P[:, 2] > 0.75) & (P[:, 2] < 1.10)]
    bar = None
    if sel.shape[0] > 40:
        ztop = float(np.percentile(sel[:, 2], 97))
        top = sel[sel[:, 2] > ztop - 0.03]
        bar = (float(np.median(top[:, 0])), float(np.median(top[:, 1])), float(np.median(top[:, 2])) - 0.012, int(top.shape[0]))
    report["handle_bar"] = None if bar is None else [round(v, 3) for v in bar[:3]] + [bar[3]]
    bx, by, bz = (bar[0], bar[1], bar[2]) if bar is not None else (0.55, 0.0, 0.88)
    m = api.motion()
    grips = {sd: top_down((bx, by + sy, bz), 0.0) for sd, sy in (("left", 0.13), ("right", -0.13))}
    m.tcp({sd: offset(t, dz=0.08) for sd, t in grips.items()}, gripper="open", free_axis=None, limits=fast)
    m.tcp(grips, limits=Limits(linear_velocity=0.10))
    m.gripper("left", "close", rows=20).gripper("right", "close", rows=20)
    m.hold(15)
    m.send(depth_next=True, label="grip_handle")
    report["handle_held"] = {sd: bool(api.holding(sd)) for sd in ("left", "right")}
    D = (bx + 0.893)
    step = math.radians(15.0)
    for k in range(6):
        m = api.motion()
        m.base(float(D * (1.0 - math.cos(step))), float(-D * math.sin(step)), float(step), speed=0.35, yaw_speed=0.8)
        m.hold(6)
        m.send(depth_next=(k == 5), label=f"swing_{k}")
    report["after_swing_odom"] = [round(float(v), 3) for v in odom()]
    for k, dist in enumerate((2.5, 2.5, 2.3)):
        m = api.motion()
        m.base(float(dist), 0.0, 0.0, speed=0.35, yaw_speed=0.5)
        m.hold(10)
        m.send(depth_next=(k == 2), label=f"push_{k}")
    report["after_push_odom"] = [round(float(v), 3) for v in odom()]
    api.stage_done()
    m = api.motion()
    m.gripper("left", "open", rows=12).gripper("right", "open", rows=12)
    m.tcp({sd: offset(t, dz=0.10) for sd, t in grips.items()}, limits=fast)
    m.base(-0.35, 0.0, 0.0, speed=0.25)
    m.send(depth_next=True, label="let_go_cart")
except MotionError as error:
    report["cart_push_error"] = str(error)[:200]
report["budget"] = api.budget()["phases"]
api.end_episode()
