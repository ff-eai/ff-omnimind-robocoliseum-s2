# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
report["segments"] = info["segments"]
FAR_STOP = 0.80
STANDOFF = 0.60
TRAY = ["delivery plate", "white tray", "plate", "tray"]


def wrap_half(angle):
    while angle > math.pi / 2.0:
        angle -= math.pi
    while angle <= -math.pi / 2.0:
        angle += math.pi
    return angle


def tray_geometry(det):
    pts = det.points
    low = float(np.percentile(pts[:, 2], 20))
    keep = pts[pts[:, 2] < low + 0.045]
    if keep.shape[0] < 50:
        keep = pts
    xy = keep[:, :2] - np.median(keep[:, :2], axis=0)
    cov = np.cov(xy.T)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, int(np.argmax(vals))]
    yaw = math.atan2(float(major[1]), float(major[0]))
    pm = xy @ major
    minor = np.array([-major[1], major[0]])
    pn = xy @ minor
    lo_m, hi_m = np.percentile(pm, 1), np.percentile(pm, 99)
    lo_n, hi_n = np.percentile(pn, 1), np.percentile(pn, 99)
    centre = np.median(keep[:, :2], axis=0) + major * 0.5 * (lo_m + hi_m) + minor * 0.5 * (lo_n + hi_n)
    top = float(np.percentile(keep[:, 2], 95))
    return {"mid": np.array([float(centre[0]), float(centre[1]), float(np.median(keep[:, 2]))]), "top": top, "yaw": float(yaw),
            "length": float(hi_m - lo_m), "width": float(hi_n - lo_n)}


def deliver():
    state = api.robot_state()
    up = api.motion()
    for side, offsets in (("left", (-1.4, 1.0, -0.5, -0.2)), ("right", (1.4, 1.0, 0.5, -0.2))):
        q = np.array(state["arm"][side], dtype=float)
        q[:4] += np.array(offsets)
        up.joints(side, q)
    up.send(depth_next=True, label="arms_up")

    far = api.find(["blue tissue box", "tissue box", "tissues"], size=(0.05, 0.40))
    if far is None:
        far = api.find(TRAY, size=(0.12, 0.50))
    report["far"] = None if far is None else far.summary()
    if far is None:
        return "no far target"
    c = far.mid
    heading = math.atan2(float(c[1]), float(c[0]))
    distance = math.hypot(float(c[0]), float(c[1])) - FAR_STOP
    report["drive"] = {"distance": round(distance, 3), "heading_deg": round(math.degrees(heading), 1)}
    m = api.motion()
    m.base(distance * math.cos(heading), distance * math.sin(heading), heading, speed=0.3, yaw_speed=0.6)
    m.hold(60)
    m.send(depth_next=True, label="drive_to_plate")
    report["odom_after_drive"] = [round(float(v), 4) for v in api.robot_state()["odom"]]

    near = api.find(TRAY, size=(0.15, 0.50))
    report["tray_near"] = None if near is None else near.summary()
    if near is None:
        return "no tray near"
    g = tray_geometry(near)
    psi = wrap_half(g["yaw"] - math.pi / 2.0)
    cp, sp = math.cos(psi), math.sin(psi)
    ox = float(g["mid"][0]) - cp * STANDOFF
    oy = float(g["mid"][1]) - sp * STANDOFF
    report["align"] = {"dx": round(ox, 3), "dy": round(oy, 3), "dyaw_deg": round(math.degrees(psi), 1)}
    m = api.motion()
    m.base(ox, oy, psi, speed=0.2, yaw_speed=0.5)
    m.hold(45)
    m.send(depth_next=True, label="align_to_tray")
    report["odom_after_align"] = [round(float(v), 4) for v in api.robot_state()["odom"]]

    pred_mid = np.array([STANDOFF, 0.0, float(g["mid"][2])])
    pred = {"mid": pred_mid, "top": g["top"], "yaw": math.pi / 2.0, "length": g["length"], "width": g["width"]}
    again = api.find(TRAY, size=(0.15, 0.50))
    report["tray_aligned"] = None if again is None else again.summary()
    use = pred
    if again is not None and float(np.linalg.norm(again.mid[:2] - pred_mid[:2])) < 0.06:
        use = tray_geometry(again)
        use["length"] = max(use["length"], g["length"])
    report["tray_source"] = "aligned detection" if use is not pred else "near detection + odometry"

    axis = np.array([math.cos(use["yaw"]), math.sin(use["yaw"]), 0.0])
    if axis[1] < 0.0:
        axis = -axis
    half = 0.5 * use["length"] + 0.006
    rim_top = use["top"]

    def side_grasps(yaw_off, pitch, halves=None):
        out = {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            h_side = (halves or {}).get(side, half)
            inward = -sign * axis
            perp = np.array([-inward[1], inward[0], 0.0])
            toward_robot = perp if float(perp[0]) < 0.0 else -perp
            h = math.cos(yaw_off) * inward + math.sin(yaw_off) * toward_robot
            a = math.cos(pitch) * h + math.sin(pitch) * np.array([0.0, 0.0, -1.0])
            a = a / float(np.linalg.norm(a))
            y = np.array([0.0, 0.0, 1.0]) - float(a[2]) * a
            y = y / float(np.linalg.norm(y))
            x_ax = np.cross(y, a)
            edge = np.array([float(use["mid"][0]), float(use["mid"][1]), rim_top - 0.005]) + sign * h_side * axis
            out[side] = pose(edge + a * 0.004, np.column_stack([x_ax, y, a]))
        return out

    orient = [(yo, pt) for pt in (0.0, 0.35, 0.6) for yo in (0.0, 0.4, 0.7)]
    grasps = side_grasps(0.0, 0.0)
    report["grasp_points"] = {s: [round(float(v), 4) for v in t[:3, 3]] for s, t in grasps.items()}
    waist_now = np.array(api.robot_state()["waist"], dtype=float)
    candidates = [None]
    for j1, j2, lean in ((-1.08, 2.65, 0.0), (-1.08, 2.65, 0.2), (-1.08, 2.65, 0.4), (-0.9, 2.4, 0.3), (-1.08, 2.65, 0.55)):
        candidates.append([j1, j2, lean - j1 - j2, float(waist_now[3]), float(waist_now[4])])
    plan = None
    errors = []
    for w, (yo, pt) in [(w, o) for w in candidates for o in orient]:
        if api.reply_elapsed() > 35.0:
            errors.append("search stopped at 35 s of compute")
            break
        trial = api.motion()
        try:
            grasps = side_grasps(yo, pt)
            if w is not None:
                trial.waist(w)
            pre = {}
            for sd, t in grasps.items():
                p_ = t.copy()
                p_[:3, 3] = t[:3, 3] - t[:3, 2] * 0.08
                pre[sd] = p_
            trial.tcp(pre, gripper=0.45)
            trial.tcp(grasps, limits=Limits(linear_velocity=0.10))
            trial.gripper("left", "close", rows=25).gripper("right", "close", rows=25)
            lifted = {}
            for sd, t in grasps.items():
                u = t.copy()
                u[2, 3] += 0.08
                lifted[sd] = u
            trial.tcp(lifted, limits=Limits(linear_velocity=0.10))
            plan = trial
            report["waist_used"] = None if w is None else [round(v, 3) for v in w]
            report["side_pinch"] = {"yaw_off": yo, "pitch": pt}
            break
        except MotionError as error:
            errors.append(f"waist {w}: {error}")
    report["grasp_errors"] = errors[-3:]
    if plan is None:
        return "no crouch / pinch motion"
    yo_used, pt_used = report["side_pinch"]["yaw_off"], report["side_pinch"]["pitch"]
    w_used = report.get("waist_used")
    try:
        tol = (api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad)
        api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = 0.006, math.radians(3.0)
        look = None
        try:
            for back in (0.10, 0.08, 0.12, 0.06):
                try:
                    trial_look = api.motion()
                    if w_used is not None:
                        trial_look.waist(w_used)
                    g0 = side_grasps(yo_used, pt_used)
                    pre0 = {}
                    for sd, t in g0.items():
                        p_ = t.copy()
                        p_[:3, 3] = t[:3, 3] - t[:3, 2] * back
                        pre0[sd] = p_
                    trial_look.tcp(pre0, gripper=0.45)
                    look = trial_look
                    report["look_back"] = back
                    break
                except MotionError as error:
                    report.setdefault("look_errors", []).append(str(error)[:80])
        finally:
            api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = tol
        if look is None:
            raise MotionError("no look pose")
        look.send(depth_next=True, label="look_at_ends")
        ends = {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            cam = f"hand_{side}"
            if not api.has_depth(cam):
                continue
            for q in TRAY:
                found = None
                for d in api.perceive(q, cam, max_detections=3):
                    P = d.points
                    rim = P[np.abs(P[:, 2] - rim_top) < 0.03]
                    if rim.shape[0] < 50:
                        continue
                    s_along = (rim[:, :2] - use["mid"][:2]) @ (sign * axis[:2])
                    edge_s = float(np.percentile(s_along, 99))
                    if 0.15 < edge_s < 0.27:
                        found = edge_s
                        break
                if found is not None:
                    ends[side] = found - 0.004
                    break
            if side not in ends:
                Pw = api.points(cam, stride=2)
                Pw = Pw[np.all(np.isfinite(Pw), axis=1)]
                rel = Pw[:, :2] - np.asarray(use["mid"][:2], dtype=float)
                s_al = rel @ (sign * axis[:2])
                s_ac = rel @ np.array([-axis[1], axis[0]])
                band = (np.abs(Pw[:, 2] - rim_top) < 0.02) & (s_al > 0.10) & (s_al < 0.27) & (np.abs(s_ac) < 0.12)
                if int(np.count_nonzero(band)) > 80:
                    edge_s = float(np.percentile(s_al[band], 99))
                    report.setdefault("ends_raw", {})[side] = [round(edge_s, 4), int(np.count_nonzero(band))]
                    if 0.15 < edge_s < 0.26:
                        ends[side] = edge_s - 0.004
        report["ends_seen"] = {k: round(v, 4) for k, v in ends.items()}
        if ends:
            g1 = side_grasps(yo_used, pt_used, ends)
            trial = api.motion()
            pre = {}
            for sd, t in g1.items():
                p_ = t.copy()
                p_[:3, 3] = t[:3, 3] - t[:3, 2] * 0.08
                pre[sd] = p_
            trial.tcp(pre, gripper=0.45)
            trial.tcp(g1, limits=Limits(linear_velocity=0.10))
            trial.gripper("left", "close", rows=25).gripper("right", "close", rows=25)
            lifted = {}
            for sd, t in g1.items():
                u = t.copy()
                u[2, 3] += 0.08
                lifted[sd] = u
            trial.tcp(lifted, limits=Limits(linear_velocity=0.10))
            plan = trial
            report["grasp_points"] = {sd: [round(float(v), 4) for v in t[:3, 3]] for sd, t in g1.items()}
    except MotionError as error:
        report["look_error"] = str(error)[:160]
    plan.send(depth_next=True, label="crouch_pinch_lift")
    report["holding"] = {s: api.holding(s) for s in ("left", "right")}
    api.stage_done()
    if not all(report["holding"].values()):
        return "lifted?"
    odom = np.array(api.robot_state()["odom"], dtype=float)
    goal = np.array([1.147, -1.637])
    d_start = goal - odom[:2]
    c, s_ = math.cos(-odom[2]), math.sin(-odom[2])
    dx, dy = c * d_start[0] - s_ * d_start[1], s_ * d_start[0] + c * d_start[1]
    report["carry_move"] = [round(float(dx), 3), round(float(dy), 3)]
    try:
        m = api.motion()
        m.base(float(dx), float(dy), 0.0, speed=0.15)
        m.hold(90)
        m.send(depth_next=False, label="carry_to_exit")
        return "lifted, carried"
    except MotionError as error:
        report["carry_error"] = str(error)[:200]
    return "lifted"


report["outcome"] = deliver()
report["budget"] = api.budget()["phases"]
api.end_episode()
