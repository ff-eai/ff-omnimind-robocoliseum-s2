# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

LID_Q = ["cup lid", "round lid", "lid", "small grey disc"]
CUP_Q = ["cup", "grey cup", "tall cup", "mug"]
table = api.table_height("head")
report["table_z"] = round(table, 4)
OPEN_Q = wuji_open()
CLOSED_Q = np.array([0.8166, 0.2442, 0.6353, 0.8531, 0.8475, 0.0, 0.7111, 1.5753] + [0.0] * 12)
PRE_Q = OPEN_Q + 0.35 * (CLOSED_Q - OPEN_Q)
SQUEEZE_Q = OPEN_Q + 0.70 * (CLOSED_Q - OPEN_Q)
PINCH_UP = 0.0025
PINCH_C = {"right": np.array([-0.0379, 0.0495, 0.0115]), "left": np.array([-0.0379, -0.0495, 0.0115])}
PINCH_A = {"right": np.array([0.716, 0.698, 0.0]), "left": np.array([0.716, -0.698, 0.0])}
BOSS_H = 0.0115
LID_H = 0.015
WAISTS = [None, [-0.83, 1.27, 0.31, 0.0, 0.0], [-0.60, 0.90, 0.50, 0.0, 0.0]]
FAST = Limits(joint_velocity=2.4, joint_acceleration=8.0)


def rot_interp(Ra, Rb, f):
    Rr = Ra.T @ Rb
    ang = math.acos(max(-1.0, min(1.0, (float(np.trace(Rr)) - 1.0) / 2.0)))
    if ang < 1e-6:
        return Ra.copy()
    if math.pi - ang < 1e-3:
        w = np.sqrt(np.maximum((np.diag(Rr) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(w))
        ax = Rr[:, k] + np.eye(3)[:, k]
        ax = ax / float(np.linalg.norm(ax))
    else:
        ax = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]) / (2.0 * math.sin(ang))
    K = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
    a = f * ang
    return Ra @ (np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K))


def lowest(mm, q1, sides):
    q0 = np.asarray(mm.arm_end, dtype=float)
    low = 9.0
    for f in (0.25, 0.5, 0.75):
        t = mm.copy()
        q = q0 + f * (q1 - q0)
        for s in sides:
            k = 0 if s == "left" else 7
            t.joints(s, q[k:k + 7], rows=2)
        for s in sides:
            low = min(low, float(t.grasp_pose(s)[2, 3]))
    return low


def high_move(mm, goals, limits, zmin, depth=0):
    trial = mm.copy()
    trial.tcp(goals, limits=limits)
    if lowest(mm, np.asarray(trial.arm_end, dtype=float), list(goals)) >= zmin:
        mm.adopt(trial)
        return mm
    if depth >= 2:
        raise MotionError("approach dips below the safe height")
    mid = {}
    for s, G in goals.items():
        A = mm.grasp_pose(s)
        G = np.asarray(G, dtype=float)
        T = np.eye(4)
        T[:3, :3] = rot_interp(A[:3, :3], G[:3, :3], 0.5)
        T[:3, 3] = 0.5 * (A[:3, 3] + G[:3, 3])
        T[2, 3] = max(float(A[2, 3]), float(G[2, 3]), zmin + 0.02)
        mid[s] = T
    high_move(mm, mid, limits, zmin, depth + 1)
    high_move(mm, goals, limits, zmin, depth + 1)
    return mm


def rim_circle(points):
    p = np.asarray(points, dtype=float)
    if p.shape[0] < 40:
        return None
    top = p[p[:, 2] >= float(np.percentile(p[:, 2], 98)) - 0.015][:, :2]
    if top.shape[0] < 25:
        return None
    A = np.column_stack([top[:, 0], top[:, 1], np.ones(top.shape[0])])
    b = -(top[:, 0] ** 2 + top[:, 1] ** 2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    c = np.array([-D / 2.0, -E / 2.0])
    r2 = float(c @ c - F)
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    rms = float(np.sqrt(np.mean((np.linalg.norm(top - c, axis=1) - r) ** 2)))
    return c, r, rms, int(top.shape[0])


def pinch_grasp(side, point, u_axis):
    a = PINCH_A[side] / float(np.linalg.norm(PINCH_A[side]))
    phi = math.atan2(float(u_axis[1]), float(u_axis[0])) - math.atan2(-float(a[1]), float(a[0]))
    xg = np.array([math.cos(phi), math.sin(phi), 0.0])
    zg = np.array([0.0, 0.0, -1.0])
    rot = np.column_stack([xg, np.cross(zg, xg), zg])
    return pose(np.asarray(point, dtype=float) - rot @ PINCH_C[side], rot)


pool = []
for q in LID_Q + CUP_Q + ["small grey disc", "round object"]:
    for d in api.plausible(api.perceive(q, max_detections=6), size=(0.03, 0.14), table_z=table, z_tolerance=0.05):
        if all(float(np.linalg.norm(d.mid[:2] - o.mid[:2])) > 0.03 for o in pool):
            pool.append(d)
report["pool"] = [[round(float(v), 3) for v in d.mid[:2]] + [round(float(d.top_z) - table, 3)] for d in pool]
lids = [d for d in pool if float(d.top_z) - table < 0.035]
cups = [d for d in pool if float(d.top_z) - table > 0.04]
lid = lids[0] if lids else api.find_object(LID_Q, size=(0.04, 0.12), table_z=table, z_tolerance=0.04)
cup = cups[0] if cups else api.find_object(CUP_Q, size=(0.04, 0.14), table_z=table, z_tolerance=0.05,
                                           exclude=[lid.mid] if lid is not None else [], exclude_radius=0.05)
report["lid"] = None if lid is None else lid.summary()
LID_C = None
if lid is not None and lid.points.shape[0] > 30:
    tp = lid.points[lid.points[:, 2] >= float(np.percentile(lid.points[:, 2], 98)) - 0.004]
    if tp.shape[0] > 15:
        LID_C = np.array([float(np.mean(tp[:, 0])), float(np.mean(tp[:, 1]))])
        report["lid_top_centre"] = [round(float(v), 4) for v in LID_C] + [int(tp.shape[0])]
report["cup"] = None if cup is None else cup.summary()
held = None
if lid is not None:
    c = np.array([float(lid.mid[0]), float(lid.mid[1])])
    sides = ["left", "right"] if float(lid.mid[1]) > 0.0 else ["right", "left"]
    errors = []
    for side in sides:
        sh = api.shoulder(side)[:2]
        cands = []
        for k in range(12):
            th = 2.0 * math.pi * k / 12.0
            u = np.array([math.cos(th), math.sin(th)])
            g0 = pinch_grasp(side, np.array([c[0], c[1], 0.0]), u)
            toward = (sh - c) / max(1e-6, float(np.linalg.norm(sh - c)))
            cands.append((-float(np.dot(g0[:2, 0], toward)), th, u))
        cands.sort(key=lambda t: (abs(abs(round(math.degrees(t[1])) % 360 - 180) - 30), t[0]))
        for waist in WAISTS:
            for score, th, u in cands[:5]:
                g = pinch_grasp(side, np.array([c[0], c[1], float(lid.top_z) - LID_H + BOSS_H + PINCH_UP]), u)
                m = api.motion()
                try:
                    m.hand(side, PRE_Q, rows=10)
                    m.tcp(side, offset(m.grasp_pose(side), dz=0.10), limits=FAST)
                    if waist is not None:
                        m.waist(waist, limits=Limits(joint_velocity=1.0, joint_acceleration=3.0))
                    high_move(m, {side: offset(g, dz=0.12)}, FAST, table + 0.12)
                    m.tcp(side, g, linear=True, limits=Limits(linear_velocity=0.12))
                    m.hand(side, SQUEEZE_Q, rows=30)
                    m.hold(8)
                    m.tcp(side, offset(g, dz=0.13), linear=True, limits=Limits(linear_velocity=0.12))
                except MotionError as error:
                    errors.append(f"{side}/{waist is not None}/{round(math.degrees(th))}: {str(error)[:60]}")
                    continue
                m.send(depth_next=True, label="pick_lid")
                held = {"side": side, "grasp": g, "axis_deg": round(math.degrees(th)), "waist": waist, "u": u, "lid_c": c}
                break
            if held is not None:
                break
        if held is not None:
            break
    report["pick_errors"] = errors[:4]
if held is not None:
    report["held"] = {"side": held["side"], "axis_deg": held["axis_deg"], "lean": held["waist"] is not None}
    chk = api.lifted_check(LID_Q, lid, size=(0.03, 0.12))
    report["lift_check"] = chk
    api.stage_done()
    g0 = held["grasp"]
    held_xy = (g0[:3, 3] + g0[:3, :3] @ PINCH_C[held["side"]])[:2]
    again = []
    for q in CUP_Q + ["round object"]:
        for d in api.plausible(api.perceive(q, max_detections=6), size=(0.03, 0.14), table_z=table, z_tolerance=0.05):
            if 0.04 < float(d.top_z) - table < 0.12 and float(np.linalg.norm(d.mid[:2] - held_xy)) > 0.08:
                again.append(d)
        if again:
            break
    if again:
        best = again[0] if cup is None else min(again, key=lambda d: float(np.linalg.norm(d.mid[:2] - cup.mid[:2])))
        if cup is None or float(np.linalg.norm(best.mid[:2] - cup.mid[:2])) < 0.05:
            report["cup_again"] = [round(float(v), 4) for v in best.mid[:2]] + [round(float(best.top_z), 4), cup is None]
            cup = best
    side = held["side"]
    Gm0 = api.grasp_pose(side)
    pin0 = (Gm0[:3, 3] + Gm0[:3, :3] @ PINCH_C[side])[:2]
    uu = np.asarray(held["u"], dtype=float) / float(np.linalg.norm(held["u"]))
    vv = np.array([-uu[1], uu[0]])
    dl = np.asarray(held["lid_c"], dtype=float) - pin0
    lid_off0 = vv * float(np.dot(dl, vv))
    if float(np.linalg.norm(lid_off0)) > 0.015:
        lid_off0 = lid_off0 * (0.015 / float(np.linalg.norm(lid_off0)))
    lid_off_hand = Gm0[:3, :3].T @ np.array([float(lid_off0[0]), float(lid_off0[1]), 0.0])
    good = False
    if cup is not None:
        for d in api.perceive(cup.label, "head", erode=0, max_detections=5):
            if float(np.linalg.norm(d.mid[:2] - cup.mid[:2])) < 0.03 and d.points.shape[0] > 50:
                f0 = rim_circle(d.points)
                good = f0 is not None and 0.024 < f0[1] < 0.040 and f0[2] < 0.004
                break
    w_place = np.array(api.robot_state()["waist"], dtype=float)
    if cup is None or not good or abs(float(cup.mid[1])) > 0.22:
        other = "right" if side == "left" else "left"
        if cup is not None:
            yaw = float(np.clip(math.atan2(float(cup.mid[1]), float(cup.mid[0])), -0.45, 0.45))
        else:
            yaw = -0.35 if side == "left" else 0.35
        m = api.motion()
        try:
            m.tcp(other, offset(m.grasp_pose(other), dx=-0.10, dz=0.15), limits=FAST)
        except MotionError as error:
            report["look_hand_error"] = str(error)[:80]
        wl = np.array(api.robot_state()["waist"], dtype=float)
        wl[4] = yaw
        m.waist(wl, limits=Limits(joint_velocity=0.8, joint_acceleration=2.0))
        m.hold(10)
        m.send(depth_next=True, label="look_cup")
        Gl = api.grasp_pose(side)
        hxy = (Gl[:3, 3] + Gl[:3, :3] @ PINCH_C[side])[:2]
        seen = []
        for q in CUP_Q + ["round object"]:
            for d in api.plausible(api.perceive(q, max_detections=6), size=(0.03, 0.14), table_z=table, z_tolerance=0.05):
                if 0.04 < float(d.top_z) - table < 0.12 and float(np.linalg.norm(d.mid[:2] - hxy)) > 0.08:
                    seen.append(d)
            if seen:
                break
        report["look"] = {"yaw": round(yaw, 3), "seen": [[round(float(v), 4) for v in d.mid[:2]] for d in seen][:3]}
        if seen:
            best = seen[0] if cup is None else min(seen, key=lambda d: float(np.linalg.norm(d.mid[:2] - cup.mid[:2])))
            if cup is None or float(np.linalg.norm(best.mid[:2] - cup.mid[:2])) < 0.06:
                cup = best
    if cup is not None:
        side, g = held["side"], held["grasp"]
        cup_top = float(cup.top_z)
        cup_pts0 = None
        for d in api.perceive(cup.label, "head", erode=0, max_detections=5):
            if float(np.linalg.norm(d.mid[:2] - cup.mid[:2])) < 0.03 and d.points.shape[0] > 50:
                cup_top = max(cup_top, float(np.percentile(d.points[:, 2], 98)))
                cup_pts0 = d.points
                break
        report["cup_top_est"] = round(cup_top, 4)
        lid_xy = g[:3, 3] + g[:3, :3] @ PINCH_C[side]
        shift = np.array([float(cup.mid[0]) - float(lid_xy[0]), float(cup.mid[1]) - float(lid_xy[1]), 0.0])
        lifted = offset(g, dz=0.13)
        lid_bottom_now = float(g[2, 3] + (g[:3, :3] @ PINCH_C[side])[2]) - BOSS_H + 0.13
        hang = float(lifted[2, 3]) - lid_bottom_now
        lid_c = np.array([float(lid_xy[0]), float(lid_xy[1]), 0.0])
        cup_c = np.array([float(cup.mid[0]), float(cup.mid[1]), 0.0])
        fit = rim_circle(cup_pts0 if cup_pts0 is not None else cup.points)
        if fit is not None and 0.024 < fit[1] < 0.040 and fit[2] < 0.005 and float(np.linalg.norm(fit[0] - cup.mid[:2])) < 0.03:
            cup_c = np.array([float(fit[0][0]), float(fit[0][1]), 0.0])
        report["cup_fit"] = None if fit is None else [round(float(fit[0][0]), 4), round(float(fit[0][1]), 4), round(1000 * fit[1], 1), round(1000 * fit[2], 1), fit[3]]
        Gm = api.grasp_pose(side)
        mm0 = api.motion()
        Gc = mm0.grasp_pose(side)
        sag = (Gm[:3, 3] + Gm[:3, :3] @ PINCH_C[side]) - (Gc[:3, 3] + Gc[:3, :3] @ PINCH_C[side])
        if float(np.linalg.norm(sag)) > 0.04:
            sag = np.zeros(3)
        lid_off = lid_off0
        report["place"] = {"cup_top": round(cup_top, 4), "shift": [round(float(v), 4) for v in shift], "cup_c": [round(float(v), 4) for v in cup_c[:2]],
                           "sag_mm": [round(1000 * float(v), 1) for v in sag], "lid_off_mm": [round(1000 * float(v), 1) for v in lid_off],
                           "lid_vs_pinch_mm": [round(1000 * float(v), 1) for v in dl]}
        w_now = w_place
        placed = False
        errors = []
        for dyaw in (0.0, 0.35, -0.35, 0.6, -0.6):
            for psi_deg in (0, 30, -30, 60, -60, 90, -90, 135, -135):
                R = rotz(math.radians(psi_deg))
                lo = (R[:3, :3] @ (g[:3, :3] @ lid_off_hand))[:2]
                cup_aim = cup_c - np.array([float(sag[0]) + float(lo[0]), float(sag[1]) + float(lo[1]), 0.0])
                T = np.eye(4)
                T[:3, :3] = R[:3, :3]
                T[:3, 3] = cup_aim - R[:3, :3] @ lid_c
                at = T @ lifted
                above = offset(at, dz=(cup_top + 0.025) - lid_bottom_now - float(sag[2]))
                m = api.motion()
                try:
                    w = w_now.copy()
                    w[4] += dyaw
                    if m.waist_end is None or float(np.max(np.abs(np.asarray(m.waist_end) - w))) > 0.01:
                        m.waist(w, limits=Limits(joint_velocity=0.6, joint_acceleration=1.5))
                    high_move(m, {side: above}, Limits(joint_velocity=1.0, joint_acceleration=3.0), cup_top + 0.02 + hang)
                    trial = m.copy()
                    trial.tcp(side, offset(above, dz=-0.03), linear=True, limits=Limits(linear_velocity=0.04))
                    m.hold(15)
                except MotionError as error:
                    errors.append(f"{dyaw}/{psi_deg}: {str(error)[:50]}")
                    continue
                report["place"].update({"psi_deg": psi_deg, "waist_yaw": dyaw})
                m.send(depth_next=True, label="lid_over_cup")
                Gm = api.grasp_pose(side)
                pm = Gm[:3, 3] + Gm[:3, :3] @ PINCH_C[side]
                lo_now = (Gm[:3, :3] @ lid_off_hand)[:2]
                exy = cup_c[:2] - lo_now - pm[:2]
                if float(np.linalg.norm(exy)) > 0.03:
                    exy = exy * (0.03 / float(np.linalg.norm(exy)))
                dz = (cup_top + 0.004) - (float(pm[2]) - BOSS_H - PINCH_UP)
                dz = min(max(dz, -0.05), 0.0)
                report["place"].update({"err_mm": [round(1000 * float(v), 1) for v in exy], "lower_mm": round(1000 * dz, 1)})
                m = api.motion()
                Gc = m.grasp_pose(side)
                try:
                    m.tcp(side, offset(Gc, dx=float(exy[0]), dy=float(exy[1])), linear=True, limits=Limits(linear_velocity=0.03))
                    m.tcp(side, offset(Gc, dx=float(exy[0]), dy=float(exy[1]), dz=dz), linear=True, limits=Limits(linear_velocity=0.03))
                except MotionError as error:
                    report["place"]["correct_error"] = str(error)[:80]
                    m = api.motion()
                    try:
                        m.tcp(side, offset(Gc, dz=dz), linear=True, limits=Limits(linear_velocity=0.03))
                    except MotionError as error2:
                        report["place"]["lower_error"] = str(error2)[:80]
                        m = api.motion()
                m.hold(10)
                m.hand(side, PRE_Q, rows=12)
                try:
                    m.tcp(side, offset(m.grasp_pose(side), dz=0.08), linear=True, limits=Limits(linear_velocity=0.12))
                except MotionError as error:
                    report["place"]["retreat_error"] = str(error)[:80]
                m.send(depth_next=False, label="lid_on_cup")
                api.stage_done()
                placed = True
                break
            if placed:
                break
        if not placed:
            report["place_errors"] = errors[:4]
report["budget"] = api.budget()["phases"]
api.end_episode()
