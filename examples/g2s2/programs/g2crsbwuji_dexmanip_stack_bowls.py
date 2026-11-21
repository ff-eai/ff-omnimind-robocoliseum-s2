# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

table = api.table_height("head")
report["table_z"] = round(table, 4)
W0 = np.array(api.robot_state()["waist"], dtype=float)
report["waist0"] = [round(float(v), 3) for v in W0]
RIM = None
R_MEAS = None
NEST = None
SQ_H = 0.07
CLEAR = 0.07
DROP = 0.02
PLAN_S = 30.0
FAST = Limits(joint_velocity=1.6, joint_acceleration=5.0)
SLOW = Limits(joint_velocity=1.2, joint_acceleration=3.0)
LIFT = Limits(joint_velocity=1.0, joint_acceleration=3.0)
WL = Limits(joint_velocity=1.0, joint_acceleration=2.5)
FREE = (0, 0, 1)
SIDES = ("left", "right")


def bowls_now():
    found = []
    for q in ("bowl", "ceramic bowl", "round bowl"):
        dets = api.plausible(api.perceive(q, max_detections=6), size=(0.08, 0.30), table_z=table, z_tolerance=0.08)
        for d in dets:
            if all(float(np.linalg.norm(d.mid[:2] - f.mid[:2])) > 0.06 for f in found):
                found.append(d)
        if len(found) >= 3:
            break
    return found


def xy(v):
    return np.asarray(v, dtype=float).reshape(-1)[:2]


def dist2(a, b):
    return float(np.linalg.norm(xy(a) - xy(b)))


def rot_z(a):
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def yaw_to(c):
    return float(np.clip(math.atan2(float(c[1]), float(c[0])), -0.6, 0.6))


def waist_at(a):
    w = W0.copy()
    w[4] = a
    return w


def palms():
    return 0.5 * (api.grasp_pose("left")[:3, 3] + api.grasp_pose("right")[:3, 3])


def rim_centre(points, r=None):
    p = np.asarray(points, dtype=float)
    if p.shape[0] < 30:
        return None
    top = p[p[:, 2] >= float(np.percentile(p[:, 2], 98)) - 0.012][:, :2]
    if top.shape[0] < 20:
        return None
    A = np.column_stack([top[:, 0], top[:, 1], np.ones(top.shape[0])])
    D, E, F = np.linalg.lstsq(A, -(top[:, 0] ** 2 + top[:, 1] ** 2), rcond=None)[0]
    c = np.array([-D / 2.0, -E / 2.0])
    rr = math.sqrt(max(float(c @ c - F), 1e-6)) if r is None else float(r)
    if r is not None:
        c = top.mean(axis=0)
    for rnd in (0, 1):
        for k in range(15):
            d = top - c
            n = np.linalg.norm(d, axis=1) + 1e-9
            if r is None:
                J = np.column_stack([-d / n[:, None], -np.ones(n.shape[0])])
                step = np.linalg.lstsq(J, -(n - rr), rcond=None)[0]
                c = c + step[:2]
                rr = rr + float(step[2])
            else:
                J = -d / n[:, None]
                step = np.linalg.lstsq(J, -(n - rr), rcond=None)[0]
                c = c + step
            if float(np.linalg.norm(step)) < 1e-4:
                break
        if rnd == 0:
            keep = np.abs(np.linalg.norm(top - c, axis=1) - rr) < 0.015
            if int(keep.sum()) < 20:
                break
            top = top[keep]
    rms = float(np.sqrt(np.mean((np.linalg.norm(top - c, axis=1) - rr) ** 2)))
    return c, rms, int(top.shape[0]), rr


def centre_of(det, near=None, tol=0.03):
    fit = rim_centre(det.points, R_MEAS)
    if fit is not None and fit[1] < 0.012 and (near is None or dist2(fit[0], near) < tol):
        return np.asarray(fit[0], dtype=float), round(1000 * fit[1], 1)
    return xy(det.mid), None


PALM_OUT = float(api.episode.embodiment.arm("left").grasp_from_parent[0, 3])


def squeeze_targets(c, half_y, z, a, out_y=0.0):
    R = rot_z(a)
    out = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        approach = R @ np.array([0.0, -sign, 0.0])
        x_axis = R @ np.array([-1.0, 0.0, 0.0])
        y_axis = np.cross(approach, x_axis)
        face = np.array([float(c[0]), float(c[1]), z]) + R @ np.array([0.0, sign * (half_y + out_y), 0.0])
        out[side] = pose(face + approach * (PALM_OUT - 0.018), np.column_stack([x_axis, y_axis, approach]))
    return out


def move(mm, goals, limits, max_jump=0.6, depth=0):
    trial = mm.copy()
    trial.tcp(goals, limits=limits, free_axis=FREE)
    if float(np.max(np.abs(np.asarray(trial.arm_end) - np.asarray(mm.arm_end)))) <= max_jump:
        mm.adopt(trial)
        return mm
    if depth >= 2:
        raise MotionError("IK branch flip on a short move")
    half = {}
    for s, G in goals.items():
        T = np.array(G, dtype=float).copy()
        T[:3, 3] = 0.5 * (mm.grasp_pose(s)[:3, 3] + np.asarray(G, dtype=float)[:3, 3])
        half[s] = T
    move(mm, half, limits, max_jump, depth + 1)
    move(mm, goals, limits, max_jump, depth + 1)
    return mm


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
    trial.tcp(goals, limits=limits, free_axis=FREE)
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
        T[2, 3] = max(float(A[2, 3]), float(G[2, 3]))
        mid[s] = T
    high_move(mm, mid, limits, zmin, depth + 1)
    high_move(mm, goals, limits, zmin, depth + 1)
    return mm


def plan_squeeze(mm, c, half_y, a):
    z = table + SQ_H
    pre = squeeze_targets(c, half_y, z, a, out_y=0.10)
    high_move(mm, {s: offset(T, dz=0.15) for s, T in pre.items()}, FAST, table + 0.15)
    move(mm, pre, FAST, 0.8)
    mm.hand("left", "open", rows=6).hand("right", "open", rows=6)
    move(mm, squeeze_targets(c, half_y, z, a), Limits(), 0.6)
    return mm


def squeeze_plan(det, m, dst=None, rise=None):
    half_y = 0.5 * float(det.extent[1])
    first = (None, None)
    for a in (0.0, yaw_to(det.mid)):
        trial = m.copy()
        try:
            if abs(a) > 0.02 or (trial.waist_end is not None and abs(float(trial.waist_end[4])) > 0.02):
                trial.waist(waist_at(a), limits=WL)
            plan_squeeze(trial, det.mid, half_y, a)
        except MotionError:
            continue
        if dst is None:
            return a, trial
        if first[1] is None:
            first = (a, trial)
        t2 = trial.copy()
        P = {s: t2.grasp_pose(s) for s in SIDES}
        try:
            move(t2, {s: offset(T, dz=0.03) for s, T in P.items()}, LIFT, 0.5)
            move(t2, {s: offset(T, dz=rise) for s, T in P.items()}, SLOW, 0.8)
            carry(a, xy(det.mid), dst, t2, quick=True)
        except MotionError:
            continue
        return a, trial
    return first


def rigid(T, c_from, c_to, da, dz):
    R = rot_z(da)
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    rel = T[:3, 3] - np.array([float(c_from[0]), float(c_from[1]), 0.0])
    out[:3, 3] = np.array([float(c_to[0]), float(c_to[1]), 0.0]) + R @ rel + np.array([0.0, 0.0, dz])
    return out


def carry_try(a0, a1, turn, src, dst, m, n=8):
    start = {s: m.grasp_pose(s) for s in SIDES}

    def at(f):
        c = (1 - f) * xy(src) + f * xy(dst)
        a = (1 - f) * a0 + f * a1
        return {s: rigid(T, src, c, (a - a0) if turn else 0.0, 0.0) for s, T in start.items()}, a

    def step(f0, f1, depth):
        targets, a = at(f1)
        trial = m.copy()
        trial.tcp(targets, limits=SLOW, waist=waist_at(a), waist_limits=WL, free_axis=FREE)
        jump = float(np.max(np.abs(np.asarray(trial.arm_end) - np.asarray(m.arm_end))))
        if jump > 0.5:
            if depth >= 2:
                raise MotionError(f"carry IK jump {jump:.2f} rad at f={f1:.3f}")
            mid = 0.5 * (f0 + f1)
            step(f0, mid, depth + 1)
            step(mid, f1, depth + 1)
            return
        m.adopt(trial)

    for k in range(1, n + 1):
        step((k - 1) / n, k / n, 0)
    return m


def carry(a0, src, dst, m, quick=False):
    full = yaw_to(dst)
    last = None
    options = ((full, True), (full, False), (0.5 * full, True), (0.5 * full, False), (a0, False))
    for a1, turn in (options[:2] if quick else options):
        if api.reply_elapsed() > PLAN_S:
            last = MotionError(f"plan time {api.reply_elapsed():.1f} s")
            break
        try:
            plan = carry_try(a0, a1, turn, src, dst, m.copy(), n=4 if quick else 8)
        except MotionError as error:
            last = error
            continue
        m.adopt(plan)
        return a1, [round(a1, 3), turn]
    raise MotionError(f"no carry plan: {last}")


def snapshot(tag):
    pm = palms()
    report.setdefault("snap", []).append({"at": tag, "bowls": [[round(float(v), 3) for v in b.mid[:2]] + [round(float(b.top_z), 3)] for b in bowls_now()],
                                          "palms": [round(float(v), 3) for v in pm],
                                          "yaw": round(float(api.robot_state()["waist"][4]), 3)})


def find_base(dets, base_c):
    near = [d for d in dets if dist2(d.mid, base_c) < 0.09]
    if not near:
        return None
    return max(near, key=lambda d: float(d.top_z))


def cycle(i, det, base_c, rim_t, a0, m, bias):
    m.hand("left", wuji_power(0.45), rows=15).hand("right", wuji_power(0.45), rows=15)
    P_cmd = {s: m.grasp_pose(s) for s in SIDES}
    m.send(depth_next=True, label=f"squeeze_{i}")
    pm = palms()
    z_hold = float(pm[2]) - table
    gap_b = float(np.linalg.norm(api.grasp_pose("left")[:3, 3] - api.grasp_pose("right")[:3, 3]))
    dets = bowls_now()
    own = [d for d in dets if dist2(d.mid, pm) < 0.10]
    held_c = xy(pm)
    if own:
        c, rms = centre_of(min(own, key=lambda d: dist2(d.mid, pm)), near=pm, tol=0.04)
        if rms is not None:
            held_c = c
    held_off = held_c - xy(pm)
    if float(np.linalg.norm(held_off)) > 0.025 or gap_b < 0.5 * 2.0 * R_MEAS:
        report.setdefault("resqueeze", []).append([i, [round(1000 * float(v), 1) for v in held_off]])
        m = api.motion()
        P = {s: m.grasp_pose(s) for s in SIDES}
        try:
            trial = m.copy()
            trial.hand("left", "open", rows=8).hand("right", "open", rows=8)
            outs = {}
            for s, T in P.items():
                To = np.array(T, dtype=float).copy()
                To[:3, 3] = To[:3, 3] - To[:3, 2] * 0.06
                outs[s] = To
            move(trial, outs, FAST, 0.6)
            move(trial, squeeze_targets(held_c, 0.5 * float(det.extent[1]), table + SQ_H, a0, out_y=0.06), FAST, 0.6)
            move(trial, squeeze_targets(held_c, 0.5 * float(det.extent[1]), table + SQ_H, a0), Limits(), 0.6)
            trial.hand("left", wuji_power(0.45), rows=15).hand("right", wuji_power(0.45), rows=15)
            m.adopt(trial)
            P_cmd = {s: m.grasp_pose(s) for s in SIDES}
            m.send(depth_next=True, label=f"resqueeze_{i}")
            pm = palms()
            gap_b = float(np.linalg.norm(api.grasp_pose("left")[:3, 3] - api.grasp_pose("right")[:3, 3]))
            z_hold = float(pm[2]) - table
            dets = bowls_now()
            own = [d for d in dets if dist2(d.mid, pm) < 0.10]
            held_c = xy(pm)
            if own:
                c, rms = centre_of(min(own, key=lambda d: dist2(d.mid, pm)), near=pm, tol=0.04)
                if rms is not None:
                    held_c = c
            held_off = held_c - xy(pm)
        except MotionError as error:
            report.setdefault("resqueeze_error", []).append(str(error)[:100])
    tg = find_base([d for d in dets if dist2(d.mid, held_c) > 0.10], base_c)
    rim_prior = rim_t
    if tg is not None:
        base_c = centre_of(tg, near=base_c, tol=0.03)[0] if dist2(tg.mid, base_c) < 0.06 else xy(tg.mid)
        seen = float(tg.top_z) + bias
        if rim_prior - 0.06 <= seen <= rim_prior + 0.03:
            rim_t = seen
    m = api.motion()
    try:
        move(m, P_cmd, Limits(), 0.6)
    except MotionError as error:
        report.setdefault("recommand_error", []).append(str(error)[:80])
    P = {s: m.grasp_pose(s) for s in SIDES}
    pz_plan = 0.5 * (P["left"][2, 3] + P["right"][2, 3])
    rise = rim_t + CLEAR + z_hold - float(pm[2])
    report.setdefault("cycle", []).append({"i": i, "gap_b_mm": round(1000 * gap_b, 1), "z_hold_mm": round(1000 * z_hold, 1), "held_off_mm": [round(1000 * float(v), 1) for v in held_off],
                                           "base_c": [round(float(v), 3) for v in base_c], "rim_t": round(rim_t, 3),
                                           "rise_mm": round(1000 * rise, 1), "plan_vs_meas_mm": round(1000 * (pz_plan - float(pm[2])), 1)})
    errs = []

    def plan_lift(mstart):
        Ps = {s: mstart.grasp_pose(s) for s in SIDES}
        for r in (max(rise, 0.06), 0.12, 0.09):
            if api.reply_elapsed() > PLAN_S:
                errs.append(f"plan time {api.reply_elapsed():.1f} s")
                break
            trial = mstart.copy()
            try:
                move(trial, {s: offset(T, dz=0.03) for s, T in Ps.items()}, LIFT, 0.5)
                move(trial, {s: offset(T, dz=r) for s, T in Ps.items()}, SLOW, 0.8)
                aa, pu = carry(a0, held_c, base_c, trial)
            except MotionError as error:
                errs.append(str(error)[:60])
                continue
            return trial, aa, pu, r
        return None

    res = plan_lift(m)
    if res is None:
        m = api.motion()
        res = plan_lift(m)
        report["cycle"][-1]["no_recommand"] = res is not None
    if res is None:
        raise MotionError(f"no lift / carry plan: {errs[-1] if errs else ''}")
    trial, a1, plan_used, r = res
    m.adopt(trial)
    report["cycle"][-1]["lift_mm"] = round(1000 * r, 1)
    report["cycle"][-1]["carry_plan"] = plan_used
    report["cycle"][-1]["rim_prior"] = round(rim_prior, 3)
    P_end = {s: m.grasp_pose(s) for s in SIDES}
    m.send(depth_next=True, label=f"carry_{i}")
    if i == 1:
        api.stage_done()
    snapshot(f"carried_{i}")
    pm2 = palms()
    gap_c = float(np.linalg.norm(api.grasp_pose("left")[:3, 3] - api.grasp_pose("right")[:3, 3]))
    report["cycle"][-1]["gap_c_mm"] = round(1000 * gap_c, 1)
    if gap_c < 0.5 * 2.0 * R_MEAS:
        report["cycle"][-1]["empty"] = True
        m = api.motion()
        m.hand("left", "open", rows=6).hand("right", "open", rows=6)
        try:
            move(m, {s: offset(T, dz=0.08) for s, T in P_end.items()}, FAST, 0.6)
        except MotionError:
            pass
        if m.waist_end is not None and abs(float(m.waist_end[4])) > 0.02:
            m.waist(waist_at(0.0), limits=WL)
        m.send(depth_next=True, label=f"empty_{i}")
        return base_c, rim_t
    dets = bowls_now()
    up = [d for d in dets if dist2(d.mid, pm2) < 0.10 and float(np.percentile(d.points[:, 2], 98)) > rim_t + CLEAR + 0.04]
    held_xy = xy(pm2) + held_off
    held_fit = None
    if up:
        fit = rim_centre(max(up, key=lambda d: float(np.percentile(d.points[:, 2], 98))).points, R_MEAS)
        if fit is not None and fit[1] < 0.012 and dist2(fit[0], held_xy) < 0.04:
            held_xy, held_fit = np.asarray(fit[0], dtype=float), round(1000 * fit[1], 1)
    low = [d for d in dets if dist2(d.mid, base_c) < 0.06 and float(np.percentile(d.points[:, 2], 98)) < rim_t + 0.03]
    base_now = base_c
    if low:
        c, rms = centre_of(low[0], near=base_c, tol=0.03)
        if rms is not None:
            base_now = c
    err = base_now - held_xy
    if float(np.linalg.norm(err)) > 0.06:
        err = err * (0.06 / float(np.linalg.norm(err)))
    descent = float(pm2[2]) - (rim_t + DROP + z_hold)
    descent = min(max(descent, 0.0), 0.15)
    report["cycle"][-1].update({"place_err_mm": [round(1000 * float(v), 1) for v in err], "held_fit_mm": held_fit,
                                "descent_mm": round(1000 * descent, 1), "yaw": round(a1, 3)})
    m = api.motion()
    try:
        move(m, P_end, SLOW, 0.6)
    except MotionError as error:
        report.setdefault("recommand_error", []).append(str(error)[:80])
    P = {s: m.grasp_pose(s) for s in SIDES}
    done = None
    for apart in (True, False):
        trial = m.copy()
        try:
            down = {s: offset(T, dx=float(err[0]), dy=float(err[1]), dz=-descent) for s, T in P.items()}
            move(trial, {s: offset(T, dx=float(err[0]), dy=float(err[1])) for s, T in P.items()}, SLOW, 0.5)
            move(trial, down, SLOW, 0.5)
            trial.hold(6)
            if apart:
                for s in SIDES:
                    down[s][:3, 3] = down[s][:3, 3] - down[s][:3, 2] * 0.04
                move(trial, down, Limits(joint_velocity=1.2, joint_acceleration=4.0), 0.5)
                trial.hold(10)
            trial.hand("left", "open", rows=6).hand("right", "open", rows=6)
            move(trial, {s: offset(T, dz=0.10) for s, T in down.items()}, FAST, 0.6)
            if trial.waist_end is not None and abs(float(trial.waist_end[4])) > 0.02:
                trial.waist(waist_at(0.0), limits=WL)
        except MotionError as error:
            report["cycle"][-1]["place_error"] = str(error)[:120]
            continue
        m.adopt(trial)
        done = apart
        break
    if done is None:
        m.hand("left", "open", rows=6).hand("right", "open", rows=6)
    report["cycle"][-1]["apart"] = done
    m.send(depth_next=True, label=f"place_{i}")
    return base_c, rim_t + NEST


bowls = bowls_now()
report["bowls"] = [b.summary() for b in bowls]
if len(bowls) >= 2:
    radii, rims, offs = [], [], []
    for b in bowls:
        for d in api.perceive(b.label, "head", erode=0, max_detections=6):
            if dist2(d.mid, b.mid) < 0.04 and d.points.shape[0] > 60:
                f = rim_centre(d.points)
                if f is not None and f[1] < 0.006 and 0.02 < f[3] < 0.25:
                    radii.append(f[3])
                top_u = float(np.percentile(d.points[:, 2], 99))
                rims.append(top_u - table)
                offs.append(top_u - float(b.top_z))
                break
    R_MEAS = float(np.median(radii)) if radii else 0.5 * float(np.median([max(b.extent[0], b.extent[1]) for b in bowls]))
    RIM = float(np.median(rims)) if rims else float(np.median([float(b.top_z) - table for b in bowls]))
    bias = min(max(float(np.median(offs)), 0.0), 0.03) if offs else 0.0
    NEST = 0.5 * RIM
    report["measured"] = {"radius": round(R_MEAS, 4), "rim_h": round(RIM, 4), "bias": round(bias, 4), "n_fits": len(radii)}
    m = api.motion()
    m.tcp({s: offset(m.grasp_pose(s), dz=0.12) for s in SIDES}, limits=FAST)
    plans = [squeeze_plan(b, m) for b in bowls]
    cost = [2 if p[1] is None else (0 if abs(p[0]) < 0.02 else 1) for p in plans]
    report["squeeze_cost"] = cost
    cen = [centre_of(b)[0] for b in bowls]

    def carry_ok(k, kb):
        if plans[k][1] is None:
            return 0
        t = plans[k][1].copy()
        P0 = {s: t.grasp_pose(s) for s in SIDES}
        try:
            move(t, {s: offset(T, dz=RIM + CLEAR + 0.06) for s, T in P0.items()}, SLOW, 0.8)
            carry(plans[k][0], xy(bowls[k].mid), cen[kb], t)
        except MotionError:
            return 0
        return 1

    feas = [sum(carry_ok(k, kb) for k in range(len(bowls)) if k != kb) for kb in range(len(bowls))]
    report["carry_feasible"] = feas
    report.pop("carry_plan", None)
    kb = max(range(len(bowls)), key=lambda k: (feas[k], cost[k], float(bowls[k].mid[0]), abs(float(bowls[k].mid[1]))))
    base_c = centre_of(bowls[kb])[0]
    rim_t = float(bowls[kb].top_z) + bias
    report["base"] = [round(float(v), 3) for v in base_c]
    order = sorted([k for k in range(len(bowls)) if k != kb], key=lambda k: dist2(bowls[k].mid, base_c))
    todo = [bowls[k] for k in order]
    first_plan = plans[order[0]]
    i = 0
    while todo and i < 4:
        i += 1
        det = todo[0]
        try:
            snapshot(f"start_{i}")
            if i == 1 and first_plan[1] is not None:
                a0, trial = first_plan
                m.adopt(trial)
            else:
                if i > 1:
                    m = api.motion()
                    if api.rows_left() < 650:
                        report["stop"] = f"rows left {api.rows_left()}"
                        break
                a0, trial = squeeze_plan(det, m, base_c, rim_t + CLEAR - table)
                if trial is None:
                    report[f"error_{i}"] = "no squeeze IK"
                    todo = todo[1:] + todo[:1]
                    if i > 2:
                        break
                    continue
                m.adopt(trial)
            base_c, rim_t = cycle(i, det, base_c, rim_t, a0, m, bias)
        except MotionError as error:
            report[f"error_{i}"] = str(error)[:300]
            break
        dets = bowls_now()
        st = find_base(dets, base_c)
        if st is None:
            report[f"lost_base_{i}"] = True
            break
        rest = [d for d in dets if dist2(d.mid, base_c) >= 0.11]
        expected = todo[1:]
        if len(rest) >= len(expected):
            todo = sorted(rest, key=lambda b: dist2(b.mid, base_c))
        else:
            keep = [e for e in expected if all(dist2(e.mid, r.mid) > 0.08 for r in rest)]
            todo = sorted(rest + keep[:len(expected) - len(rest)], key=lambda b: dist2(b.mid, base_c))
        report.setdefault("todo", []).append(len(todo))
try:
    snapshot("end")
except Exception as error:
    report["snap_error"] = str(error)[:120]
report["budget"] = api.budget()["phases"]
api.end_episode()
