# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

side = "right"
TOSS = "pop"
VMAX = 9.0
report["segments"] = list(api.task_info().get("segments") or [])
PALM_UP = np.column_stack([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def best_mask(text, lo, hi, cam="head"):
    best = None
    for item in api.segment(text, cam):
        area = int(item["mask"].sum())
        if lo <= area <= hi and (best is None or item["score"] > best["score"]):
            best = item
    return best


view = api.camera("head")


HANDLE_TEXT = ("pan handle", "wok handle", "handle")
MASK_AREA = {"head": (150, 4000), "hand_right": (300, 40000), "hand_left": (300, 40000)}


def view_plane(cam, item):
    v, u = np.nonzero(item["mask"])
    if u.size < 40:
        return None
    vw = api.camera(cam)
    r = vw.model.pixel_rays(u.astype(float), v.astype(float))
    r = r[np.all(np.isfinite(r), axis=1)]
    xy = r[:, :2]
    c = np.median(xy, axis=0)
    Uu, S, Vt = np.linalg.svd(xy - c, full_matrices=False)
    d = Vt[0]
    t = (xy - c) @ d
    a = np.array([c[0] + d[0] * float(np.percentile(t, 2)), c[1] + d[1] * float(np.percentile(t, 2)), 1.0])
    b = np.array([c[0] + d[0] * float(np.percentile(t, 98)), c[1] + d[1] * float(np.percentile(t, 98)), 1.0])
    R = vw.T_base_optical[:3, :3]
    n = R @ np.cross(a, b)
    n = n / np.linalg.norm(n)
    rays = (r / np.linalg.norm(r, axis=1, keepdims=True)) @ R.T
    return {"n": n, "o": vw.T_base_optical[:3, 3].copy(), "rays": rays, "elong": float(S[0] / max(float(S[1]), 1e-9)),
            "w": float(item["score"]), "xy": xy, "c": c, "d": d, "R": R}


def fit_axis(planes):
    keys = sorted(planes)
    N = np.array([planes[k]["n"] for k in keys])
    W = np.array([planes[k]["w"] for k in keys])
    M = (N * W[:, None]).T @ N
    evals, evecs = np.linalg.eigh(M)
    L = evecs[:, 0]
    if L[0] < 0:
        L = -L
    A = np.vstack([N * W[:, None], L[None, :]])
    rhs = np.concatenate([np.array([float(planes[k]["n"] @ planes[k]["o"]) for k in keys]) * W, [0.0]])
    X0 = np.linalg.lstsq(A, rhs, rcond=None)[0]
    res = {k: float(planes[k]["n"] @ (X0 - planes[k]["o"])) for k in keys}
    return X0, L, res, float(evals[1])


def ray_params(X0, L, o, rays):
    w0 = X0 - o
    bb = rays @ L
    dd = float(L @ w0)
    e = rays @ w0
    den = np.maximum(1.0 - bb * bb, 1e-9)
    sv = (bb * e - dd) / den
    tv = (e - bb * dd) / den
    gap = np.linalg.norm((X0[None, :] + sv[:, None] * L[None, :]) - (o[None, :] + tv[:, None] * rays), axis=1)
    return sv, gap


def measure_handle():
    head = api.camera("head")
    if head.model is None or head.T_base_optical is None:
        return None
    planes = {}
    for cam in ("head", "hand_right", "hand_left"):
        vw = api.camera(cam)
        if vw.model is None or vw.T_base_optical is None:
            continue
        lo, hi = MASK_AREA[cam]
        item = None
        for text in HANDLE_TEXT:
            item = best_mask(text, lo, hi, cam)
            if item is not None and item["score"] >= 0.2:
                break
        if item is None or item["score"] < 0.2:
            continue
        p = view_plane(cam, item)
        if p is not None and p["elong"] >= 3.0:
            planes[cam] = p
    out = {"views": sorted(planes)}
    if "head" not in planes or len(planes) < 2:
        return out
    X0, L, res, ev = fit_axis(planes)
    if len(planes) > 2:
        worst = max(res, key=lambda k: abs(res[k]))
        if abs(res[worst]) > 0.012 and worst != "head":
            out["dropped"] = [worst, round(1000.0 * res[worst], 1)]
            planes.pop(worst)
            X0, L, res, ev = fit_axis(planes)
    out["plane_res_mm"] = {k: round(1000.0 * v, 1) for k, v in res.items()}
    out["cond"] = round(ev, 3)
    out["views"] = sorted(planes)
    if float(abs(L[0])) < 0.5:
        out["error"] = "axis not toward the robot"
        return out
    if ev < 0.02:
        out["error"] = "views too alike"
        return out
    sv, gap = ray_params(X0, L, planes["head"]["o"], planes["head"]["rays"])
    keep = gap < 0.03
    if int(keep.sum()) < 30:
        out["error"] = "head rays off the axis"
        return out
    s_tip, s_root = float(np.percentile(sv[keep], 0.5)), float(np.percentile(sv[keep], 98))
    tips = [s_tip]
    for k in planes:
        if k != "head":
            s2, g2 = ray_params(X0, L, planes[k]["o"], planes[k]["rays"])
            if int((g2 < 0.03).sum()) >= 30:
                tips.append(float(np.percentile(s2[g2 < 0.03], 0.5)))
    s_tip = float(np.median(tips))
    tip = X0 + s_tip * L
    ph = planes["head"]
    tt = (ph["xy"] - ph["c"]) @ ph["d"]
    ww = (ph["xy"] - ph["c"]) @ np.array([-ph["d"][1], ph["d"][0]])
    edges = np.linspace(float(np.percentile(tt, 5)), float(np.percentile(tt, 95)), 11)
    spans = []
    for i in range(10):
        sel = (tt >= edges[i]) & (tt < edges[i + 1])
        if int(sel.sum()) >= 8:
            spans.append(float(np.percentile(ww[sel], 97) - np.percentile(ww[sel], 3)))
    mid = X0 + 0.5 * (s_tip + s_root) * L
    depth = float(ph["R"][:, 2] @ (mid - ph["o"]))
    radius = 0.5 * float(np.median(spans)) * depth if spans else float(np.percentile(gap[keep], 95))
    out["radius_gap"] = float(np.percentile(gap[keep], 95))
    radius = float(np.clip(radius, 0.004, 0.03))
    out.update({"tip": tip, "dir": L, "free_len": s_root - s_tip, "radius": radius})
    pan = best_mask("frying pan", 2500, 40000)
    if pan is not None and pan["score"] >= 0.2:
        v, u = np.nonzero(pan["mask"])
        v_top = float(np.percentile(v, 0.5))
        u_top = float(np.median(u[v <= v_top + 3]))
        out["pan"] = (float(np.percentile(u, 0.5)), float(np.percentile(u, 99.5)), v_top)
        ray = head.model.pixel_rays(np.array([u_top]), np.array([v_top]))[0]
        ray = head.T_base_optical[:3, :3] @ (ray / np.linalg.norm(ray))
        o = head.T_base_optical[:3, 3]
        z_h = float((X0 + s_root * L)[2])
        if ray[2] < -1e-3:
            out["top_point"] = o + ray * ((z_h - o[2]) / ray[2])
    return out


def top_rise_px(top_point, lift):
    uv0, vis0 = view.project_base_points(np.array([top_point]))
    uv1, vis1 = view.project_base_points(np.array([top_point + np.array([0.0, 0.0, lift])]))
    return float(uv0[0, 1] - uv1[0, 1]), float(uv1[0, 1])


def pan_top_row(width0):
    pan = best_mask("frying pan", 2500, 60000)
    if pan is None or pan["score"] < 0.2:
        return None
    v, u = np.nonzero(pan["mask"])
    width = float(np.percentile(u, 99.5) - np.percentile(u, 0.5))
    report.setdefault("pan_width_px", []).append(round(width, 1))
    if width0 is not None and abs(width - width0) > 0.25 * width0:
        return None
    return float(np.percentile(v, 0.5))


PALM_UP_OFF = (0.018, 0.015)
VARIANTS = [(d, rise, roll, so, yaw) for d in (0.095, 0.074, 0.054) for yaw in (-0.40, 0.0) for rise, roll, so in
            ((PALM_UP_OFF[0], 0.0, 0.14), (PALM_UP_OFF[1], -0.15, 0.13), (PALM_UP_OFF[0], 0.15, 0.14))]
VARIANTS = [(0.074, PALM_UP_OFF[0], 0.0, 0.18, -0.40), (0.064, PALM_UP_OFF[0], 0.0, 0.18, -0.40)] + VARIANTS
APP_DX = -0.05
BELOW_DZ = -0.028
SLIDE_ROWS = 32
GRIP_Q = np.array([0.23, 0.06, 1.0, 0.09, 1.56, 0.22, 1.06, 0.99, 1.55, 0.07, 1.29, 1.51, 1.56, 0.05, 1.3, 1.53, 1.56, 0.21, 1.55, 1.18])
APPROACH = wuji_open()
APPROACH[0] = 0.60
Q_SEED = np.array([-0.922, -1.663, 1.308, -1.448, 1.553, 0.822, -0.135])
APPROACH_1 = "side"
KIN = api.episode.kin


def plan_grasp(m, hd, tag, skip=(), hold=100, max_rows=None, approach="side"):
    errors = []
    yaw_h = math.atan2(float(hd["dir"][1]), float(hd["dir"][0]))
    for gx, rise0, roll, side_off, yaw in VARIANTS:
        if (gx, rise0, roll, side_off, yaw) in skip:
            continue
        hp = hd["tip"] + gx * hd["dir"]
        rise = hd["radius"] + rise0
        rot = rotz(yaw_h + yaw)[:3, :3] @ PALM_UP
        rot = rot @ roty(roll)[:3, :3] if roll else rot
        grasp = pose(hp + np.array([0.0, 0.0, rise]), rot)
        below = offset(grasp, dz=BELOW_DZ)
        if approach == "side":
            pre = offset(below, dx=APP_DX if side_off > 0.15 else 0.0, dy=-side_off)
        else:
            pre = offset(below, dz=-0.07)
        trial = m.copy()
        try:
            trial.hand(side, APPROACH, rows=10)
            pos_now = api.episode.positions()
            def ik(T, seed):
                r = KIN.solve(side, T, seed, pos_now, frame="grasp", restarts=0, max_iterations=200,
                              position_tolerance_m=0.002, rotation_tolerance_rad=math.radians(1.0))
                return np.asarray(r.joints, dtype=float) if r.ok else None
            q_grasp = ik(grasp, Q_SEED)
            q_below = ik(below, q_grasp) if q_grasp is not None else None
            q_pre = ik(pre, q_below) if q_below is not None else None
            q_lift = ik(offset(grasp, dz=0.10), q_grasp) if q_grasp is not None else None
            q_high = ik(offset(pre, dz=0.15), q_pre) if q_pre is not None else None
            chain = all(v is not None for v in (q_grasp, q_below, q_pre, q_lift, q_high)) and \
                max(float(np.max(np.abs(a - b))) for a, b in ((q_high, q_pre), (q_pre, q_below), (q_below, q_grasp), (q_grasp, q_lift))) < 0.8
            report[f"branch_{tag}"] = None if q_grasp is None else [round(float(v), 2) for v in q_grasp]
            report[f"chain_{tag}"] = bool(chain)
            def straight(q0, Ta, Tb, n):
                rows_q, q = [], q0
                for k in range(1, n + 1):
                    u = k / n
                    u = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
                    T = Ta.copy()
                    T[:3, 3] = Ta[:3, 3] + u * (Tb[:3, 3] - Ta[:3, 3])
                    r = KIN.solve(side, T, q, pos_now, frame="grasp", restarts=0, max_iterations=80,
                                  position_tolerance_m=0.002, rotation_tolerance_rad=math.radians(1.0))
                    if not r.ok:
                        return None
                    qn = np.asarray(r.joints, dtype=float)
                    if float(np.max(np.abs(qn - q))) > 0.15:
                        return None
                    rows_q.append(qn)
                    q = qn
                return rows_q
            lines = None
            if chain:
                l1 = straight(q_high, offset(pre, dz=0.15), pre, 24)
                l2 = straight(l1[-1], pre, below, SLIDE_ROWS if side_off > 0.15 else 26) if l1 else None
                l3 = straight(l2[-1], below, grasp, 16) if l2 else None
                lines = (l1 + l2 + l3) if l3 else None
            report[f"straight_{tag}"] = lines is not None
            if chain and lines is not None:
                trial.joints(side, q_high, limits=Limits(joint_velocity=2.2, joint_acceleration=8.0))
                for qk in lines:
                    trial.joints(side, qk, rows=1)
                q_grasp = lines[-1]
                q_lift = ik(offset(grasp, dz=0.10), q_grasp)
                if q_lift is None:
                    raise MotionError("no lift IK from the straight-line grasp")
            if chain and lines is None:
                trial.joints(side, q_high, limits=Limits(joint_velocity=2.2, joint_acceleration=8.0))
                trial.joints(side, q_pre, rows=24)
                trial.joints(side, q_below, rows=26)
                trial.joints(side, q_grasp, rows=16)
            if chain:
                trial.hand(side, GRIP_Q, rows=25)
                trial.hold(6)
                trial.joints(side, q_lift, rows=34)
                trial.hold(hold)
            else:
                trial.tcp(side, pre, limits=Limits(joint_velocity=2.2, joint_acceleration=8.0))
                trial.tcp(side, below, linear=True, limits=Limits(linear_velocity=0.30))
                trial.tcp(side, grasp, linear=True, limits=Limits(linear_velocity=0.10))
                trial.hand(side, GRIP_Q, rows=25)
                trial.hold(6)
                trial.tcp(side, offset(grasp, dz=0.10), linear=True, limits=Limits(linear_velocity=0.16))
                trial.hold(hold)
        except MotionError as error:
            errors.append(f"{gx}/{rise0}/{roll}/{yaw}: {str(error)[:80]}")
            continue
        if max_rows is not None and trial.rows > max_rows:
            errors.append(f"{gx}/{rise0}/{roll}/{yaw}: {trial.rows} rows > {max_rows} left")
            continue
        m.adopt(trial)
        report[f"variant_{tag}"] = [gx, round(rise, 4), roll, side_off, yaw]
        report[f"grip_point_{tag}"] = [round(float(v), 4) for v in hp]
        return {"grasp": grasp, "hp": hp, "variant": (gx, rise0, roll, side_off, yaw)}
    report[f"plan_errors_{tag}"] = errors[:6]
    return None


stage = api.motion()
stage.hand(side, APPROACH, rows=12)
try:
    stage.tcp(side, offset(api.grasp_pose(side), dz=0.10), linear=True, limits=Limits(linear_velocity=0.25))
    staged = "rise"
except MotionError:
    staged = "hand_only"
report["staged"] = staged
stage.send(label="stage_hand")


def handle_report(hd):
    out = {}
    for k, val in (hd or {}).items():
        if isinstance(val, np.ndarray):
            out[k] = [round(float(x), 4) for x in val]
        elif isinstance(val, tuple):
            out[k] = [round(float(x), 1) for x in val]
        elif isinstance(val, float):
            out[k] = round(val, 4)
        else:
            out[k] = val
    return out


hd = measure_handle()
report["handle_1"] = handle_report(hd)
if hd is None or "tip" not in hd:
    api.motion().hold(2).send(label="look_again")
    hd = measure_handle()
    report["handle_1b"] = handle_report(hd)
if hd is not None and "tip" not in hd:
    hd = None
top0 = hd["pan"][2] if hd is not None and "pan" in hd else None
width0 = (hd["pan"][1] - hd["pan"][0]) if hd is not None and "pan" in hd else None
lifted = False
grip = None
used = []
for attempt in (1,):
    m = api.motion()
    way = APPROACH_1
    report[f"approach_{attempt}"] = way
    if hd is None:
        break
    grip = plan_grasp(m, hd, attempt, skip=used, hold=100, approach=way)
    if grip is None:
        break
    used.append(grip["variant"])
    report[f"rows_lift_{attempt}"] = m.rows
    m.send(label=f"grasp_lift_hold_{attempt}")
    want = grip["grasp"][:3, 3] + np.array([0.0, 0.0, 0.10])
    got = api.grasp_pose(side)[:3, 3]
    report[f"reach_err_mm_{attempt}"] = [round(1000.0 * float(v), 1) for v in (got - want)]
    report[f"arm_q_{attempt}"] = [round(float(v), 3) for v in api.robot_state()["arm"][side]]
    if top0 is None or view.model is None or "top_point" not in hd:
        lifted = None
        break
    expect, v_top = top_rise_px(hd["top_point"], 0.10)
    top1 = pan_top_row(width0)
    rise_px = None if top1 is None else top0 - top1
    report[f"rim_rise_px_{attempt}"] = [None if rise_px is None else round(rise_px, 1), round(expect, 1), round(v_top, 1)]
    if rise_px is None:
        lifted = None
        break
    if rise_px >= max(6.0, 0.30 * expect):
        lifted = True
        break
report["grip_point"] = None if grip is None else [round(float(v), 4) for v in grip["hp"]]
report["lifted_visual"] = lifted


def smoother(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)


TOSS_NEED = 175
if grip is not None and lifted is not False and api.rows_left() > TOSS_NEED:
    if True:
        api.stage_done()
        kin = api.episode.kin
        g_now = api.grasp_pose(side)
        q = np.array(api.robot_state()["arm"][side], dtype=float)
        positions = api.episode.positions()
        m = api.motion()
        fails = 0
        qs = []
        if TOSS == "pop":
            knots = [(0.0, 0), (-0.05, 12), (0.15, 8), (0.03, 8), (0.03, 6), (0.0, 10)]
            z0, base = 0.0, g_now[:3, 3].copy()
            for z1, n in knots[1:]:
                for k in range(1, n + 1):
                    zz = z0 + (z1 - z0) * smoother(k / n)
                    T = g_now.copy()
                    T[:3, 3] = base + np.array([0.0, 0.0, zz])
                    res = kin.solve(side, T, q, positions, frame="grasp", restarts=0, max_iterations=60,
                                    position_tolerance_m=0.004, rotation_tolerance_rad=math.radians(2.0))
                    if res.ok:
                        q = np.asarray(res.joints, dtype=float)
                    else:
                        fails += 1
                    qs.append(q)
                z0 = z1
        prev = np.array(api.robot_state()["arm"][side], dtype=float)
        stretched = 0
        for qk in qs:
            steps = max(1, int(math.ceil(float(np.max(np.abs(qk - prev))) / (VMAX * api.episode.row_dt))))
            stretched += steps - 1
            for j in range(1, steps + 1):
                m.joints(side, prev + (qk - prev) * (j / steps), rows=1)
            prev = qk
        m.hold(25)
        report["toss"] = {"mode": TOSS, "rows": m.rows, "ik_fails": fails, "stretched_rows": stretched}
        if fails < 10:
            m.send(label="toss")
if grip is not None and api.rows_left() > 70:
    try:
        m = api.motion()
        here = api.grasp_pose(side)
        m.tcp(side, pose(np.array([here[0, 3], here[1, 3], float(grip["grasp"][2, 3]) + 0.008]), grip["grasp"][:3, :3]),
              limits=Limits(joint_velocity=1.2, joint_acceleration=4.0))
        m.hand(side, APPROACH, rows=10)
        m.tcp(side, offset(grip["grasp"], dz=-0.03, dy=-0.06), linear=True, limits=Limits(linear_velocity=0.2))
        m.send(label="put_back", settle=4)
        report["put_back"] = True
    except MotionError as error:
        report["put_back"] = str(error)[:80]
report["budget"] = api.budget()["phases"]
api.end_episode()
