# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

side = "right"
TOSS = "pop"
VMAX = 9.0
report["segments"] = list(api.task_info().get("segments") or [])
RIM_C = np.array([0.9949, 0.0, 1.050])
RIM_R = 0.16
H_NEAR = np.array([0.800, 0.0, 1.030])
H_END = np.array([0.661, 0.0, 1.039])
GRIP_X = 0.756
PALM_UP = np.column_stack([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def handle_point(x):
    t = (H_NEAR[0] - x) / (H_NEAR[0] - H_END[0])
    return H_NEAR + t * (H_END - H_NEAR)


def best_mask(text, lo, hi):
    best = None
    for item in api.segment(text, "head"):
        area = int(item["mask"].sum())
        if lo <= area <= hi and (best is None or item["score"] > best["score"]):
            best = item
    return best


view = api.camera("head")


def fit_offset():
    if view.model is None or view.T_base_optical is None:
        return 0.0, 0.0, None, {}
    pan = best_mask("frying pan", 2500, 40000)
    handle = best_mask("pan handle", 150, 4000)
    obs = {}
    if pan is not None and pan["score"] >= 0.2:
        v, u = np.nonzero(pan["mask"])
        obs["pan"] = (float(np.percentile(u, 0.5)), float(np.percentile(u, 99.5)), float(np.percentile(v, 0.5)))
    if handle is not None and handle["score"] >= 0.2:
        v, u = np.nonzero(handle["mask"])
        bottom = v >= v.max() - 5
        obs["end"] = (float(np.mean(u[bottom])), float(v.max()))
    if not obs:
        return 0.0, 0.0, None, obs
    grid = np.arange(-0.07, 0.0701, 0.002)
    DX, DY = np.meshgrid(grid, grid, indexing="ij")
    cand = np.column_stack([DX.ravel(), DY.ravel()])
    n = cand.shape[0]
    ang = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    rim = RIM_C + RIM_R * np.column_stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)])
    cost = np.zeros(n)
    if "pan" in obs:
        pts = (rim[None, :, :] - np.column_stack([cand, np.zeros(n)])[:, None, :]).reshape(-1, 3)
        uv, vis = view.project_base_points(pts)
        uv = uv.reshape(n, -1, 2)
        cost += (uv[:, :, 0].min(axis=1) - obs["pan"][0]) ** 2 + (uv[:, :, 0].max(axis=1) - obs["pan"][1]) ** 2
        cost += (uv[:, :, 1].min(axis=1) - obs["pan"][2]) ** 2
    if "end" in obs:
        pts = H_END[None, :] - np.column_stack([cand, np.zeros(n)])
        uv, vis = view.project_base_points(pts)
        cost += 2.0 * ((uv[:, 0] - obs["end"][0]) ** 2 + (uv[:, 1] - obs["end"][1]) ** 2)
    k = int(np.argmin(cost))
    terms = (3 if "pan" in obs else 0) + (4 if "end" in obs else 0)
    fit = {"dx": round(float(cand[k, 0]), 4), "dy": round(float(cand[k, 1]), 4), "rms_px": round(float(math.sqrt(cost[k] / max(1, terms))), 2)}
    if fit["rms_px"] <= 8.0 and abs(cand[k, 0]) < 0.069 and abs(cand[k, 1]) < 0.069:
        return float(cand[k, 0]), float(cand[k, 1]), fit, obs
    return 0.0, 0.0, fit, obs


def rim_rise_px(shift, lift):
    ang = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    rim = RIM_C - shift + RIM_R * np.column_stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)])
    uv0, vis0 = view.project_base_points(rim)
    uv1, vis1 = view.project_base_points(rim + np.array([0.0, 0.0, lift]))
    return float(uv0[:, 1].min() - uv1[:, 1].min()), float(uv1[:, 1].min())


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


VARIANTS = [(gx, rise, roll, so, yaw) for gx in (GRIP_X, 0.735, 0.715) for yaw in (-0.40, 0.0) for rise, roll, so in
            ((0.029, 0.0, 0.14), (0.026, -0.15, 0.13), (0.029, 0.15, 0.14))]
VARIANTS = [(0.735, 0.029, 0.0, 0.18, -0.40), (0.725, 0.029, 0.0, 0.18, -0.40)] + VARIANTS
APP_DX = -0.05
BELOW_DZ = -0.028
SLIDE_ROWS = 32
GRIP_Q = np.array([0.23, 0.06, 1.0, 0.09, 1.56, 0.22, 1.06, 0.99, 1.55, 0.07, 1.29, 1.51, 1.56, 0.05, 1.3, 1.53, 1.56, 0.21, 1.55, 1.18])
APPROACH = wuji_open()
APPROACH[0] = 0.60
Q_GOOD = np.array([-0.601, -1.546, 1.232, -1.263, 2.09, 0.803, -0.402])
APPROACH_1 = "side"
KIN = api.episode.kin


def plan_grasp(m, shift, tag, skip=(), hold=40, max_rows=None, approach="side"):
    errors = []
    for gx, rise, roll, side_off, yaw in VARIANTS:
        if (gx, rise, roll, side_off, yaw) in skip:
            continue
        hp = handle_point(gx) - shift
        rot = rotz(yaw)[:3, :3] @ PALM_UP if yaw else PALM_UP
        rot = rot @ roty(roll)[:3, :3] if roll else rot
        grasp = pose(hp + np.array([0.0, 0.0, rise]), rot)
        below = offset(grasp, dz=BELOW_DZ)
        if approach == "side":
            pre = offset(below, dx=APP_DX if side_off > 0.15 else 0.0, dy=-side_off)
        elif approach == "front":
            pre = offset(below, dx=-0.10)
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
            q_grasp = ik(grasp, Q_GOOD)
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
            errors.append(f"{gx}/{rise}/{roll}/{yaw}: {str(error)[:80]}")
            continue
        if max_rows is not None and trial.rows > max_rows:
            errors.append(f"{gx}/{rise}/{roll}/{yaw}: {trial.rows} rows > {max_rows} left")
            continue
        m.adopt(trial)
        report[f"variant_{tag}"] = [gx, rise, roll, side_off, yaw]
        report[f"grip_point_{tag}"] = [round(float(v), 4) for v in hp]
        return {"grasp": grasp, "hp": hp, "variant": (gx, rise, roll, side_off, yaw)}
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
dx, dy, fit, obs = fit_offset()
report["obs"] = {k: [round(x, 1) for x in val] for k, val in obs.items()}
report["fit"] = fit
report["offset"] = [round(dx, 4), round(dy, 4)]
top0 = obs["pan"][2] if "pan" in obs else None
width0 = (obs["pan"][1] - obs["pan"][0]) if "pan" in obs else None
lifted = False
grip = None
used = []
for attempt in (1, 2):
    m = api.motion()
    if attempt == 2:
        m.hand(side, APPROACH, rows=8)
        m.tcp(side, offset(api.grasp_pose(side), dz=-0.04, dy=-0.06), linear=True, limits=Limits(linear_velocity=0.3))
        m.send(label="release_retry", settle=4)
        ndx, ndy, nfit, nobs = fit_offset()
        report["fit_retry"] = nfit
        if nfit is not None and nfit["rms_px"] <= 8.0:
            dx, dy = ndx, ndy
        if "pan" in nobs:
            top0, width0 = nobs["pan"][2], nobs["pan"][1] - nobs["pan"][0]
        m = api.motion()
    blocked = attempt == 2 and float(np.linalg.norm(report.get("reach_err_mm_1") or [0.0])) > 30.0
    way = APPROACH_1 if attempt == 1 else ("front" if blocked else APPROACH_1)
    report[f"approach_{attempt}"] = way
    grip = plan_grasp(m, np.array([dx, dy, 0.0]), attempt, skip=used if not blocked else (), hold=40 if attempt == 1 else 26,
                      max_rows=None if attempt == 1 else api.rows_left() - 14, approach=way)
    if grip is None:
        break
    used.append(grip["variant"])
    report[f"rows_lift_{attempt}"] = m.rows
    m.send(label=f"grasp_lift_hold_{attempt}")
    want = grip["grasp"][:3, 3] + np.array([0.0, 0.0, 0.10])
    got = api.grasp_pose(side)[:3, 3]
    report[f"reach_err_mm_{attempt}"] = [round(1000.0 * float(v), 1) for v in (got - want)]
    report[f"arm_q_{attempt}"] = [round(float(v), 3) for v in api.robot_state()["arm"][side]]
    if top0 is None or view.model is None:
        lifted = None
        break
    expect, v_top = rim_rise_px(np.array([dx, dy, 0.0]), 0.10)
    top1 = pan_top_row(width0)
    rise_px = None if top1 is None else top0 - top1
    report[f"rim_rise_px_{attempt}"] = [None if rise_px is None else round(rise_px, 1), round(expect, 1), round(v_top, 1)]
    if rise_px is None:
        lifted = None
        break
    if rise_px >= max(6.0, 0.30 * expect):
        lifted = True
        break
    if api.rows_left() < 260:
        break
report["grip_point"] = None if grip is None else [round(float(v), 4) for v in grip["hp"]]
report["lifted_visual"] = lifted


def milestone(name):
    for item in api.episode.request.task_progress or ():
        if str(item.get("class_name")) == name:
            prog = item.get("progress") or {}
            return str(prog.get("STATUS")) == "SUCCESS" or int(prog.get("SCORE") or 0) == 1
    return None


def smoother(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)


def toss_sample(u, g_pitch, t_pitch, t_lift, lunge):
    ug, ut = 0.7 / 1.45, 0.4 / 1.45
    if u < ug:
        s = smoother(u / ug)
        return 0.0, 0.012 * s, -g_pitch * s
    if u < ug + ut:
        p = (u - ug) / ut
        if p < 0.40:
            s = smoother(p / 0.40)
            dx, dz, pitch = -0.010 * s, 0.012 - 0.010 * s, -g_pitch
        elif p < 0.70:
            s = smoother((p - 0.40) / 0.30)
            dx, dz = -0.010 + 0.025 * s, 0.002 + (t_lift - 0.002) * s
            pitch = -g_pitch + (g_pitch + t_pitch) * s
        else:
            s = smoother((p - 0.70) / 0.30)
            dx, dz, pitch = 0.015 * (1.0 - s), t_lift * (1.0 - s), t_pitch * (1.0 - s)
        lunge_s = 0.0 if p < 0.40 else (smoother((p - 0.40) / 0.30) if p < 0.70 else 1.0 - smoother((p - 0.70) / 0.30))
        return dx + lunge * lunge_s, dz, pitch
    return 0.0, 0.0, 0.0


if grip is not None and lifted is not False and api.rows_left() > 240:
    report["stable_lift_seen"] = milestone("WokStableLift")
    if report["stable_lift_seen"] is not False:
        api.stage_done()
        kin = api.episode.kin
        g_now = api.grasp_pose(side)
        lifted_by = float(g_now[2, 3] - grip["grasp"][2, 3])
        pivot = np.array([RIM_C[0] - dx, RIM_C[1] - dy, RIM_C[2] - 0.09 + lifted_by])
        d0 = g_now[:3, 3] - pivot
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
        for scale, lunge in (((1.2, -0.04), (1.2, -0.04)) if TOSS == "chef" else ()):
            n = int(round(1.45 * scale / api.episode.row_dt))
            for k in range(1, n + 1):
                ddx, ddz, pitch = toss_sample(k / n, 0.24, 0.30, 0.075, lunge)
                Rp = roty(-pitch)[:3, :3]
                T = np.eye(4)
                T[:3, :3] = Rp @ g_now[:3, :3]
                T[:3, 3] = pivot + np.array([ddx, 0.0, ddz]) + Rp @ d0
                res = kin.solve(side, T, q, positions, frame="grasp", restarts=0, max_iterations=60,
                                position_tolerance_m=0.004, rotation_tolerance_rad=math.radians(2.0))
                if res.ok:
                    q = np.asarray(res.joints, dtype=float)
                else:
                    fails += 1
                qs.append(q)
        prev = np.array(api.robot_state()["arm"][side], dtype=float)
        stretched = 0
        for qk in qs:
            steps = max(1, int(math.ceil(float(np.max(np.abs(qk - prev))) / (VMAX * api.episode.row_dt))))
            stretched += steps - 1
            for j in range(1, steps + 1):
                m.joints(side, prev + (qk - prev) * (j / steps), rows=1)
            prev = qk
        m.hold(25)
        report["toss"] = {"mode": TOSS, "rows": m.rows, "ik_fails": fails, "stretched_rows": stretched, "pivot": [round(float(v), 3) for v in pivot]}
        if fails < 10:
            m.send(label="toss")
            report["food_toss_seen"] = milestone("WokFoodToss")
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
