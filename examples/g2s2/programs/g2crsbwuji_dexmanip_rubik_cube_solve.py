# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
report["segments"] = info["segments"]
side = "left"
KIN = api.episode.kin
HAND_TABLE = [
    (0.040, [0.5354, -0.1386, 1.5622, 0.2926, 0.8952, 0.3699, 0.1329, 0.6872, 0.6072, 0.2310, 1.4139, 0.6840, 0.8950, 0.3699, 1.2403, 1.5633, 0.7637, 0.3699, 1.5401, 1.5734],
     [0.7656, -0.0422, 1.0164, 0.5583, 1.1614, 0.3699, -0.1030, 1.1775, 0.8068, 0.1530, 1.1134, 0.5348, 0.8950, 0.3699, 1.2403, 1.5633, 0.7637, 0.3699, 1.5401, 1.5734]),
    (0.045, [0.5032, -0.1386, 1.5622, 0.3677, 1.0641, 0.3699, -0.2952, 0.8375, 0.5979, 0.2367, 1.4261, 0.7004, 0.8943, 0.3699, 1.2242, 1.5633, 0.7527, 0.3699, 1.5325, 1.5734],
     [0.7313, -0.0822, 1.0601, 0.5835, 1.2940, 0.3699, -0.4776, 1.2671, 0.7943, 0.1565, 1.1324, 0.5445, 0.8943, 0.3699, 1.2242, 1.5633, 0.7527, 0.3699, 1.5325, 1.5734]),
    (0.050, [0.4723, -0.1386, 1.5622, 0.4397, 1.1272, 0.3699, -0.4356, 0.7593, 0.6917, 0.2151, 1.2405, 1.0529, 0.8937, 0.3699, 1.2076, 1.5633, 0.7415, 0.3699, 1.5246, 1.5734],
     [0.6981, -0.1282, 1.1048, 0.6036, 1.2762, 0.3699, -0.4776, 1.1916, 0.7819, 0.1601, 1.1513, 0.5540, 0.8937, 0.3699, 1.2076, 1.5633, 0.7415, 0.3699, 1.5246, 1.5734]),
    (0.055, [0.4426, -0.1386, 1.5622, 0.5088, 1.1054, 0.3699, -0.3784, 0.6136, 0.7102, 0.2140, 1.2102, 1.1423, 0.8933, 0.3699, 1.1906, 1.5633, 0.7303, 0.3699, 1.5164, 1.5734],
     [0.5307, -0.1386, 1.5622, -0.0081, 1.2609, 0.3699, -0.4776, 1.1105, 0.7727, 0.1633, 1.1613, 0.5775, 0.8933, 0.3699, 1.1906, 1.5633, 0.7303, 0.3699, 1.5164, 1.5734]),
    (0.060, [0.4141, -0.1386, 1.5622, 0.5754, 1.0845, 0.3699, -0.3096, 0.4425, 0.7382, 0.2111, 1.1689, 1.2513, 0.8932, 0.3699, 1.1731, 1.5633, 0.7189, 0.3699, 1.5079, 1.5734],
     [0.4911, -0.1386, 1.5622, 0.0755, 1.2486, 0.3699, -0.4776, 1.0232, 0.7622, 0.1668, 1.1751, 0.5944, 0.8932, 0.3699, 1.1731, 1.5633, 0.7189, 0.3699, 1.5079, 1.5734]),
    (0.065, [0.3867, -0.1386, 1.5622, 0.6397, 1.0642, 0.3699, -0.2202, 0.2261, 0.7140, 0.2192, 1.2121, 1.2193, 0.8932, 0.3699, 1.1551, 1.5633, 0.7075, 0.3699, 1.4990, 1.5734],
     [0.4531, -0.1386, 1.5622, 0.1544, 1.2395, 0.3699, -0.4776, 0.9286, 0.7507, 0.1707, 1.1921, 0.6059, 0.8932, 0.3699, 1.1551, 1.5633, 0.7075, 0.3699, 1.4990, 1.5734]),
    (0.070, [0.3603, -0.1386, 1.5622, 0.7020, 1.0446, 0.3699, -0.0716, -0.1251, 0.7175, 0.2215, 1.2126, 1.2596, 0.8935, 0.3699, 1.1366, 1.5633, 0.6961, 0.3699, 1.4897, 1.5734],
     [0.4166, -0.1386, 1.5622, 0.2290, 1.2256, 0.3699, -0.4595, 0.8214, 0.8070, 0.1639, 1.0408, 0.8787, 0.8935, 0.3699, 1.1366, 1.5633, 0.6961, 0.3699, 1.4897, 1.5734]),
    (0.075, [0.3349, -0.1386, 1.5622, 0.7624, 1.0475, 0.3699, 0.0108, -0.4073, 0.7353, 0.2210, 1.1941, 1.3346, 0.8940, 0.3699, 1.1175, 1.5633, 0.6846, 0.3699, 1.4801, 1.5734],
     [0.3814, -0.1386, 1.5622, 0.2998, 1.2015, 0.3699, -0.4111, 0.6961, 0.8401, 0.1613, 0.9608, 1.0328, 0.8940, 0.3699, 1.1175, 1.5633, 0.6846, 0.3699, 1.4801, 1.5734])]
Q_DEMO = np.array([0.112, -0.633, 0.193, -1.63, -0.126, -0.208, 0.037])
TOOL_G = np.array([-0.005, 0.0, 0.015])
H0 = -math.pi / 2
DEPTH, EXTRA = 0.003, 0.003
MAX_RMS = 0.006
MIN_IOU = 0.80
RELEASE_SHAPE = None


def grasp_rot(h):
    return np.column_stack([[-math.cos(h), -math.sin(h), 0.0], [-math.sin(h), math.cos(h), 0.0], [0.0, 0.0, -1.0]])


def grasp_at(p, h):
    T = np.eye(4)
    T[:3, :3] = grasp_rot(h)
    T[:3, 3] = np.asarray(p, dtype=float) - T[:3, :3] @ TOOL_G
    return T


def ik(T, seed):
    r = KIN.solve(side, T, seed, api.episode.positions(), frame="grasp", restarts=0, max_iterations=600,
                  position_tolerance_m=0.0025, rotation_tolerance_rad=math.radians(8.0), orientation_weight=0.05)
    return np.asarray(r.joints, dtype=float) if r.ok else None


def locate():
    for text in ("rubik's cube", "small colorful cube"):
        f = api.locate_rgb(text, cameras=("head", "hand_right"), max_rms_m=MAX_RMS)
        if f is not None:
            return f
    for text in ("rubik's cube", "small colorful cube"):
        f = api.locate_rgb(text)
        if f is not None:
            return f
    return None


def short(f):
    return None if f is None else {"point": [round(float(v), 4) for v in f["point"]], "rms_m": round(f["rms_m"], 4), "cameras": f["cameras"]}


def hand_shapes(side_len):
    ss = [r[0] for r in HAND_TABLE]
    x = min(max(side_len, ss[0]), ss[-1])
    i = 0
    while i < len(ss) - 2 and x > ss[i + 1]:
        i += 1
    w = (x - ss[i]) / (ss[i + 1] - ss[i])
    return ((1.0 - w) * np.array(HAND_TABLE[i][1]) + w * np.array(HAND_TABLE[i + 1][1]),
            (1.0 - w) * np.array(HAND_TABLE[i][2]) + w * np.array(HAND_TABLE[i + 1][2]))


def hull2d(P):
    pts = sorted([(float(p[0]), float(p[1])) for p in P])
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def inside(poly, U, V):
    ok = np.ones(U.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ok &= ((b[0] - a[0]) * (V - a[1]) - (b[1] - a[1]) * (U - a[0])) >= 0
    return ok


SIGNS = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])


def cube_corners(c, s_len, psi):
    R = np.array([[math.cos(psi), -math.sin(psi), 0.0], [math.sin(psi), math.cos(psi), 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(c, dtype=float)[None, :] + (0.5 * s_len * SIGNS) @ R.T


def cube_mask(cam):
    best = None
    for text in ("rubik's cube", "small colorful cube"):
        for item in api.segment(text, cam):
            area = int(item["mask"].sum())
            if 80 <= area <= 60000 and (best is None or item["score"] > best["score"]):
                best = item
        if best is not None and best["score"] >= 0.3:
            break
    return best


def fit_cube(c0):
    views = []
    for cam, step in (("head", 2), ("hand_right", 3)):
        vw = api.camera(cam)
        item = cube_mask(cam)
        if item is None or vw.model is None or vw.T_base_optical is None:
            continue
        v, u = np.nonzero(item["mask"])
        u0, u1 = max(0, int(u.min()) - 25), min(item["mask"].shape[1] - 1, int(u.max()) + 25)
        v0, v1 = max(0, int(v.min()) - 25), min(item["mask"].shape[0] - 1, int(v.max()) + 25)
        UU, VV = np.meshgrid(np.arange(u0, u1 + 1, step), np.arange(v0, v1 + 1, step))
        views.append((vw, UU.ravel().astype(float), VV.ravel().astype(float), item["mask"][VV.ravel(), UU.ravel()].astype(bool)))
    if not views:
        return None

    def score(c, s_len, psi):
        tot = 0.0
        P = cube_corners(c, s_len, psi)
        for vw, U, V, M in views:
            uv, ok = vw.project_base_points(P)
            if not bool(np.all(np.isfinite(uv))):
                return -1.0
            H = inside(hull2d(uv), U, V)
            tot += float(np.sum(H & M)) / max(float(np.sum(H | M)), 1.0)
        return tot / len(views)

    best = (-1.0, np.asarray(c0, dtype=float), 0.055, 0.0)
    for psi_d in range(-45, 45, 3):
        for s_len in np.arange(0.040, 0.0701, 0.0025):
            val = score(c0, float(s_len), math.radians(psi_d))
            if val > best[0]:
                best = (val, np.asarray(c0, dtype=float), float(s_len), math.radians(psi_d))
    for it in range(2):
        v0, c, s_len, psi = best
        for dx in (-0.004, 0.0, 0.004):
            for dy in (-0.004, 0.0, 0.004):
                for dz in (-0.004, 0.0, 0.004):
                    val = score(c + np.array([dx, dy, dz]), s_len, psi)
                    if val > best[0]:
                        best = (val, c + np.array([dx, dy, dz]), s_len, psi)
        v0, c, s_len, psi = best
        for dpsi in (-2, -1, 1, 2):
            for ds in (-0.002, -0.001, 0.0, 0.001, 0.002):
                val = score(c, s_len + ds, psi + math.radians(dpsi))
                if val > best[0]:
                    best = (val, c, s_len + ds, psi + math.radians(dpsi))
    return {"iou": best[0], "centre": best[1], "side": best[2], "yaw": best[3], "views": len(views)}


Q_HOME = np.array(api.robot_state()["arm"][side], dtype=float)
found = locate()
report["cube"] = short(found)
if found is not None:
    c = np.asarray(found["point"], dtype=float)
    try:
        if found["rms_m"] > MAX_RMS or "hand_left" in found["cameras"]:
            raise MotionError(f"localisation not precise enough for the turn (rms {found['rms_m']:.4f}, {found['cameras']})")
        cube = fit_cube(c)
        report["cube_fit"] = None if cube is None else {"iou": round(cube["iou"], 3), "centre": [round(float(v), 4) for v in cube["centre"]],
                                                         "side_mm": round(1000.0 * cube["side"], 1), "yaw_deg": round(math.degrees(cube["yaw"]), 1),
                                                         "views": cube["views"]}
        if cube is None or cube["iou"] < MIN_IOU:
            raise MotionError("cube fit too poor for the turn")
        c = cube["centre"]
        HQ_OPEN, HQ_CLOSE = hand_shapes(cube["side"])
        Hc = H0 + cube["yaw"]
        P0 = np.array([c[0], c[1], c[2] + cube["side"] / 3.0])
        q_above = ik(grasp_at(P0 + [0, 0, 0.08], Hc), Q_DEMO)
        q_near = ik(grasp_at(P0 + [0, 0, 0.012], Hc), q_above) if q_above is not None else None
        q_press = ik(grasp_at(P0 - [0, 0, DEPTH], Hc), q_near) if q_near is not None else None
        if q_above is None or q_near is None or q_press is None:
            raise MotionError("no approach IK")
        qs, q = [], q_press
        n = 18
        for i in range(1, n + 1):
            T = grasp_at(P0 - [0, 0, DEPTH + EXTRA * i / n], Hc + (math.pi / 2) * i / n)
            qi = ik(T, q)
            if qi is None:
                raise MotionError(f"no twist IK at step {i}")
            if float(np.max(np.abs(qi - q))) > 0.8:
                raise MotionError(f"twist joint jump {float(np.max(np.abs(qi - q))):.2f} at step {i}")
            qs.append(qi)
            q = qi
        h_end = Hc + math.pi / 2
        q_up2 = ik(grasp_at(P0 + [0, 0, 0.02], h_end), q)
        q_clear = ik(grasp_at(P0 + [0, 0.10, 0.14], h_end), q_up2 if q_up2 is not None else q)
        m = api.motion()
        m.hand(side, HQ_OPEN, rows=8)
        m.joints(side, q_above, rows=15)
        m.joints(side, q_near, rows=4)
        m.hold(2)
        m.joints(side, q_press, rows=2)
        m.hand(side, HQ_CLOSE, rows=2)
        m.hold(1)
        for qi in qs:
            m.joints(side, qi, rows=1)
        m.hand(side, HQ_OPEN, rows=3)
        if q_up2 is not None:
            m.joints(side, q_up2, rows=4, gripper=RELEASE_SHAPE)
        if q_clear is not None:
            m.joints(side, q_clear, rows=10)
        m.joints(side, Q_HOME, rows=15)
        report["turn_rows"] = m.rows
        report["turn_q_jump"] = round(float(max(np.max(np.abs(b - a)) for a, b in zip([q_press] + qs[:-1], qs))), 3)
        m.send(label="turn", settle=6)
        report["turn"] = "sent"
    except MotionError as error:
        report["turn"] = str(error)[:160]
    found2 = locate()
    report["cube_after_turn"] = short(found2)
    if found2 is not None:
        found = found2
    c = found["point"]
    report["side"] = side
    grasp = pose((float(c[0]), float(c[1]), float(c[2]) + 0.005), np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]))
    sent = False
    errors = []
    for above, linear in ((0.10, True), (0.07, True), (0.13, True), (0.10, False), (0.06, False)):
        try:
            m = api.motion()
            m.tcp(side, offset(grasp, dz=above), gripper="open", free_axis=(0, 0, 1))
            m.tcp(side, grasp, linear=linear, free_axis=(0, 0, 1))
            m.hand(side, wuji_power(0.75), rows=20)
            m.tcp(side, offset(grasp, dz=0.10), linear=linear, free_axis=(0, 0, 1))
            if m.rows > 250:
                raise MotionError(f"{m.rows} rows > one Rubik reply")
            report["rows"] = m.rows
            report["variant"] = [above, linear]
            m.send(label="pick_cube")
            api.stage_done()
            sent = True
            break
        except MotionError as error:
            errors.append(str(error)[:120])
    if not sent:
        report["pick_errors"] = errors
    else:
        m = api.motion()
        m.hold(30)
        m.send(label="hold_lifted", settle=0)
report["budget"] = api.budget()["phases"]
api.end_episode()
