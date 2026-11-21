# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

table = api.table_height("head")
info = api.task_info()
side = api.instructed_arm(api.segment_text(0)) or ("right" if "right arm" in str(info.get("prompt", "")).lower() else "left")
sgn = 1.0 if side == "left" else -1.0
report["side"] = side
OPEN_Q = wuji_open()
TILT_Q = np.array([0.33, 0.29, 0.99, 0.11, 0.50, 0.13, 0.01, 0.63, 1.55, -0.19, 1.18, 0.54, 1.55, -0.14, 1.25, 0.55, 1.30, -0.10, 1.53, 0.20])
PAD_I = {"left": np.array([-0.0819, -0.0147, -0.0003]), "right": np.array([-0.0792, 0.0386, 0.0052])}
PAD_C = {"left": np.array([[-0.09678, -0.0269, -0.04604], [0.00557, 0.02281, 0.0964], [0.0113, 0.01113, 0.05138],
                           [0.04833, 0.00466, -0.00363], [0.05047, 0.00308, -0.01159], [0.02499, -0.00175, -0.00496],
                           [-0.00318, -0.00391, -0.01543], [-0.00784, -0.00641, -0.02399], [-0.00781, -0.00633, -0.02365],
                           [-0.00257, -0.00189, -0.00786]]),
         "right": np.array([[-0.09408, 0.05084, -0.04054], [0.00556, -0.02281, 0.09642], [0.01087, -0.01487, 0.05053],
                            [0.04834, -0.00467, -0.00363], [0.05048, -0.00308, -0.01158], [0.02426, -0.00473, -0.00645],
                            [-0.00318, 0.00391, -0.01543], [-0.00784, 0.00641, -0.024], [-0.00781, 0.00633, -0.02366],
                            [-0.00253, 0.00229, -0.00776]])}
R_TILT = {"left": np.column_stack([[-0.8, 0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, -1.0]]),
          "right": np.column_stack([[-0.8, -0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, -1.0]])}
PITCH_UP = np.array([[math.cos(-0.35), 0.0, math.sin(-0.35)], [0.0, 1.0, 0.0], [-math.sin(-0.35), 0.0, math.cos(-0.35)]])
CLOSED_Q = np.array([0.8166, 0.2442, 0.6353, 0.8531, 0.8475, 0.0, 0.7111, 1.5753] + [0.0] * 12)
PINCH_PRE = OPEN_Q + 0.50 * (CLOSED_Q - OPEN_Q)
PINCH_SQ = OPEN_Q + 1.00 * (CLOSED_Q - OPEN_Q)
PINCH_C = {"right": np.array([-0.0379, 0.0495, 0.0115]), "left": np.array([-0.0379, -0.0495, 0.0115])}
PRESS = 0.025
free = "right" if side == "left" else "left"
R_SUP = np.column_stack([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])


def near_robot(P):
    kin = api.episode.kin
    pos = api.episode.positions()
    near = np.zeros(P.shape[0], dtype=bool)
    for sd in ("l", "r"):
        for link, rad in ((f"hand_{sd}_base_link", 0.14), (f"arm_{sd}_link7", 0.10), (f"arm_{sd}_link6", 0.10),
                          (f"arm_{sd}_link5", 0.10), (f"arm_{sd}_link4", 0.10)):
            c = kin.link_pose(link, pos)[:3, 3]
            near |= np.linalg.norm(P - c[None, :], axis=1) < rad
    return near


def measure_shelf(pts):
    out = {}
    reg = pts[(pts[:, 0] > 0.45) & (pts[:, 0] < 1.20) & (np.abs(pts[:, 1]) < 0.40)]
    reg = reg[~near_robot(reg)]
    zz = reg[(reg[:, 2] > 0.80) & (reg[:, 2] < 1.10)]
    if zz.shape[0] < 500:
        return None
    hz, ez = np.histogram(zz[:, 2], bins=np.arange(0.80, 1.1001, 0.005))
    b0 = None
    for i in [int(v) for v in np.argsort(-hz)[:6]]:
        zc = 0.5 * float(ez[i] + ez[i + 1])
        layer = zz[np.abs(zz[:, 2] - zc) <= 0.008]
        yb = np.floor((layer[:, 1] + 0.3) / 0.02).astype(int)
        cover = len(set(int(v) for v in yb if 0 <= v < 30)) / 30.0
        if cover >= 0.6:
            b0 = float(np.median(layer[:, 2]))
            out["cover"] = round(cover, 2)
            break
    if b0 is None:
        return None
    top = reg[np.abs(reg[:, 2] - b0) < 0.012]
    out["E"] = float(np.percentile(top[:, 0], 2))
    strip = reg[(reg[:, 0] > out["E"] + 0.004) & (reg[:, 0] < out["E"] + 0.04) & (np.abs(reg[:, 2] - b0) < 0.015)]
    out["B0"] = b0
    out["B"] = float(np.median(strip[:, 2])) if strip.shape[0] >= 50 else b0
    out["y"] = [float(np.percentile(top[:, 1], 1)), float(np.percentile(top[:, 1], 99))]
    up = reg[(reg[:, 0] > out["E"]) & (reg[:, 0] < out["E"] + 0.30) & (reg[:, 2] > out["B"] + 0.20) & (reg[:, 2] < out["B"] + 0.60)]
    if up.shape[0] >= 300:
        hu, eu = np.histogram(up[:, 2], bins=np.arange(out["B"] + 0.20, out["B"] + 0.6001, 0.01))
        j = int(np.argmax(hu))
        zu = 0.5 * float(eu[j] + eu[j + 1])
        out["U_top"] = float(np.median(up[np.abs(up[:, 2] - zu) <= 0.008][:, 2]))
    return out


def pad_at(sd, mcp, pip):
    b = np.array([1.0, mcp, pip, mcp * mcp, mcp * pip, pip * pip, mcp ** 3, mcp * mcp * pip, mcp * pip * pip, pip ** 3])
    return b @ PAD_C[sd]


def finger_state(sd):
    st = api.robot_state()
    q = np.asarray(st["effector"][sd], dtype=float)
    G = np.asarray(st["grasp"][sd], dtype=float)
    return {"pad": G[:3, 3] + G[:3, :3] @ pad_at(sd, float(q[4]), float(q[6])), "nominal": G[:3, 3] + G[:3, :3] @ PAD_I[sd],
            "mcp": float(q[4]), "pip": float(q[6])}


pts = api.points("head", stride=1)
shelf = measure_shelf(pts)
report["shelf"] = None if shelf is None else {k: ([round(x, 4) for x in v] if isinstance(v, list) else round(v, 4)) for k, v in shelf.items()}
if shelf is None:
    shelf = {}
BOARD = shelf.get("B", float("nan"))
EDGE = shelf.get("E", float("nan"))
CAP = shelf["U_top"] - 0.06 if "U_top" in shelf else BOARD + 0.30
SUP_X = EDGE - 0.11
SUP_Z = BOARD - 0.085
T_LO, T_HI = BOARD + 0.16, CAP - 0.01
box = pts[(pts[:, 0] > EDGE - 0.11) & (pts[:, 0] < EDGE + 0.29) & (np.abs(pts[:, 1]) < 0.16) & (pts[:, 2] > BOARD + 0.01)
          & (pts[:, 2] < CAP + 0.035)] if shelf else pts[:0]
bands = {}
for k in range(-28, 27):
    b = box[np.abs(box[:, 1] - 0.005 * k) < 0.004]
    face = b[(b[:, 2] > BOARD + 0.06) & (b[:, 2] < BOARD + 0.18)]
    if face.shape[0] < 15:
        continue
    fr = float(np.percentile(face[:, 0], 3))
    near = b[(b[:, 0] > fr - 0.005) & (b[:, 0] < fr + 0.06)]
    if near.shape[0] < 15:
        continue
    bands[k] = (fr, float(np.percentile(near[:, 2], 97)))
report["bands"] = {str(k): [round(v[0], 3), round(v[1], 3)] for k, v in sorted(bands.items())}
fronts = [v[0] for v in bands.values() if v[0] > EDGE]
if len(fronts) >= 10:
    F_MED = float(np.median(fronts))
    F_LO, F_HI = max(EDGE, F_MED - 0.04), F_MED + 0.04
else:
    F_MED = float("nan")
    F_LO, F_HI = EDGE, EDGE
report["fronts"] = {"n": len(fronts), "median": None if len(fronts) < 10 else round(F_MED, 4), "window": [round(F_LO, 4), round(F_HI, 4)]}
Y_LO, Y_HI = (shelf["y"][0] + 0.02, shelf["y"][1] - 0.02) if shelf and "y" in shelf else (-1.0, 1.0)


def hanging_near(k):
    lo, hi = (k - 7, k + 14) if sgn > 0 else (k - 14, k + 7)
    return any(j in bands and bands[j][0] < F_LO for j in range(lo, hi + 1))


def slot_at(k):
    if not Y_LO <= 0.005 * k <= Y_HI:
        return None
    five = [bands.get(k + j) for j in (-2, -1, 0, 1, 2)]
    if any(t is None for t in five):
        return None
    frs = [t[0] for t in five]
    tps = [t[1] for t in five]
    if min(frs) < F_LO or max(frs) > F_HI or min(tps) < T_LO or max(tps) > T_HI:
        return None
    if hanging_near(k):
        return None
    nb = max(bands[j][1] for j in range(k - 6, k + 7) if j in bands)
    return {"k": k, "y": 0.005 * k, "front": max(frs), "front_c": frs[2], "top": max(tps), "nb_top": nb,
            "spread": max(tps) - min(tps), "fspread": max(frs) - min(frs)}


runs = []
cur = []
for k in sorted(bands):
    if cur and k == cur[-1] + 1 and abs(bands[k][1] - bands[cur[-1]][1]) < 0.004 and abs(bands[k][0] - bands[cur[-1]][0]) < 0.006:
        cur.append(k)
    else:
        if cur:
            runs.append(cur)
        cur = [k]
if cur:
    runs.append(cur)
tall = []
for run in runs:
    if len(run) < 4:
        continue
    yc = 0.0025 * (run[0] + run[-1])
    kc = int(round(yc / 0.005))
    frs = [bands[k][0] for k in run]
    tps = [bands[k][1] for k in run]
    top = float(np.median(tps))
    fr = float(np.median(frs))
    if abs(yc) > 0.115 or not Y_LO <= yc <= Y_HI or not (F_LO <= fr <= F_HI and T_LO <= top <= T_HI) or hanging_near(kc):
        continue
    nl = bands.get(run[0] - 2) or bands.get(run[0] - 1)
    nr = bands.get(run[-1] + 2) or bands.get(run[-1] + 1)
    taller = min(top - (nl[1] if nl else top), top - (nr[1] if nr else top))
    if taller < 0.005:
        continue
    nb = max(bands[j][1] for j in range(kc - 6, kc + 7) if j in bands)
    tall.append({"k": kc, "y": yc, "front": max(frs), "front_c": fr, "top": max(tps), "nb_top": nb,
                 "spread": max(tps) - min(tps), "fspread": max(frs) - min(frs), "w": 0.005 * len(run), "taller": taller,
                 "score": -min(taller, 0.02) - 0.5 * min(0.005 * len(run), 0.035) + (0.0 if yc * sgn > 0 else 0.005)})
tall.sort(key=lambda s: s["score"])
report["tall_runs"] = [{q: (round(v, 4) if isinstance(v, float) else v) for q, v in s.items()} for s in tall[:3]]
cands = []
for kk in list(range(5, 23)) + list(range(0, -13, -1)):
    s = slot_at(int(round(sgn * kk)))
    if s is not None:
        s["score"] = s["spread"] + 0.5 * s["fspread"] + 0.1 * abs(abs(s["y"]) - 0.06) + (0.01 if kk <= 0 else 0.0)
        cands.append(s)
cands.sort(key=lambda s: s["score"])
slots = []
for s in cands:
    if all(abs(s["y"] - t["y"]) >= 0.02 for t in slots):
        slots.append(s)
if not slots:
    mids = [(kq, v) for kq, v in bands.items() if 0.025 <= sgn * 0.005 * kq < 0.11 and Y_LO <= 0.005 * kq <= Y_HI
            and F_LO < v[0] < F_HI and T_LO < v[1] < T_HI and not hanging_near(kq)]
    if mids:
        kq, v = min(mids, key=lambda t: abs(abs(0.005 * t[0]) - 0.06))
        nbv = [bands[j][1] for j in range(kq - 6, kq + 7) if j in bands]
        slots.append({"k": kq, "y": 0.005 * kq, "front": v[0], "front_c": v[0], "top": v[1],
                      "nb_top": max(nbv), "spread": 0.0, "fspread": 0.0, "score": 9.0})
report["slots"] = [{q: (round(v, 4) if isinstance(v, float) else v) for q, v in s.items()} for s in slots[:3]]

tilted = None
errors = []
tries = []
inside = None
for s in slots[:3]:
    if tilted is not None:
        break
    if inside is not None and api.rows_left() < 1000:
        errors.append(f"rows left {api.rows_left()}: no retry")
        break
    y = s["y"]
    top = s["top"]
    xc = s["front"] + 0.03
    z_over = min(max(top + 0.02, s["nb_top"] + 0.015), CAP)
    for yaw in (0.0, -0.25 * sgn, 0.25 * sgn):
        rot = rotz(yaw)[:3, :3] @ R_TILT[side] @ PITCH_UP
        g0 = pose(np.array([xc, y, top - PRESS]) - rot @ PAD_I[side], rot)
        dz_over = z_over - (top - PRESS)
        m = api.motion()
        try:
            if inside is not None:
                Gc0 = m.grasp_pose(side)
                m.tcp(side, offset(Gc0, dz=inside), linear=True, limits=Limits(linear_velocity=0.10))
                m.tcp(side, offset(Gc0, dx=-0.07, dz=inside), linear=True, limits=Limits(linear_velocity=0.12))
            m.hand(side, TILT_Q, rows=12)
            m.tcp(side, offset(g0, dx=-0.16, dz=dz_over), limits=Limits(joint_velocity=1.6, joint_acceleration=6.0))
            m.tcp(side, offset(g0, dx=-0.035, dz=dz_over), linear=True, limits=Limits(linear_velocity=0.10))
            trial = m.copy()
            trial.tcp(side, offset(g0, dz=dz_over), linear=True, limits=Limits(linear_velocity=0.06))
            trial.tcp(side, g0, linear=True, limits=Limits(linear_velocity=0.03))
            m.hold(12)
        except MotionError as error:
            errors.append(f"{round(y, 3)}/{yaw}: {str(error)[:80]}")
            continue
        m.send(label="over_book")
        f0 = finger_state(side)
        m = api.motion()
        Gc = m.grasp_pose(side)
        e = (Gc[:3, 3] + Gc[:3, :3] @ PAD_I[side]) - f0["nominal"]
        if float(np.linalg.norm(e)) > 0.04:
            e = e * (0.04 / float(np.linalg.norm(e)))
        report["pad_err_mm"] = [round(1000 * float(v), 1) for v in e]
        g0 = offset(g0, dx=float(e[0]), dy=float(e[1]), dz=float(e[2]))
        sup_now = None
        if report.get("support") is None:
            for dy_s in (0.0, 0.015, 0.03):
                ys = y + (dy_s if free == "left" else -dy_s)
                try:
                    trial = m.copy()
                    trial.hand(free, OPEN_Q, rows=8)
                    trial.tcp(free, pose(np.array([SUP_X - 0.10, ys, SUP_Z - 0.03]), R_SUP), limits=Limits(joint_velocity=2.2, joint_acceleration=7.0))
                    trial.tcp(free, pose(np.array([SUP_X, ys, SUP_Z]), R_SUP), linear=True, limits=Limits(linear_velocity=0.15))
                except MotionError as error:
                    report.setdefault("support_errors", []).append(f"{round(ys, 3)}: {str(error)[:60]}")
                    continue
                m.adopt(trial)
                sup_now = [SUP_X, round(ys, 3), SUP_Z]
                break
        try:
            m.tcp(side, offset(g0, dz=dz_over), linear=True, limits=Limits(linear_velocity=0.06))
            if dz_over > PRESS + 0.012:
                m.tcp(side, offset(g0, dz=PRESS + 0.01), linear=True, limits=Limits(linear_velocity=0.06))
            m.tcp(side, g0, linear=True, limits=Limits(linear_velocity=0.03))
            m.hold(10)
        except MotionError as error:
            errors.append(f"{round(y, 3)}/{yaw}: {str(error)[:80]}")
            continue
        m.send(label="press_book")
        if sup_now is not None:
            report["support"] = sup_now
        f1 = finger_state(side)
        m = api.motion()
        Gc = m.grasp_pose(side)
        pad_c = Gc[:3, 3] + Gc[:3, :3] @ PAD_I[side]
        blocked = float(f1["nominal"][2] - pad_c[2]) + float(e[2])
        yielded = (f0["mcp"] - f1["mcp"]) > 0.03 or (f0["pip"] - f1["pip"]) > 0.05
        contact = yielded or blocked > 0.008
        tries.append({"y": round(y, 3), "mcp": [round(f0["mcp"], 3), round(f1["mcp"], 3)], "pip": [round(f0["pip"], 3), round(f1["pip"], 3)],
                      "blocked_mm": round(1000 * blocked, 1), "pad": [round(float(v), 4) for v in f1["pad"]], "contact": bool(contact)})
        if not contact:
            inside = dz_over + 0.01
            break
        piv = np.array([s["front_c"] + 0.008, BOARD])
        d = min(max(xc - float(piv[0]), 0.01), 0.05)
        h = top - BOARD
        ex = float(pad_c[0] - f1["nominal"][0])
        ez = max(float(pad_c[2] - f1["nominal"][2]), float(e[2]) - 0.015)
        extra = max(0.0, float(e[2]) - ez)
        report["press"] = {"pad_meas": [round(float(v), 4) for v in f1["pad"]], "top": round(top, 4), "d": round(d, 4), "h": round(h, 4),
                           "ex_mm": round(1000 * ex, 1), "ez_mm": round(1000 * ez, 1)}
        try:
            for a_deg in (0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 76):
                a = math.radians(a_deg)
                f = 1.0 if a_deg <= 40 else max(0.0, 1.0 - (a_deg - 40) / 36.0)
                rel = np.array([d * math.cos(a) - h * math.sin(a), d * math.sin(a) + h * math.cos(a)])
                nrm = np.array([-math.sin(a), math.cos(a)])
                tgt = piv + rel - (0.008 + 0.022 * f) * nrm
                m.tcp(side, offset(Gc, dx=float(tgt[0] + ex - pad_c[0]), dz=float(tgt[1] + float(e[2]) - extra * f - pad_c[2])),
                      linear=True, limits=Limits(linear_velocity=0.05))
            m.hold(30)
        except MotionError as error:
            errors.append(f"{round(y, 3)}/{yaw}/arc: {str(error)[:80]}")
            inside = dz_over + 0.01
            break
        m.send(depth_next=True, label="tilt_book")
        tilted = {"front": s["front_c"] + 0.008, "top": top, "h": h, "y": y, "yaw": yaw}
        api.stage_done()
        pts2 = api.points("head", stride=1)
        st2 = api.robot_state()
        band = pts2[(np.abs(pts2[:, 1] - y) < 0.012) & (pts2[:, 0] > EDGE - 0.21) & (pts2[:, 0] < float(piv[0]) - 0.03) & (pts2[:, 2] > BOARD + 0.08) & (pts2[:, 2] < BOARD + 0.23)]
        for sd in (side, free):
            band = band[np.linalg.norm(band - np.asarray(st2["grasp"][sd], dtype=float)[:3, 3], axis=1) > 0.10]
        band = band[np.linalg.norm(band - finger_state(side)["pad"], axis=1) > 0.06]
        report["after_arc_pts"] = int(band.shape[0])
        if band.shape[0] >= 30 and report.get("support") is not None:
            k1 = np.linalg.lstsq(np.column_stack([band[:, 0], np.ones(band.shape[0])]), band[:, 2], rcond=None)[0]
            z80 = float(np.percentile(band[:, 2], 80))
            up = band[np.abs(band[:, 2] - z80) < 0.015]
            report["after_arc_fit"] = [round(float(k1[0]), 3), round(z80, 4), int(up.shape[0])]
            if up.shape[0] >= 20 and BOARD + 0.13 <= float(np.median(up[:, 2])) <= BOARD + 0.20:
                z_up = float(np.median(up[:, 2]))
                x_near = float(np.percentile(up[:, 0], 3))
                drag = x_near + 0.5 * h - (EDGE - 0.08)
                report["lying"] = {"z_up": round(z_up, 4), "x_near": round(x_near, 3), "drag": round(drag, 3)}
                if drag > 0.02:
                    xp = x_near + 0.08
                    way = [np.array([min(float(pad_c[0]), x_near - 0.06), z_up + 0.06]), np.array([xp, z_up + 0.04]),
                           np.array([xp, z_up + 0.01]), np.array([xp, z_up - 0.025]), np.array([xp - min(drag, 0.16), z_up - 0.025])]
                    m = api.motion()
                    try:
                        for k, pt in enumerate(way):
                            m.tcp(side, offset(Gc, dx=float(pt[0] + ex - pad_c[0]), dz=float(pt[1] + float(e[2]) - pad_c[2])),
                                  linear=True, limits=Limits(linear_velocity=(0.14, 0.12, 0.10, 0.03, 0.06)[k]))
                        end = way[-1]
                        m.tcp(side, offset(Gc, dx=float(end[0] + ex - pad_c[0]), dz=float(end[1] + 0.045 + float(e[2]) - pad_c[2])),
                              linear=True, limits=Limits(linear_velocity=0.08))
                        Gf = m.grasp_pose(free)
                        m.tcp(free, offset(Gf, dx=-0.07), linear=True, limits=Limits(linear_velocity=0.06))
                        x_re = EDGE - 0.10
                        m.tcp(side, offset(Gc, dx=float(x_re + ex - pad_c[0]), dz=float(end[1] + 0.045 + float(e[2]) - pad_c[2])),
                              linear=True, limits=Limits(linear_velocity=0.10))
                        m.tcp(side, offset(Gc, dx=float(x_re + ex - pad_c[0]), dz=float(end[1] + 0.02 + float(e[2]) - pad_c[2])),
                              linear=True, limits=Limits(linear_velocity=0.08))
                        m.tcp(side, offset(Gc, dx=float(x_re + ex - pad_c[0]), dz=float(end[1] - 0.005 + float(e[2]) - pad_c[2])),
                              linear=True, limits=Limits(linear_velocity=0.04))
                        m.hold(20)
                    except MotionError as error:
                        report["drag_error"] = str(error)[:80]
                        m = None
                    if m is not None:
                        m.send(label="drag_book")
                        report["dragged"] = True
            elif z80 >= BOARD + 0.20:
                a = math.radians(76.0)
                rel = np.array([d * math.cos(a) - h * math.sin(a), d * math.sin(a) + h * math.cos(a)])
                tgt = piv + rel - 0.008 * np.array([-math.sin(a), math.cos(a)]) + 0.02 * np.array([math.cos(a), math.sin(a)])
                m = api.motion()
                try:
                    m.tcp(side, offset(Gc, dx=float(tgt[0] + ex - pad_c[0]), dz=float(tgt[1] + float(e[2]) - pad_c[2])),
                          linear=True, limits=Limits(linear_velocity=0.02))
                    m.hold(20)
                except MotionError as error:
                    report["slide_error"] = str(error)[:80]
                    m = None
                if m is not None:
                    m.send(label="slide_up")
                    report["slid"] = True
        break
report["press_tries"] = tries
report["tilt"] = tilted
report["tilt_errors"] = errors[:4]
report["budget"] = api.budget()["phases"]
api.end_episode()
