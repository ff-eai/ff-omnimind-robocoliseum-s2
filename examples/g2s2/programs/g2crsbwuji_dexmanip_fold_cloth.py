# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

view = api.camera("head")
OPEN_Q = wuji_open()
report["segments"] = list(api.task_info().get("segments") or [])
GRIP = "mixed"


def unit_rot(rows):
    u, s, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    m = u @ vt
    if np.linalg.det(m) < 0:
        m = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return m


ROT = {
    "near_L": unit_rot([[-0.7622, 0.6262, -0.1642], [0.5814, 0.7737, 0.2517], [0.2847, 0.0964, -0.9538]]),
    "near_R": unit_rot([[-0.3974, -0.9085, 0.1289], [-0.8849, 0.3423, -0.3159], [0.2429, -0.2396, -0.94]]),
    "far_L": unit_rot([[-0.8687, 0.4595, -0.185], [0.4136, 0.8784, 0.2395], [0.2726, 0.1315, -0.9531]]),
    "far_R": unit_rot([[-0.7306, -0.6826, -0.015], [-0.6573, 0.7091, -0.2554], [0.185, -0.1767, -0.9667]]),
    "hem_L": unit_rot([[-0.8194, 0.5324, -0.2124], [0.4432, 0.8234, 0.3544], [0.3636, 0.1963, -0.9107]]),
}
QC = {
    "near_L": [0.733, 0.074, 1.0, 0.147, 1.067, -0.131, 0.0, 0.569, 1.152, 0.066, 0.015, 0.569, 1.05, 0.112, 0.102, 0.748, 0.494, 0.239, 0.578, 0.676],
    "near_R": [0.525, 0.025, 0.665, 0.265, 0.896, 0.37, 0.42, 0.678, 1.109, 0.1, 0.534, 0.112, 1.187, 0.083, 0.336, 0.189, 0.652, -0.068, 1.187, 0.098],
    "far_L": [0.729, -0.064, 1.017, 0.108, 1.203, -0.111, 0.0, 0.705, 1.219, 0.068, 0.072, 0.505, 1.137, 0.119, 0.272, 0.614, 0.899, 0.264, 1.277, 0.363],
    "far_R": [0.531, -0.139, 0.809, 0.155, 0.991, 0.357, 0.556, 0.672, 1.245, 0.073, 0.706, 0.37, 1.476, 0.06, 0.621, 0.397, 1.021, -0.087, 1.241, 0.24],
    "hem_L": [0.885, 0.014, 1.112, 0.121, 0.873, -0.028, 0.0, 0.694, 0.947, 0.107, 0.046, 0.482, 0.975, 0.116, 0.203, 0.653, 1.025, 0.225, 0.68, 0.354],
}
QP = {
    "near_L": [0.46, 0.174, 0.014, 0.492, 0.443, -0.119, 0.0, 0.436, 0.461, 0.024, 0.001, 0.464, 0.376, 0.071, 0.0, 0.625, 0.051, 0.226, 0.0, 0.789],
    "near_R": [0.206, 0.221, 0.246, 0.431, 0.195, 0.233, 0.001, 0.36, 0.359, 0.031, 0.389, 0.037, 0.548, 0.03, 0.194, 0.131, 0.217, -0.253, 0.744, 0.406],
    "far_L": [0.593, 0.005, 0.5, 0.213, 0.569, -0.072, 0.0, 0.544, 0.644, 0.084, 0.004, 0.519, 0.633, 0.183, 0.055, 0.678, 0.309, 0.37, 0.893, 0.589],
    "far_R": [0.181, -0.019, 0.553, 0.288, 0.439, 0.222, 0.246, 0.47, 0.588, 0.009, 0.507, 0.193, 0.76, -0.008, 0.36, 0.297, 0.399, -0.185, 1.228, 0.276],
    "hem_L": [0.364, -0.028, 0.276, 0.07, 0.18, 0.008, 0.002, 0.262, 0.272, 0.112, 0.003, 0.379, 0.34, 0.18, 0.002, 0.556, 0.156, 0.366, 0.03, 0.459],
}
Q_FIST = [0.658, -0.139, 1.374, 0.115, 1.206, 0.275, 0.745, 0.86, 1.305, 0.034, 1.023, 0.673, 1.556, 0.078, 0.963, 0.563, 1.227, -0.127, 1.414, 0.186]
FLEX_UP = np.array([1.60, 0.93, 1.56, 1.57] + [1.55, 0.37, 1.54, 1.57] * 4)
PICK_RULE = {"near_L": ("x_n", 0.02, "y_h", -0.02), "near_R": ("x_n", -0.02, "y_c", 0.04),
             "far_L": ("x_f", -0.02, "y_h", -0.04), "far_R": ("x_f", -0.02, "y_c", 0.04),
             "hem_L": ("x_m", 0.0, "y_h", -0.02)}
SIDE_SHORT = 0.02
HEM_TO_COLLAR = 0.10
PICK_DZ = -0.005
LAND_DZ = 0.02


def grip(key):
    if GRIP == "fist" or (GRIP == "mixed" and key.startswith("near")):
        return np.array(Q_FIST, dtype=float)
    out = np.array(QC[key], dtype=float)
    for k in (2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19):
        out[k] = min(float(FLEX_UP[k]), out[k] + 0.5)
    return out


def plane_points(mask, z, stride=2, vw=None):
    vw = view if vw is None else vw
    v, u = np.nonzero(mask[::stride, ::stride])
    rays = vw.model.pixel_rays(u * stride, v * stride)
    d = rays @ vw.T_base_optical[:3, :3].T
    o = vw.T_base_optical[:3, 3]
    t = (z - o[2]) / d[:, 2]
    keep = t > 0
    return o + d[keep] * t[keep, None]


def shirt_mask(cam, lo, hi):
    best = None
    for q in ("t-shirt", "shirt", "clothes"):
        for item in api.segment(q, cam):
            area = int(item["mask"].sum())
            if lo <= area <= hi and (best is None or area > best[1]):
                best = (item, area, q)
        if best is not None:
            break
    return best


def print_sweep():
    views = {}
    for cam, hi in (("head", 6000), ("hand_left", 180000), ("hand_right", 180000)):
        vw = api.camera(cam)
        best = None
        for q in ("logo", "white logo", "airplane icon"):
            for item in api.segment(q, cam):
                area = int(item["mask"].sum())
                if 20 <= area <= hi and (best is None or item["score"] > best["score"]):
                    best = item
            if best is not None and best["score"] >= 0.3:
                break
        if best is not None and best["score"] >= 0.3 and vw.model is not None:
            views[cam] = (vw, best["mask"].astype(bool))
    if "head" not in views or len(views) < 2:
        return None
    hv, hm = views["head"]
    best = None
    for z in np.arange(0.55, 0.95, 0.001):
        P = plane_points(hm, float(z), 1, hv)
        sc = []
        for cam in views:
            if cam == "head":
                continue
            vw, wm = views[cam]
            uv, ok = vw.project_base_points(P)
            ok = ok & np.all(np.isfinite(uv), axis=1)
            if int(ok.sum()) < 10:
                continue
            uu = np.clip(uv[ok, 0].astype(int), 0, wm.shape[1] - 1)
            vv = np.clip(uv[ok, 1].astype(int), 0, wm.shape[0] - 1)
            proj = np.zeros(wm.shape, dtype=bool)
            proj[vv, uu] = True
            sc.append(float(np.mean(wm[vv, uu])) * float(np.sum(proj & wm)) / max(1.0, float(np.sum(wm))))
        if sc and (best is None or float(np.mean(sc)) > best[1]):
            best = (float(z), float(np.mean(sc)))
    report["print_sweep"] = None if best is None else [round(best[0], 4), round(best[1], 4), sorted(views)]
    return None if best is None or best[1] < 0.1 else best[0]


def cloth_height(head_mask):
    zp = print_sweep()
    if zp is not None:
        report["cloth_z_from"] = ["print sweep"]
        return zp
    for text in ("logo", "white logo", "airplane icon"):
        f = api.locate_rgb(text, cameras=("head", "hand_left", "hand_right"), min_score=0.1, max_rms_m=0.006)
        if f is not None and len(f["cameras"]) >= 2:
            report["cloth_z_from"] = [text, round(float(f["rms_m"]), 4), f["cameras"]]
            return float(f["point"][2])
    wr = []
    for cam in ("hand_left", "hand_right"):
        b = shirt_mask(cam, 20000, 700000)
        vw = api.camera(cam)
        if b is not None and vw.model is not None:
            wr.append((vw, b[0]["mask"].astype(bool)))
    best = None
    for z in np.arange(0.55, 0.95, 0.004):
        P = plane_points(head_mask, float(z), 3)
        sc = []
        for vw, wm in wr:
            uv, ok = vw.project_base_points(P)
            ok = ok & np.all(np.isfinite(uv), axis=1)
            if int(ok.sum()) > 50:
                sc.append(float(np.mean(wm[np.clip(uv[ok, 1].astype(int), 0, wm.shape[0] - 1), np.clip(uv[ok, 0].astype(int), 0, wm.shape[1] - 1)])))
        if sc and (best is None or float(np.mean(sc)) > best[1]):
            best = (float(z), float(np.mean(sc)))
    report["cloth_z_from"] = ["plane sweep", None if best is None else round(best[1], 3)]
    return None if best is None else best[0]


def shirt_features():
    b = shirt_mask("head", 8000, 250000)
    if b is None or view.model is None:
        return None
    m = b[0]["mask"].astype(bool)
    zc = cloth_height(m)
    if zc is None:
        return None
    S = plane_points(m, zc)
    ylo, yhi = float(np.percentile(S[:, 1], 1)), float(np.percentile(S[:, 1], 99))
    prof = []
    for y0 in np.arange(ylo, yhi, 0.01):
        B = S[(S[:, 1] >= y0) & (S[:, 1] < y0 + 0.01)]
        if B.shape[0] >= 20:
            prof.append((float(y0) + 0.005, float(np.percentile(B[:, 0], 2)), float(np.percentile(B[:, 0], 98))))
    if len(prof) < 20:
        return None
    P = np.array(prof)
    w = P[:, 2] - P[:, 1]
    half = len(P) // 2
    hem_plus_y = float(np.median(w[half:])) < float(np.median(w[:half]))
    if not hem_plus_y:
        report["shirt_features"] = {"error": "hem not toward +y: this program's fold sequence does not cover that"}
        return None
    narrow = P[half:]
    keep = narrow[np.abs((narrow[:, 2] - narrow[:, 1]) - float(np.median(narrow[:, 2] - narrow[:, 1]))) < 0.03]
    x_n, x_f = float(np.median(keep[:, 1])), float(np.median(keep[:, 2]))
    y_h, y_c = yhi, ylo
    f = {"z": zc, "x_n": x_n, "x_f": x_f, "x_m": 0.5 * (x_n + x_f), "W": x_f - x_n, "y_h": y_h, "y_c": y_c, "L": y_h - y_c}
    report["shirt_features"] = {k: round(v, 4) for k, v in f.items()}
    report.setdefault("shirt", []).append({"q": b[2], "area": b[1], "s": round(float(b[0]["score"]), 3)})
    return f


def pick_xy(key, f):
    xr, dx, yr, dy = PICK_RULE[key]
    return f[xr] + dx, f[yr] + dy


def land_xy(key, f):
    x, y = pick_xy(key, f)
    if key == "hem_L":
        return x, f["y_c"] + HEM_TO_COLLAR
    if key.startswith("near"):
        return f["x_m"] - SIDE_SHORT, y
    return f["x_m"] + SIDE_SHORT, y


def at(key, xy, z):
    return pose([xy[0], xy[1], z], ROT[key])


def above(T, h):
    out = T.copy()
    out[:3, 3] = out[:3, 3] - h * out[:3, 2]
    return out


JV = Limits(joint_velocity=1.0, joint_acceleration=3.0)
JV_SLOW = Limits(joint_velocity=0.4, joint_acceleration=1.2)
CREEP = Limits(joint_velocity=0.25, joint_acceleration=1.0)


def fold(keys, f, label, pick_dx=0.0, lift=0.10):
    m = api.motion()
    sides = {k: ("left" if k.endswith("_L") else "right") for k in keys}
    g = {k: offset(at(k, pick_xy(k, f), f["z"] + PICK_DZ), dx=pick_dx) for k in keys}
    over = {k: at(k, land_xy(k, f), f["z"] + PICK_DZ + lift) for k in keys}
    down = {k: at(k, land_xy(k, f), f["z"] + LAND_DZ) for k in keys}
    for k in keys:
        m.hand(sides[k], np.array(QP[k]), rows=10)
    m.tcp({sides[k]: above(g[k], 0.08) for k in keys}, limits=JV)
    m.tcp({sides[k]: above(g[k], 0.02) for k in keys}, limits=JV_SLOW)
    m.tcp({sides[k]: g[k] for k in keys}, limits=CREEP)
    for k in keys:
        m.hand(sides[k], grip(k), rows=30)
    m.hold(12)
    m.tcp({sides[k]: offset(g[k], dz=lift) for k in keys}, limits=CREEP)
    m.tcp({sides[k]: over[k] for k in keys}, limits=JV_SLOW)
    m.tcp({sides[k]: down[k] for k in keys}, limits=CREEP)
    m.hold(10)
    for k in keys:
        m.hand(sides[k], OPEN_Q, rows=15)
    m.hold(15)
    m.tcp({sides[k]: offset(down[k], dz=0.12) for k in keys}, limits=JV)
    m.send(label=label)


def hem_arc(f, label, pick_dx=0.0, steps=6):
    k = "hem_L"
    m = api.motion()
    g = offset(at(k, pick_xy(k, f), f["z"] + PICK_DZ), dx=pick_dx)
    land = at(k, land_xy(k, f), f["z"] + LAND_DZ)
    y0, y1 = float(g[1, 3]), float(land[1, 3])
    yf, r = 0.5 * (y0 + y1), 0.5 * (y0 - y1)
    z0 = float(g[2, 3])
    m.hand("left", np.array(QP[k]), rows=10)
    m.tcp("left", above(g, 0.08), limits=JV)
    m.tcp("left", above(g, 0.02), limits=JV_SLOW)
    m.tcp("left", g, limits=CREEP)
    m.hand("left", grip(k), rows=30)
    m.hold(12)
    for i in range(1, steps + 1):
        a = math.pi * i / steps
        p = np.array([g[0, 3] + (land[0, 3] - g[0, 3]) * i / steps, yf + r * math.cos(a), z0 + 0.02 + r * math.sin(a)])
        m.tcp("left", pose(p, g[:3, :3]), linear=True, limits=Limits(linear_velocity=0.10))
    m.tcp("left", pose([land[0, 3], land[1, 3], f["z"] + LAND_DZ], g[:3, :3]), linear=True, limits=Limits(linear_velocity=0.05))
    m.hold(10)
    m.hand("left", OPEN_Q, rows=15)
    m.hold(15)
    m.tcp("left", offset(pose([land[0, 3], land[1, 3], f["z"] + LAND_DZ], g[:3, :3]), dz=0.15), limits=JV)
    m.send(label=label)


f = shirt_features()
if f is not None:
    report["picks"] = {k: [round(float(v), 4) for v in pick_xy(k, f)] for k in PICK_RULE}
    report["lands"] = {k: [round(float(v), 4) for v in land_xy(k, f)] for k in PICK_RULE}
    done = []
    for keys, label, dxs in ((("near_L", "near_R"), "fold_near_side", (0.0, 0.02)),
                             (("far_L", "far_R"), "fold_far_side", (0.0, -0.02, -0.04, -0.06)),
                             (("hem_L",), "fold_hem_to_collar", (0.0, -0.02))):
        errors = []
        for dx in dxs:
            try:
                if label == "fold_hem_to_collar":
                    hem_arc(f, label, pick_dx=dx)
                else:
                    fold(keys, f, label, pick_dx=dx)
                done.append([label, dx])
                break
            except MotionError as error:
                errors.append(f"{dx}: {str(error)[:100]}")
        if errors:
            report.setdefault("plan_errors", {})[label] = errors
    report["done"] = done
report["budget"] = api.budget()["phases"]
api.end_episode()
