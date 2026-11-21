# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

view = api.camera("head")
OPEN_Q = wuji_open()
Q_CAN = np.array([0.678, -0.06, 1.028, 0.134, 0.795, -0.30, 0.648, 0.811, 0.987, 0.033, 0.176, 0.748, 1.295, 0.127, 0.374, 0.691,
                  1.261, 0.158, 1.096, 0.397])
Q_CAN_PRE = np.array([0.45, 0.09, 0.05, 0.33, 0.31, -0.26, 0.11, 0.74, 0.46, -0.01, 0.01, 0.66, 0.51, 0.18, 0.03, 0.69, 0.41, 0.37,
                      0.06, 0.52])
Q_BAG = np.array([0.658, -0.139, 1.374, 0.115, 1.206, 0.275, 0.745, 0.86, 1.305, 0.034, 1.023, 0.673, 1.556, 0.078, 0.963, 0.563,
                  1.227, -0.127, 1.414, 0.186])
CAN_GRASP_D = np.array([0.0274, 0.0252, -0.0209])
CAN_REL_D = np.array([-0.003, 0.0296, 0.2312])
BAG_GRASP_D = np.array([0.1211, -0.0022, -0.0513])
BAG_TOP_D = np.array([-0.0163, -0.0401, 0.5811])
BAG_REL_D = np.array([0.0976, 0.0278, 0.2088])
PARK_DZ = 0.35
report["segments"] = list(api.task_info().get("segments") or [])


def unit_rot(rows):
    u, s, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    m = u @ vt
    if np.linalg.det(m) < 0:
        m = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return m


R_CAN = unit_rot([[-0.638, 0.72, -0.273], [0.73, 0.678, 0.086], [0.248, -0.144, -0.958]])
R_CAN_REL = unit_rot([[-0.125, 0.988, 0.095], [0.99, 0.131, -0.054], [-0.066, 0.088, -0.994]])
R_BAG = unit_rot([[-0.732, -0.674, -0.095], [-0.625, 0.72, -0.301], [0.272, -0.161, -0.949]])
R_BAG_TOP = unit_rot([[-0.187, -0.695, 0.695], [-0.888, 0.423, 0.184], [-0.421, -0.582, -0.696]])
R_BAG_REL = unit_rot([[-0.347, -0.936, 0.053], [-0.935, 0.34, -0.1], [0.076, -0.085, -0.994]])


def yawed(rot, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ rot


def plane_points(mask, z, stride=2):
    v, u = np.nonzero(mask[::stride, ::stride])
    rays = view.model.pixel_rays(u * stride, v * stride)
    d = rays @ view.T_base_optical[:3, :3].T
    o = view.T_base_optical[:3, 3]
    t = (z - o[2]) / d[:, 2]
    keep = t > 0
    return o + d[keep] * t[keep, None]


def masks(queries, lo, hi, n=3):
    found = []
    for q in queries:
        for item in api.segment(q, "head"):
            area = int(item["mask"].sum())
            if lo <= area <= hi:
                found.append(dict(item, query=q))
        if len(found) >= n:
            break
    found.sort(key=lambda it: -float(it["score"]))
    return found


def first_plan(side, label, build, variants):
    errors = []
    for var in variants:
        m = api.motion()
        try:
            build(m, var)
        except MotionError as error:
            errors.append(f"{var}: {str(error)[:80]}")
            continue
        report.setdefault("plans", []).append({"label": label, "variant": var, "errors_before": len(errors)})
        m.send(label=label)
        return var
    report.setdefault("plan_errors", {})[label] = errors[:4]
    return None


JV = Limits(joint_velocity=1.2, joint_acceleration=4.0)
JV_SLOW = Limits(joint_velocity=0.7, joint_acceleration=2.5)

def ray_of(vw, mask):
    v, u = np.nonzero(mask)
    r = vw.model.pixel_rays(np.array([float(np.mean(u))]), np.array([float(np.mean(v))]))[0]
    return vw.T_base_optical[:3, 3], vw.T_base_optical[:3, :3] @ (r / np.linalg.norm(r))


def closest(o1, d1, o2, d2):
    w0 = o1 - o2
    b_ = float(d1 @ d2)
    dd, e = float(d1 @ w0), float(d2 @ w0)
    den = max(1.0 - b_ * b_, 1e-9)
    s_ = (b_ * e - dd) / den
    t_ = (e - b_ * dd) / den
    p1, p2 = o1 + s_ * d1, o2 + t_ * d2
    return 0.5 * (p1 + p2), float(np.linalg.norm(p1 - p2))


def masks_of(texts, cam, lo, hi, smin):
    out = []
    for q in texts:
        for it in api.segment(q, cam):
            a_ = int(it["mask"].sum())
            if lo <= a_ <= hi and float(it["score"]) >= smin:
                out.append(it)
        if out:
            break
    return out


z_top = None
if view.model is not None:
    hv, lv = api.camera("head"), api.camera("hand_left")
    tops = []
    if lv.model is not None:
        hl = masks_of(["can lid", "top of the can"], "head", 100, 5000, 0.2)
        wl = masks_of(["can lid", "top of the can"], "hand_left", 300, 60000, 0.2)
        pairs = sorted([closest(*ray_of(hv, a_["mask"]), *ray_of(lv, b_["mask"])) for a_ in hl for b_ in wl], key=lambda t: t[1])
        tops = [p_ for p_, dist in pairs if dist <= 0.01]
    if tops:
        z_top = float(tops[0][2])
        report["can_top"] = [round(float(v), 4) for v in tops[0]]
bin_c = None
bin_r = None


def reachable(c):
    return float(c[0]) >= 0.25 and float(np.hypot(c[0], c[1])) <= 0.85


if z_top is not None:
    rejected = []
    for q in ("trash can", "bin", "garbage can"):
        f = api.locate_rgb(q, cameras=("head", "hand_left", "hand_right"), min_score=0.05, max_rms_m=0.02)
        if f is not None:
            c = np.asarray(f["point"], dtype=float)[:2]
            if not reachable(c):
                rejected.append([round(float(v), 3) for v in c] + [f["cameras"], q])
                continue
            bin_c = c
            report["bin"] = {"centre": [round(float(v), 4) for v in bin_c], "rms": round(f["rms_m"], 4), "cams": f["cameras"], "query": q}
            break
    report["bin_rejected"] = rejected
    if bin_c is None:
        hv = api.camera("head")
        hb = masks_of(["trash can", "bin", "garbage can"], "head", 1500, 80000, 0.05)
        if hb and hv.model is not None:
            best = max(hb, key=lambda it: float(it["score"]))
            o_, d_ = ray_of(hv, best["mask"])
            if d_[2] < -1e-6:
                c = (o_ + d_ * ((z_top - o_[2]) / d_[2]))[:2]
                report["bin_head_ray"] = [round(float(v), 4) for v in c] + [round(float(best["score"]), 3)]
                if reachable(c):
                    bin_c = c
                    report["bin"] = {"centre": [round(float(v), 4) for v in bin_c], "cams": ["head"], "head_ray": True}
    if bin_c is not None:
        bins = masks_of(["trash can", "bin", "garbage can"], "head", 1500, 80000, 0.0)
        if bins:
            B = plane_points(max(bins, key=lambda it: float(it["score"]))["mask"], z_top)
            bin_r = 0.5 * float(np.percentile(B[:, 1], 97) - np.percentile(B[:, 1], 3))
        report["bin_r"] = None if bin_r is None else round(bin_r, 3)
away = (bin_r + 0.04) if bin_r is not None else 0.12

cans = []
if bin_c is not None:
    for b in masks(["soda can", "beverage can", "can", "bottle"], 150, 20000, n=4):
        P = plane_points(b["mask"], z_top - 0.015)
        if P.shape[0] < 20:
            continue
        c = np.median(P, axis=0)[:2]
        if float(np.linalg.norm(c - bin_c)) < away or any(float(np.linalg.norm(c - d)) < 0.04 for d in cans):
            continue
        cans.append(c)
    cans.sort(key=lambda c: -float(c[0]))
    report["cans"] = [[round(float(v), 3) for v in c] for c in cans[:2]]

if bin_c is not None:
    bags = []
    for q in ("transparent plastic bag", "plastic bag", "transparent bag", "bag"):
        for item in api.segment(q, "head"):
            area = int(item["mask"].sum())
            if 6000 <= area <= 150000:
                Pm = plane_points(item["mask"], z_top - 0.05)
                if Pm.shape[0] >= 50 and float(np.linalg.norm(np.median(Pm, axis=0)[:2] - bin_c)) > away:
                    bags.append(dict(item, query=q, area=area))
        if bags:
            break
    bags.sort(key=lambda it: -it["area"])
    bag_pt = None
    if bags:
        G = plane_points(bags[0]["mask"], z_top - 0.05)
        G = G[np.linalg.norm(G[:, :2] - bin_c[None, :], axis=1) > away]
        if G.shape[0] >= 50:
            x_near = float(np.percentile(G[:, 0], 3))
            y_mid = float(np.median(G[:, 1]))
            bag_pt = np.array([x_near + BAG_GRASP_D[0], y_mid + BAG_GRASP_D[1]])
            report["bag"] = {"x_near": round(x_near, 3), "y_mid": round(y_mid, 3), "query": bags[0]["query"],
                             "score": round(float(bags[0]["score"]), 3), "area": bags[0]["area"]}
    if bag_pt is not None:
        def bag_motion(m, var):
            dx, dy, yaw = var
            R = yawed(R_BAG, yaw)
            g = pose([bag_pt[0] + dx, bag_pt[1] + dy, z_top + BAG_GRASP_D[2]], R)
            n = R[:, 2]
            m.hand("right", OPEN_Q, rows=10)
            m.tcp("right", pose(g[:3, 3] - 0.10 * n, R), limits=JV)
            m.tcp("right", g, linear=True, limits=Limits(linear_velocity=0.05))
            m.hand("right", Q_BAG, rows=45)
            m.hold(15)
            m.tcp("right", offset(g, dz=0.12), linear=True, limits=Limits(linear_velocity=0.08))
            top = pose([bin_c[0] + BAG_TOP_D[0], bin_c[1] + BAG_TOP_D[1], z_top + BAG_TOP_D[2]], R_BAG_TOP)
            m.tcp("right", top, limits=JV_SLOW)
            over = pose([bin_c[0] + BAG_REL_D[0], bin_c[1] + BAG_REL_D[1], z_top + BAG_REL_D[2] + 0.22], R_BAG_REL)
            m.tcp("right", over, limits=JV_SLOW)
            rel = pose([bin_c[0] + BAG_REL_D[0], bin_c[1] + BAG_REL_D[1], z_top + BAG_REL_D[2]], R_BAG_REL)
            m.tcp("right", rel, linear=True, limits=Limits(linear_velocity=0.12))
            m.hold(10)
            m.hand("right", OPEN_Q, rows=20)
            m.hold(15)
            m.tcp("right", offset(rel, dz=0.15), linear=True, limits=Limits(linear_velocity=0.15))
            m.tcp("right", pose([bin_c[0] - 0.05, bin_c[1] - 0.28, z_top + PARK_DZ], R_BAG_REL), limits=JV)
        report["bag_plan"] = first_plan("right", "bag_into_bin", bag_motion,
                                        [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0), (0.0, 0.0, -10.0), (-0.01, 0.0, 20.0), (0.0, 0.0, -20.0)])

for k, c in enumerate(cans[:2]):
    def can_motion(m, var, c=c):
        yaw, dz_rel = var
        R = yawed(R_CAN, yaw)
        off = yawed(np.eye(3), yaw) @ np.array([CAN_GRASP_D[0], CAN_GRASP_D[1], 0.0])
        g = pose([c[0] + off[0], c[1] + off[1], z_top + CAN_GRASP_D[2]], R)
        n = R[:, 2]
        rel = pose([bin_c[0] + CAN_REL_D[0], bin_c[1] + CAN_REL_D[1], z_top + CAN_REL_D[2] + dz_rel], R_CAN_REL)
        m.hand("left", Q_CAN_PRE, rows=12)
        m.tcp("left", pose(g[:3, 3] - 0.10 * n, R), limits=JV)
        m.tcp("left", pose(g[:3, 3] - 0.03 * n, R), linear=True, limits=Limits(linear_velocity=0.06))
        m.tcp("left", g, linear=True, limits=Limits(linear_velocity=0.03))
        m.hand("left", Q_CAN, rows=30)
        m.hold(12)
        m.tcp("left", offset(g, dz=0.10), linear=True, limits=Limits(linear_velocity=0.08))
        m.tcp("left", offset(rel, dz=0.10), limits=JV_SLOW)
        m.tcp("left", rel, linear=True, limits=Limits(linear_velocity=0.08))
        m.hold(6)
        m.hand("left", Q_CAN_PRE, rows=15)
        m.hold(20)
        m.tcp("left", offset(rel, dz=0.12), linear=True, limits=Limits(linear_velocity=0.15))
    report.setdefault("can_plan", []).append(first_plan("left", f"can_{k + 1}", can_motion,
                                                        [(0.0, 0.0), (10.0, 0.0), (-10.0, 0.0), (20.0, 0.03), (-20.0, 0.03)]))
report["budget"] = api.budget()["phases"]
api.end_episode()
