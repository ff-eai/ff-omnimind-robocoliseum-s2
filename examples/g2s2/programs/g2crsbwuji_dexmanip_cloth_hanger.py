# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

view = api.camera("head")
OPEN_Q = wuji_open()
Q_CLOTH = np.array([0.732, 0.081, 1.068, 0.143, 0.738, -0.121, 0.018, 1.053, 0.963, 0.046, 0.089, 0.681, 1.04, 0.102, 0.115, 0.686,
                    1.135, 0.183, 0.518, 0.564])
Q_HANGER = np.array([0.478, -0.027, 0.691, 0.436, 0.728, 0.176, 0.511, 0.903, 0.901, -0.001, 0.857, 0.209, 1.09, 0.038, 0.58, 0.309,
                     0.566, -0.274, 1.395, 0.415])
Q_HANGER_PRE = np.array([0.22, -0.13, 0.72, 0.17, 0.51, 0.03, 0.08, 0.88, 0.55, -0.06, 0.68, 0.2, 0.79, -0.05, 0.31, 0.26, 0.27, -0.37,
                         1.36, 0.41])
report["segments"] = list(api.task_info().get("segments") or [])
BOOST = 0.5
FLEX_UP = np.array([1.60, 0.93, 1.56, 1.57] + [1.55, 0.37, 1.54, 1.57] * 4)


Q_FIST = np.array([0.658, -0.139, 1.374, 0.115, 1.206, 0.275, 0.745, 0.86, 1.305, 0.034, 1.023, 0.673, 1.556, 0.078, 0.963, 0.563,
                   1.227, -0.127, 1.414, 0.186])


def boosted(q):
    out = np.array(q, dtype=float)
    for k in (2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19):
        out[k] = min(float(FLEX_UP[k]), out[k] + BOOST)
    return out


def unit_rot(rows):
    u, s, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    m = u @ vt
    if np.linalg.det(m) < 0:
        m = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return m


R_CLOTH = unit_rot([[-0.964, 0.232, -0.13], [0.188, 0.939, 0.287], [0.188, 0.252, -0.949]])
R_CLOTH_TOP = unit_rot([[-0.769, 0.438, 0.465], [0.524, 0.849, 0.066], [-0.366, 0.295, -0.883]])
R_HANGER = unit_rot([[-0.768, -0.629, 0.123], [-0.608, 0.654, -0.45], [0.202, -0.42, -0.885]])
R_HANGER_UP = unit_rot([[-0.749, -0.657, 0.08], [-0.661, 0.75, -0.034], [-0.038, -0.079, -0.996]])
HANGER_GRASP_D = np.array([0.0351, 0.0468])
HANGER_GRASP_DZ = -0.0053
CLOTH_GRASP_D = np.array([0.1309, 0.0275])
CLOTH_GRASP_DZ = 0.004
CLOTH_TOP_REL = np.array([0.1828, -0.1451, 0.4956])


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


def candidates(queries, lo, hi):
    out = []
    for q in queries:
        for item in api.segment(q, "head"):
            area = int(item["mask"].sum())
            if lo <= area <= hi:
                out.append(dict(item, query=q, area=area))
        if out:
            break
    out.sort(key=lambda it: -float(it["score"]))
    return out


def first_plan(label, build, variants):
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
JV_SLOW = Limits(joint_velocity=0.6, joint_acceleration=2.0)

def best_mask(queries, cam, lo, hi):
    best = None
    for q in queries:
        for item in api.segment(q, cam):
            area = int(item["mask"].sum())
            if lo <= area <= hi and (best is None or item["score"] > best["score"]):
                best = item
        if best is not None and best["score"] >= 0.3:
            break
    return best


def plane_hits(vw, mask, z, stride):
    v, u = np.nonzero(mask[::stride, ::stride])
    rays = vw.model.pixel_rays((u * stride).astype(float), (v * stride).astype(float))
    d = rays @ vw.T_base_optical[:3, :3].T
    o = vw.T_base_optical[:3, 3]
    t = (z - o[2]) / d[:, 2]
    keep = t > 0
    return o + d[keep] * t[keep, None]


def hanger_plane():
    views = {}
    for cam, hi in (("head", 20000), ("hand_left", 600000), ("hand_right", 600000)):
        vw = api.camera(cam)
        it = best_mask(["wooden hanger", "clothes hanger", "hanger"], cam, 300, hi)
        if it is not None and vw.model is not None and it["score"] >= 0.3:
            views[cam] = (vw, it["mask"].astype(bool))
    report["hanger_views"] = sorted(views)
    if "head" not in views or len(views) < 2:
        return None
    hv, hm = views["head"]
    best = None
    for z in np.arange(0.55, 0.95, 0.002):
        P = plane_hits(hv, hm, float(z), 1)
        sc = []
        for cam in views:
            if cam == "head":
                continue
            vw, wm = views[cam]
            uv, ok = vw.project_base_points(P)
            ok = ok & np.all(np.isfinite(uv), axis=1)
            if int(ok.sum()) < 30:
                continue
            uu = np.clip(uv[ok, 0].astype(int), 0, wm.shape[1] - 1)
            vv = np.clip(uv[ok, 1].astype(int), 0, wm.shape[0] - 1)
            proj = np.zeros(wm.shape, dtype=bool)
            proj[vv, uu] = True
            sc.append(float(np.mean(wm[vv, uu])) * float(np.sum(proj & wm)) / max(1.0, float(np.sum(wm))))
        if sc and (best is None or float(np.mean(sc)) > best[1]):
            best = (float(z), float(np.mean(sc)))
    report["hanger_plane"] = None if best is None else [round(best[0], 4), round(best[1], 4)]
    return None if best is None or best[1] < 0.15 else best[0]


ZH = hanger_plane()
hangers = candidates(["wooden hanger", "clothes hanger", "hanger"], 600, 20000)
shirts = candidates(["t-shirt", "shirt"], 8000, 200000)
report["masks"] = {k: [{"q": it["query"], "s": round(float(it["score"]), 3), "a": it["area"]} for it in v[:3]]
                   for k, v in (("hanger", hangers), ("shirt", shirts))}

hanger_end = None
if view.model is not None and hangers and ZH is not None:
    H = plane_points(hangers[0]["mask"], ZH)
    y_tip = float(np.percentile(H[:, 1], 2))
    slab = H[H[:, 1] < y_tip + 0.06]
    x_bar = float(np.median(slab[:, 0]))
    hanger_end = np.array([x_bar, y_tip])
    report["hanger"] = {"bar_x": round(x_bar, 3), "tip_y": round(y_tip, 3), "plane_z": round(ZH, 4)}

cloth_pt = None
if view.model is not None and shirts and ZH is not None:
    S = plane_points(shirts[0]["mask"], ZH)
    x2 = float(np.percentile(S[:, 0], 2))
    ymid = 0.5 * (float(np.percentile(S[:, 1], 2)) + float(np.percentile(S[:, 1], 98)))
    cloth_pt = np.array([x2, ymid]) + CLOTH_GRASP_D
    report["shirt"] = {"x2": round(x2, 3), "ymid": round(ymid, 3), "grasp": [round(float(v), 3) for v in cloth_pt]}

if hanger_end is not None:
    def hanger_motion(m, var):
        dx, dy, yaw = var
        R = yawed(R_HANGER, yaw)
        p = np.array([hanger_end[0] + HANGER_GRASP_D[0] + dx, hanger_end[1] + HANGER_GRASP_D[1] + dy, ZH + HANGER_GRASP_DZ])
        g = pose(p, R)
        m.hand("right", Q_HANGER_PRE, rows=12)
        m.tcp("right", pose(p + 0.06 * R[:, 0] - 0.08 * R[:, 2], R), limits=JV)
        m.tcp("right", pose(p + 0.025 * R[:, 0] - 0.02 * R[:, 2], R), linear=True, limits=Limits(linear_velocity=0.05))
        m.tcp("right", g, linear=True, limits=Limits(linear_velocity=0.025))
        m.hand("right", boosted(Q_HANGER), rows=40)
        m.hold(12)
        m.tcp("right", pose(p + np.array([-0.011, -0.012, 0.087]), R_HANGER_UP), linear=True, limits=Limits(linear_velocity=0.05))
        lifted = pose(p + np.array([-0.03, 0.0, 0.22]), R_HANGER_UP)
        m.tcp("right", lifted, linear=True, limits=Limits(linear_velocity=0.08))
        m.hold(20)
    report["hanger_plan"] = first_plan("lift_hanger", hanger_motion,
                                       [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0), (0.0, 0.0, -10.0), (-0.01, 0.01, 0.0)])

if cloth_pt is not None:
    def cloth_motion(m, var):
        dx, dy, yaw = var
        R = yawed(R_CLOTH, yaw)
        p = np.array([cloth_pt[0] + dx, cloth_pt[1] + dy, ZH + CLOTH_GRASP_DZ])
        g = pose(p, R)
        m.hand("left", OPEN_Q, rows=10)
        m.tcp("left", pose(p + 0.04 * R[:, 0] - 0.10 * R[:, 2], R), limits=JV)
        m.tcp("left", pose(p + 0.015 * R[:, 0] - 0.03 * R[:, 2], R), linear=True, limits=Limits(linear_velocity=0.06))
        m.tcp("left", g, linear=True, limits=Limits(linear_velocity=0.025))
        m.hand("left", boosted(Q_CLOTH), rows=45)
        m.hold(15)
        m.tcp("left", offset(g, dz=0.10), linear=True, limits=Limits(linear_velocity=0.06))
        m.tcp("left", pose(p + CLOTH_TOP_REL, R_CLOTH_TOP), limits=JV_SLOW)
        m.hold(30)
    report["cloth_plan"] = first_plan("lift_cloth", cloth_motion,
                                      [(0.0, 0.0, 0.0), (0.0, 0.0, 10.0), (0.0, 0.0, -10.0), (0.01, 0.0, 20.0)])
report["budget"] = api.budget()["phases"]
api.end_episode()
