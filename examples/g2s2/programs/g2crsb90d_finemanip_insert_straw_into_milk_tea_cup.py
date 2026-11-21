# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

SETTLE_ROWS = int(round(2.0 / float(api.task_info()["row_dt_s"])))
seg0 = api.segment_text(0)
side = api.instructed_arm(seg0) or "left"
table = api.table_height("head")
report["side"] = side
cup = api.find_object(["milk tea cup", "cup with lid", "plastic cup"], size=(0.04, 0.20))
report["cup"] = None if cup is None else cup.summary()

r_straw = None
w_seen = None


def grow(mask, k):
    out = mask.copy()
    for i_ in range(k):
        out = out | np.roll(out, 1, 0) | np.roll(out, -1, 0) | np.roll(out, 1, 1) | np.roll(out, -1, 1)
    return out


def hole_from_head():
    if cup is None:
        return None
    view = api.camera("head")
    if view.model is None or view.T_base_optical is None:
        return None
    zp = float(cup.top_z)
    H, W = view.rgb.shape[:2]
    vv, uu = np.mgrid[0:H, 0:W]
    rays = view.model.pixel_rays(uu.ravel().astype(float), vv.ravel().astype(float))
    Rb = view.T_base_optical[:3, :3]
    ob = view.T_base_optical[:3, 3]
    dirs = rays @ Rb.T
    dz = dirs[:, 2]
    okz = np.abs(dz) > 1e-6
    t = np.where(okz, (zp - ob[2]) / np.where(okz, dz, 1.0), -1.0)
    P = ob + dirs * t[:, None]
    c = np.asarray(cup.mid[:2], dtype=float)
    rr = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    lum = view.rgb.reshape(-1, 3).astype(float).mean(axis=1)
    sel = (t > 0.05) & (rr < 0.045) & (lum < 70.0)
    if 15 <= int(sel.sum()) <= 400 and view.depth_m is not None:
        dark = sel.reshape(H, W)
        Pb = view.points_from_mask(grow(dark, 4) & ~grow(dark, 1), stride=1)
        Pb = Pb[np.all(np.isfinite(Pb), axis=1)]
        Pb = Pb[(Pb[:, 2] > float(cup.top_z) - 0.03) & (Pb[:, 2] < float(cup.top_z) + 0.005)]
        if Pb.shape[0] >= 10:
            zp = float(np.median(Pb[:, 2]))
            t = np.where(okz, (zp - ob[2]) / np.where(okz, dz, 1.0), -1.0)
            P = ob + dirs * t[:, None]
            rr = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
            sel = (t > 0.05) & (rr < 0.045) & (lum < 70.0)
    n = int(sel.sum())
    if n < 15 or n > 400:
        return {"n": n}
    xy = np.median(P[sel][:, :2], axis=0)
    if float(np.linalg.norm(xy - c)) > 0.035:
        return {"n": n, "far": round(float(np.linalg.norm(xy - c)), 4)}
    return {"xy": xy, "z": zp, "n": n}


hole = hole_from_head()
if hole is None:
    report["hole"] = None
else:
    report["hole"] = {k: ([round(float(v), 4) for v in hole[k]] if k == "xy" else (round(float(hole[k]), 4) if isinstance(hole[k], float) else hole[k]))
                      for k in hole}


def elongated(P):
    if P.shape[0] < 15:
        return None
    c = np.mean(P[:, :2], axis=0)
    cov = np.cov((P[:, :2] - c).T)
    vals, vecs = np.linalg.eigh(cov)
    ax = vecs[:, 1]
    proj = (P[:, :2] - c) @ ax
    side_ = (P[:, :2] - c) @ np.array([-ax[1], ax[0]])
    length = float(np.percentile(proj, 98) - np.percentile(proj, 2))
    width = float(np.percentile(side_, 95) - np.percentile(side_, 5))
    return c, math.atan2(float(ax[1]), float(ax[0])), length, width


def tip_end(P, c, yaw, r_s):
    ax = np.array([math.cos(yaw), math.sin(yaw)])
    pp = np.array([-ax[1], ax[0]])
    a = (P[:, :2] - c) @ ax
    p = (P[:, :2] - c) @ pp
    p = p - float(np.median(p))
    out = {}
    for sgn in (1.0, -1.0):
        aa = a * sgn
        amax = float(np.percentile(aa, 99.7))
        sel = aa > amax - 2.56 * r_s
        xs, ys = [], []
        for b0 in np.linspace(-0.7 * r_s, 0.7 * r_s, 9):
            m_ = sel & (np.abs(p - b0) < 0.088 * r_s)
            if int(m_.sum()) >= 3:
                xs.append(float(b0))
                ys.append(float(np.percentile(aa[m_], 95)))
        if len(xs) < 5:
            return None
        A = np.vstack([np.array(xs), np.ones(len(xs))]).T
        beta = float(np.linalg.lstsq(A, np.array(ys), rcond=None)[0][0])
        out[sgn] = (abs(beta), amax)
    s_tip = 1.0 if out[1.0][0] >= out[-1.0][0] else -1.0
    u = ax * s_tip
    e = c + u * out[s_tip][1]
    return u, e, (round(out[1.0][0], 3), round(out[-1.0][0], 3))


def straw_width_r(cam_name, c_, yaw_, tz_, h_top):
    if not api.has_depth(cam_name):
        return None
    P_ = api.points(cam_name, stride=1)
    P_ = P_[np.all(np.isfinite(P_), axis=1)]
    u_ = np.array([math.cos(yaw_), math.sin(yaw_)])
    v_ = np.array([-math.sin(yaw_), math.cos(yaw_)])
    rel_ = P_[:, :2] - np.asarray(c_, dtype=float)
    s_, t_ = rel_ @ u_, rel_ @ v_
    thr_ = max(0.003, 0.5 * float(h_top))
    sel_ = (np.abs(s_) < 0.04) & (np.abs(t_) < 0.012) & (P_[:, 2] > tz_ + thr_) & (P_[:, 2] < tz_ + 0.03)
    corr_ = (np.abs(s_) < 0.04) & (np.abs(t_) < 0.012)
    report["r_width_debug"] = {"pts": int(P_.shape[0]), "corridor": int(corr_.sum()), "raised": int(sel_.sum()),
                               "corr_z_pct": None if int(corr_.sum()) < 5 else [round(float(np.percentile(P_[corr_, 2], q)) - float(tz_), 4) for q in (5, 50, 95)]}
    if int(sel_.sum()) < 60:
        return None
    widths = []
    for s0 in np.arange(-0.04, 0.04, 0.005):
        sl = sel_ & (s_ >= s0) & (s_ < s0 + 0.005)
        if int(sl.sum()) >= 10:
            widths.append(float(np.percentile(t_[sl], 98) - np.percentile(t_[sl], 2)) / 0.96)
    report["r_width_debug"]["slices"] = [round(w_, 4) for w_ in widths]
    if len(widths) < 4:
        return None
    r_ = 0.5 * float(np.median(widths))
    return r_ if 0.002 < r_ < 0.012 else None


def straw_find(cam, table_z, near=None):
    best = None
    for q in ("white drinking straw", "straw", "white straw"):
        for d in api.perceive(q, cam, max_detections=4):
            P = d.points
            P = P[(P[:, 2] > table_z - 0.01) & (P[:, 2] < table_z + 0.025)]
            e = elongated(P)
            if e is None:
                continue
            c, yaw, length, width = e
            if cup is not None and float(np.linalg.norm(c - cup.mid[:2])) < 0.07:
                continue
            if near is not None and float(np.linalg.norm(c - near)) > 0.06:
                continue
            if 0.05 < length < 0.22 and width < 0.025:
                if best is None or d.score > best[0]:
                    best = (float(d.score), c, yaw, length, width, "sam:" + q, P)
        if best is not None:
            break
    if best is not None:
        return best
    view = api.camera(cam)
    img = view.rgb.astype(float)
    lo_c, hi_c = np.min(img, axis=2), np.max(img, axis=2)
    P_all = view.points_from_mask(np.ones(lo_c.shape, dtype=bool), stride=1)
    ys, xs = np.nonzero(np.ones(lo_c.shape, dtype=bool))
    if P_all.shape[0] != ys.shape[0]:
        return None
    layer = (P_all[:, 2] > table_z - 0.005) & (P_all[:, 2] < table_z + 0.02)
    if np.count_nonzero(layer) < 100:
        return None
    lum = lo_c[ys, xs]
    thr = float(np.median(lum[layer])) + 30.0
    sel = layer & (lum > thr) & ((hi_c[ys, xs] - lo_c[ys, xs]) < 35.0)
    P = P_all[sel]
    if cup is not None:
        P = P[np.linalg.norm(P[:, :2] - cup.mid[:2], axis=1) > 0.07]
    if near is not None:
        P = P[np.linalg.norm(P[:, :2] - near, axis=1) < 0.10]
    if P.shape[0] < 15:
        return None
    c0 = np.median(P[:, :2], axis=0)
    P = P[np.linalg.norm(P[:, :2] - c0, axis=1) < 0.11]
    e = elongated(P)
    if e is None:
        return None
    c, yaw, length, width = e
    if 0.05 < length < 0.22 and width < 0.03:
        return (0.0, c, yaw, length, width, "colour", P)
    return None


picked = False
tip = None
found = straw_find("head", table)
report["head_straw"] = None if found is None else {"src": found[5], "c": [round(float(v), 4) for v in found[1]], "yaw_deg": round(math.degrees(found[2]), 1),
                                                    "len": round(found[3], 3), "width": round(found[4], 3)}
if found is not None:
    c, yaw = found[1], found[2]
    h0 = float(np.percentile(found[6][:, 2], 95)) - float(table)
    r_straw = 0.5 * h0 if 0.004 < h0 < 0.03 else min(max(0.5 * float(found[4]), 0.002), 0.015)
    report["r_straw_head"] = round(r_straw, 4)
    fast = Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30)
    tz = table
    try:
        m = None
        for h_, tl_, cy, fa in [(h_, tl_, cy, fa) for h_, tl_ in ((0.14, 0.0), (0.10, 0.0), (0.12, 0.35))
                                for cy, fa in ((yaw + math.pi / 2.0, None), (yaw - math.pi / 2.0, None), (yaw + math.pi / 2.0, (0.0, 0.0, 1.0)))]:
            try:
                m = api.motion()
                g_now = api.grasp_pose(side).copy()
                g_up = g_now.copy()
                g_up[2, 3] = max(float(g_now[2, 3]) + 0.15, table + 0.30)
                try:
                    m.tcp(side, g_up, linear=True, limits=Limits(linear_velocity=0.35))
                    lifted_first = True
                except MotionError:
                    m = api.motion()
                    lifted_first = False
                m.tcp(side, top_down((float(c[0]), float(c[1]), table + 0.30), cy, tilt=tl_, tilt_toward=math.pi), gripper="open", free_axis=fa,
                      limits=Limits(joint_velocity=3.0, joint_acceleration=12.0, linear_velocity=0.30))
                try:
                    m2 = m.copy()
                    m2.tcp(side, top_down((float(c[0]), float(c[1]), table + h_), cy, tilt=tl_, tilt_toward=math.pi), linear=True, free_axis=fa,
                           limits=Limits(linear_velocity=0.35))
                    m = m2
                except MotionError:
                    m = api.motion()
                    if lifted_first:
                        m.tcp(side, g_up, linear=True, limits=Limits(linear_velocity=0.35))
                    m.tcp(side, top_down((float(c[0]), float(c[1]), table + h_), cy, tilt=tl_, tilt_toward=math.pi), gripper="open", free_axis=fa,
                          limits=Limits(joint_velocity=3.0, joint_acceleration=12.0, linear_velocity=0.30))
                    report["hover_descent"] = "lift+joint"
                report["hover"] = [h_, tl_, None if fa is None else "free", "lift" if lifted_first else "no-lift"]
                break
            except MotionError as error:
                report.setdefault("hover_errors", []).append(str(error)[:90])
                m = None
        if m is None:
            raise MotionError("no hover IK over the straw")
        m.send(depth_next=True, label="hover_straw")
        cam = f"hand_{side}"
        if api.has_depth(cam):
            tw = api.table_height(cam)
            if math.isfinite(tw) and abs(float(tw) - table) < 0.03:
                tz = float(tw)
            again = straw_find(cam, tz, near=np.asarray(c, dtype=float))
            if again is not None:
                c, yaw = again[1], again[2]
                hw = float(np.percentile(again[6][:, 2], 95)) - float(tz)
                r_fw = straw_width_r(cam, c, yaw, tz, hw)
                report["r_straw_height"] = round(0.5 * hw, 4)
                report["r_straw_width"] = None if r_fw is None else round(r_fw, 4)
                if r_fw is not None:
                    r_straw = r_fw
                elif 0.004 < hw < 0.03:
                    r_straw = 0.5 * hw
                elif 0.004 < float(again[4]) < 0.03:
                    r_straw = 0.5 * float(again[4])
                w_seen = float(again[4])
                report["r_straw"] = round(r_straw, 4)
                report["wrist_straw"] = {"src": again[5], "c": [round(float(v), 4) for v in again[1]], "yaw_deg": round(math.degrees(again[2]), 1),
                                         "len": round(again[3], 3), "width": round(again[4], 3)}
                tip = tip_end(again[6], np.asarray(c, dtype=float), yaw, r_straw)
                if tip is not None:
                    report["tip"] = {"u": [round(float(v), 3) for v in tip[0]], "end": [round(float(v), 4) for v in tip[1]], "slopes": tip[2]}
        report["table_wrist"] = round(tz, 4)
    except MotionError as error:
        report["hover_error"] = str(error)[:160]
    for k in range(2):
        try:
            plan = api.top_grasp(side, center=(float(c[0]), float(c[1]), tz + r_straw), top_z=tz + 2.0 * r_straw, bottom_z=tz, yaw=yaw,
                                 tip_clearance=0.002, contact_width=2.0 * r_straw, lift=0.06, above=0.06, limits=fast,
                                 settle_after_close=20)
            plan.send(depth_next=True, label=f"pick_straw_{k}")
            picked = bool(api.holding(side))
            report.setdefault("tries", []).append(picked)
            if picked:
                break
            o = api.motion()
            o.gripper(side, "open", rows=12)
            o.send(depth_next=True, label="reopen")
            cam = f"hand_{side}"
            if api.has_depth(cam):
                again = straw_find(cam, tz, near=np.asarray(c, dtype=float))
                if again is not None:
                    c, yaw = again[1], again[2]
                    report.setdefault("refind", []).append([round(float(v), 4) for v in again[1]] + [round(math.degrees(again[2]), 1)])
                    tip2 = tip_end(again[6], np.asarray(c, dtype=float), yaw, r_straw)
                    if tip2 is not None:
                        tip = tip2
        except MotionError as error:
            report.setdefault("plan_errors", []).append(str(error)[:120])
            break
report["holding"] = picked
if picked:
    api.stage_done()


def cyl_axis(pts):
    best_ = None
    for ya in np.arange(-0.004, 0.00401, 0.0002):
        dy = pts[:, 1] - ya
        ok = np.abs(dy) < r_straw * 0.95
        if int(ok.sum()) < 30:
            continue
        zs = pts[ok, 2] + np.sqrt(r_straw ** 2 - dy[ok] ** 2)
        za = float(np.median(zs))
        cost = float(np.median(np.abs(zs - za)))
        if best_ is None or cost < best_[2]:
            best_ = (float(ya), za, cost, int(ok.sum()))
    return best_


def held_straw_axis(cam_name, G_, d_ref):
    view = api.camera(cam_name)
    img = view.rgb.astype(float)
    mx = img.max(axis=2)
    sat = mx - img.min(axis=2)
    P_ = view.points_from_mask((sat < 40) & (mx > 150), stride=2)
    if P_.shape[0] < 30:
        return None
    q_ = (P_ - G_[:3, 3]) @ G_[:3, :3]
    sel = q_[(np.abs(q_[:, 1]) < 0.012) & (q_[:, 2] > -0.03) & (q_[:, 2] < 0.04) & (np.abs(q_[:, 0]) > 0.015) & (np.abs(q_[:, 0]) < 0.09)]
    if sel.shape[0] < 40:
        return None
    segs = []
    for lo, hi in ((0.015, 0.09), (-0.09, -0.015)):
        s_ = sel[(sel[:, 0] > lo) & (sel[:, 0] < hi)]
        if s_.shape[0] < 40:
            continue
        f_ = cyl_axis(s_)
        if f_ is None or f_[2] > 0.002:
            continue
        segs.append((np.array([float(np.median(s_[:, 0])), f_[0], f_[1]]), f_[2], f_[3]))
    if not segs:
        return None
    if len(segs) == 2:
        dv = segs[0][0] - segs[1][0]
        dv = dv / float(np.linalg.norm(dv))
        if float(dv @ d_ref) < 0.0:
            dv = -dv
        c_ = 0.5 * (segs[0][0] + segs[1][0])
    else:
        dv = np.asarray(d_ref, dtype=float)
        c_ = segs[0][0]
    return {"c": c_, "d": dv, "n": int(sel.shape[0]), "segs": len(segs), "mad_mm": float(1000 * max(sg[1] for sg in segs))}


def head_servo(hole_guess, tip_z):
    view = api.camera("head")
    if view.model is None or view.T_base_optical is None or view.depth_m is None:
        return None
    out = {}
    H_, W_ = view.rgb.shape[:2]
    vv, uu = np.mgrid[0:H_, 0:W_]
    rays = view.model.pixel_rays(uu.ravel().astype(float), vv.ravel().astype(float))
    Rb = view.T_base_optical[:3, :3]
    ob = view.T_base_optical[:3, 3]
    dirs = rays @ Rb.T
    dz = dirs[:, 2]
    okz = np.abs(dz) > 1e-6
    zp_ = float(hole_guess[2])
    t_ = np.where(okz, (zp_ - ob[2]) / np.where(okz, dz, 1.0), -1.0)
    P_ = ob + dirs * t_[:, None]
    rgb = view.rgb.reshape(-1, 3).astype(float)
    lum = rgb.mean(axis=1)
    dep = view.depth_m.ravel()
    near_plane = np.isfinite(dep) & (dep > 0) & (dep < t_ - 0.01)
    dark = (t_ > 0.05) & (np.hypot(P_[:, 0] - hole_guess[0], P_[:, 1] - hole_guess[1]) < 0.020) & (lum < 70.0) & ~near_plane
    out["hole_n"] = int(dark.sum())
    if out["hole_n"] >= 20:
        out["hole"] = np.median(P_[dark][:, :2], axis=0)
    mx = rgb.max(axis=1)
    sat = mx - rgb.min(axis=1)
    white = (sat < 40) & (mx > 150) & np.isfinite(dep) & (dep > 0)
    Q = ob + dirs[white] * dep[white][:, None]
    sel = Q[(Q[:, 2] > tip_z + 0.003) & (Q[:, 2] < tip_z + 0.035) & (np.hypot(Q[:, 0] - hole_guess[0], Q[:, 1] - hole_guess[1]) < 0.015)]
    out["straw_n"] = int(sel.shape[0])
    if sel.shape[0] >= 30:
        med = np.median(sel[:, :2], axis=0)
        away = med - ob[:2]
        away = away / max(1e-6, float(np.linalg.norm(away)))
        out["straw"] = med + 0.785 * r_straw * away
    return out


if picked and tip is not None and hole is not None and "xy" in hole:
    G0 = api.grasp_pose(side)
    R0 = G0[:3, :3]
    u_tip3 = np.array([float(tip[0][0]), float(tip[0][1]), 0.0])
    d_g = R0.T @ u_tip3
    d_g = d_g / float(np.linalg.norm(d_g))
    g_close = G0[:3, 3] - np.array([0.0, 0.0, 0.06])
    c3 = np.array([float(c[0]), float(c[1]), tz + r_straw])
    q = R0.T @ (c3 - g_close)
    ero = min(max(r_straw - 0.5 * w_seen, 0.0), 0.006) if (w_seen is not None and r_straw is not None) else 0.0
    d_tip = float((np.asarray(tip[1], dtype=float) - np.asarray(c, dtype=float)) @ np.asarray(tip[0], dtype=float)) + ero
    report["tip_erosion"] = round(ero, 4)
    a_vec = q + d_tip * d_g
    report["in_hand_geom"] = {"q": [round(float(v), 4) for v in q], "d": [round(float(v), 3) for v in d_g], "d_tip": round(d_tip, 4),
                              "a": [round(float(v), 4) for v in a_vec]}
    alt = None
    wcam = f"hand_{side}"
    if api.has_depth(wcam):
        ih = held_straw_axis(wcam, G0, d_g)
        if ih is not None:
            a_v = ih["c"] + ih["d"] * float((a_vec - ih["c"]) @ ih["d"])
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(ih["d"] @ d_g)))))
            report["in_hand_view"] = {"a": [round(float(v), 4) for v in a_v], "d": [round(float(v), 3) for v in ih["d"]], "ang_deg": round(ang, 2),
                                      "n": ih["n"], "segs": ih["segs"], "mad_mm": round(ih["mad_mm"], 2)}
            if ang < 6.0 and float(np.linalg.norm(a_v - a_vec)) < 0.010:
                alt = (a_vec, d_g)
                a_vec, d_g = a_v, ih["d"]
                report["in_hand_used"] = "view"
    H3 = np.array([float(hole["xy"][0]), float(hole["xy"][1]), float(hole["z"])])

    def rot_for(phi, dg):
        b1 = np.asarray(dg, dtype=float)
        b2 = np.array([0.0, 0.0, 1.0]) - b1 * float(b1[2])
        b2 = b2 / float(np.linalg.norm(b2))
        B = np.column_stack([b1, b2, np.cross(b1, b2)])
        w1 = np.array([0.0, 0.0, -1.0])
        w2 = np.array([math.cos(phi), math.sin(phi), 0.0])
        return np.column_stack([w1, w2, np.cross(w1, w2)]) @ B.T

    def grasp_for_tip(p_tip, R_, av):
        T = np.eye(4)
        T[:3, :3] = R_
        T[:3, 3] = np.asarray(p_tip, dtype=float) - R_ @ np.asarray(av, dtype=float)
        return T

    away = np.asarray(c, dtype=float) - np.asarray(cup.mid[:2], dtype=float)
    away = away / max(1e-6, float(np.linalg.norm(away)))
    up_xy = np.asarray(c, dtype=float) + 0.05 * away

    sgn_side = 1.0 if side == "left" else -1.0
    phis = [0.0, -0.5 * sgn_side, -1.0 * sgn_side, 0.5 * sgn_side, -1.4 * sgn_side]
    waist0 = np.array(api.robot_state()["waist"], dtype=float)
    torsos = [None]
    if waist0.size == 5:
        for lean in (0.12, 0.24):
            w = waist0.copy()
            w[2] += lean
            torsos.append(w)
    up = np.array([0.0, 0.0, 1.0])
    lifted = G0.copy()
    lifted[2, 3] += 0.12
    up_tip = np.array([float(up_xy[0]), float(up_xy[1]), float(H3[2]) + 0.10])
    JL = Limits(joint_velocity=3.5, joint_acceleration=14.0)
    tol_saved = (api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad)
    chosen = None
    for tol in ((0.0004, math.radians(0.3)), tol_saved):
        api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = tol
        for w in torsos:
            for phi in phis:
                R_ = rot_for(phi, d_g)
                try:
                    sc = api.motion()
                    if w is not None:
                        sc.waist(w)
                    sc.tcp(side, lifted, linear=True, limits=Limits(linear_velocity=0.35))
                    sc.tcp(side, grasp_for_tip(up_tip, R_, a_vec), limits=JL)
                    sc.tcp(side, grasp_for_tip(H3 + 0.015 * up, R_, a_vec), linear=True, limits=Limits(linear_velocity=0.35))
                    sc.tcp(side, grasp_for_tip(H3 - 0.015 * up, R_, a_vec), linear=True, limits=Limits(linear_velocity=0.08))
                    chosen = (w, phi, R_, tol)
                    break
                except MotionError as error:
                    report.setdefault("plan_errors_insert", []).append(f"{tol[0]:.4f} {phi:.1f}: {str(error)[:80]}")
            if chosen is not None:
                break
        if chosen is not None:
            break
    if chosen is not None:
        w, phi, R_, tol = chosen
        api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = tol
        report["insert_plan"] = {"phi": round(phi, 2), "waist": None if w is None else [round(float(v), 3) for v in w], "tol_mm": round(1000 * tol[0], 2),
                                 "hole": [round(float(v), 4) for v in H3], "a": [round(float(v), 4) for v in a_vec],
                                 "alt": None if alt is None else [round(float(v), 4) for v in alt[0]], "up_xy": [round(float(v), 4) for v in up_xy]}
        tries_log = []
        try:
            m = api.motion()
            if w is not None:
                m.waist(w)
            m.tcp(side, lifted, linear=True, limits=Limits(linear_velocity=0.35))
            m.tcp(side, grasp_for_tip(up_tip, R_, a_vec), limits=JL)
            m.tcp(side, grasp_for_tip(H3 + 0.02 * up, R_, a_vec), linear=True, limits=Limits(linear_velocity=0.35))
            m.hold(4)
            m.send(depth_next=True, label="pre_insert")
            sv = head_servo(H3, float(H3[2]) + 0.02)
            corr = np.zeros(3)
            log0 = {"try": 0}
            if sv is not None:
                log0.update({"hole_n": sv.get("hole_n"), "straw_n": sv.get("straw_n")})
                if "hole" in sv:
                    log0["hole"] = [round(float(v), 4) for v in sv["hole"]]
                if "straw" in sv:
                    log0["straw"] = [round(float(v), 4) for v in sv["straw"]]
                if "hole" in sv and "straw" in sv:
                    d_ = np.asarray(sv["hole"], dtype=float) - np.asarray(sv["straw"], dtype=float)
                    log0["d_mm"] = [round(1000 * float(v), 1) for v in d_]
                    if float(np.linalg.norm(d_)) < 0.008:
                        corr = np.array([float(d_[0]), float(d_[1]), 0.0])
                        log0["used"] = True
            tgt0 = H3 + corr
            m = api.motion()
            m.tcp(side, grasp_for_tip(tgt0 + 0.02 * up, R_, a_vec), linear=True, limits=Limits(linear_velocity=0.06))
            m.hold(2)
            m.tcp(side, grasp_for_tip(tgt0 - 0.015 * up, R_, a_vec), linear=True, limits=Limits(linear_velocity=0.08))
            m.hold(SETTLE_ROWS)
            log0["rows"] = int(m.rows)
            tries_log.append(log0)
            m.send(depth_next=True, label="insert0")
            a_dir = np.array([math.cos(phi), math.sin(phi), 0.0])
            retries = [("shift_p", a_vec, d_g, 0.0015 * np.array([-a_dir[1], a_dir[0], 0.0]))]
            if alt is not None and float(np.linalg.norm(np.asarray(alt[0]) - a_vec)) > 0.0015:
                retries.append(("alt", alt[0], alt[1], np.zeros(3)))
            else:
                retries.append(("shift", a_vec, d_g, 0.0015 * a_dir))
            for k_try, (kind, av2, dg2, sh) in enumerate(retries, start=1):
                if api.rows_left() < 130:
                    tries_log.append({"skip": kind, "rows_left": int(api.rows_left())})
                    break
                R2 = rot_for(phi, dg2)
                tgt = tgt0 + sh
                m = api.motion()
                m.tcp(side, grasp_for_tip(tgt + 0.012 * up, R2, av2), linear=True, limits=Limits(linear_velocity=0.12))
                m.hold(3)
                m.tcp(side, grasp_for_tip(tgt - 0.015 * up, R2, av2), linear=True, limits=Limits(linear_velocity=0.08))
                m.hold(SETTLE_ROWS)
                tries_log.append({"try": k_try, "kind": kind, "rows": int(m.rows)})
                m.send(depth_next=True, label=f"insert{k_try}")
        except MotionError as error:
            tries_log.append({"error": str(error)[:120]})
        report["insert_tries"] = tries_log
        api.stage_done()
    api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = tol_saved
report["budget"] = api.budget()["phases"]
api.end_episode()
