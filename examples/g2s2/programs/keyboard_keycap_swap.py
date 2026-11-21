# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
active = info["active_arms"]
report["embodiment"] = info["embodiment"]
table = api.table_height("head")
report["table_z"] = round(table, 4)
kb = api.find_object(["white keyboard", "numeric keypad"], size=(0.05, 0.40))
report["keyboard"] = None if kb is None else kb.summary()
excl = [kb.mid] if kb is not None else []
queries = ["small gray tool", "gray plastic holder", "gray keycap puller"]
puller = api.find_object(queries, size=(0.02, 0.12), table_z=table, z_tolerance=0.05, exclude=excl, exclude_radius=0.07)
report["puller"] = None if puller is None else puller.summary()

FAST = Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30)
MEAS = {"keytop": None, "pitch": None, "l_puller": None}
GEN_LIFT = 0.08
DEPTH_INTO = 0.025
TIP_OVER_CASE = 0.004
CAMS = {"g2_omnipicker": (240.0, 140.0), "g1_omnipicker": (240.0, 140.0), "agilex": (320.0, 150.0), "arx": (240.0, 110.0)}
TIP = {"g2_omnipicker": 0.0, "g1_omnipicker": 0.0, "agilex": 0.020, "arx": 0.020}.get(report["embodiment"], 0.0)
V_FREE, LOOK_V = CAMS.get(report["embodiment"], (240.0, 140.0))
LAYOUT_U = [(0.0, 3.0, 1, 1), (1.0, 3.0, 1, 1), (2.0, 3.0, 1, 1), (3.0, 3.0, 1, 1),
            (0.0, 2.0, 1, 1), (1.0, 2.0, 1, 1), (2.0, 2.0, 1, 1), (3.0, 1.5, 1, 2),
            (0.0, 1.0, 1, 1), (1.0, 1.0, 1, 1), (2.0, 1.0, 1, 1),
            (0.0, 0.0, 1, 1), (1.0, 0.0, 1, 1), (2.0, 0.0, 1, 1), (3.0, -0.5, 1, 2),
            (0.5, -1.0, 2, 1), (2.0, -1.0, 1, 1)]
KB_CENTRE_U = np.array([1.5, 1.0])
TOP_H, BOT_H, EDGE_U = 0.30, 0.45, 0.6


def template(p):
    tops = []
    for x, y, wx, wy in LAYOUT_U:
        a = (TOP_H + 0.5 * (wx - 1)) * p
        b = (TOP_H + 0.5 * (wy - 1)) * p
        for u in np.arange(-a, a + 1e-9, 0.07 * p):
            for v in np.arange(-b, b + 1e-9, 0.07 * p):
                tops.append((x * p + u, y * p + v))

    def inside_any(px, py, shrink):
        for x, y, wx, wy in LAYOUT_U:
            if abs(px - x * p) < (BOT_H + 0.5 * (wx - 1)) * p - shrink and abs(py - y * p) < (BOT_H + 0.5 * (wy - 1)) * p - shrink:
                return True
        return False

    gaps = []
    for gx in (0.5 * p, 1.5 * p, 2.5 * p):
        for gy in np.arange(-1.45 * p, 3.45 * p, 0.05 * p):
            if not inside_any(gx, gy, 0.04 * p):
                gaps.append((gx, gy))
    for gy in (-0.5 * p, 0.5 * p, 1.5 * p, 2.5 * p):
        for gx in np.arange(-0.45 * p, 3.45 * p, 0.05 * p):
            if not inside_any(gx, gy, 0.04 * p):
                gaps.append((gx, gy))
    edges = []
    for ex in (-EDGE_U * p, (3.0 + EDGE_U) * p):
        for ey in np.arange(-1.45 * p, 3.45 * p, 0.07 * p):
            edges.append((ex, ey))
    for ey in (-(1.0 + EDGE_U) * p, (3.0 + EDGE_U) * p):
        for ex in np.arange(-0.45 * p, 3.45 * p, 0.07 * p):
            edges.append((ex, ey))
    return np.array(tops), np.array(gaps), np.array(edges)


def measure_pitch(view, c0, yaw0, z_top):
    img = view.rgb.astype(float)
    w = img.min(axis=2)
    half, step = 0.11, 0.0005
    B = bev(view, c0, half, step, z_top, w)
    out = []
    ss = np.arange(-0.08, 0.0801, 0.0005)
    tt = np.arange(-0.03, 0.0301, 0.002)
    for th in (yaw0, yaw0 + math.pi / 2.0):
        d = np.array([math.cos(th), math.sin(th)])
        dp = np.array([-d[1], d[0]])
        XY = c0[None, None, :] + ss[:, None, None] * d[None, None, :] + tt[None, :, None] * dp[None, None, :]
        vals = gather(B, c0, half, step, XY)
        prof = np.nanmean(vals, axis=1)
        ok = np.isfinite(prof)
        if int(ok.sum()) < 200:
            continue
        prof = np.where(ok, prof - float(np.nanmean(prof)), 0.0)
        best = None
        for lag in range(30, 51):
            ac = float(np.sum(prof[:-lag] * prof[lag:])) / max(1e-9, float(np.sum(prof * prof)))
            if best is None or ac > best[1]:
                best = (lag * 0.0005, ac)
        out.append(best)
    if not out:
        return None
    pb = max(out, key=lambda t: t[1])
    return pb[0], {"axes": [[round(a_, 4), round(b_, 3)] for a_, b_ in out]}


def look_pose(side, cam, target, drop, yaw):
    G = api.grasp_pose(side)
    V = api.camera(cam)
    T_go = np.linalg.inv(G) @ V.T_base_optical
    d = V.model.pixel_rays(np.array([float(V.model.cx)]), np.array([LOOK_V]))[0]
    d = d / float(np.linalg.norm(d))
    Tg = top_down((0.0, 0.0, 0.0), yaw)
    R = Tg[:3, :3]
    r = R @ (T_go[:3, :3] @ d)
    if float(r[2]) > -0.5:
        return None
    t = drop / (-float(r[2]))
    Tg[:3, 3] = np.asarray(target, dtype=float) - t * r - R @ T_go[:3, 3]
    return Tg


def wrist_yaw(side):
    G = api.grasp_pose(side)
    return math.atan2(float(G[1, 1]), float(G[0, 1]))


def bev(view, centre, half, step, z, chan):
    n = int(round(2 * half / step)) + 1
    xs = centre[0] - half + step * np.arange(n)
    ys = centre[1] - half + step * np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    uv, ok = view.project_base_points(P)
    ok = ok & (uv[:, 1] < V_FREE)
    out = np.full(X.size, np.nan)
    ui = np.clip(np.round(uv[:, 0]).astype(int), 0, chan.shape[1] - 1)
    vi = np.clip(np.round(uv[:, 1]).astype(int), 0, chan.shape[0] - 1)
    out[ok] = chan[vi[ok], ui[ok]]
    return out.reshape(n, n)


def gather(B, centre, half, step, pts):
    n = B.shape[0]
    i = np.round((pts[..., 0] - (centre[0] - half)) / step).astype(int)
    j = np.round((pts[..., 1] - (centre[1] - half)) / step).astype(int)
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    out = np.full(pts.shape[:-1], np.nan)
    out[ok] = B[i[ok], j[ok]]
    return out


def match_keyboard(view, prior_c, prior_yaw, z_top, p):
    img = view.rgb.astype(float)
    w = img.min(axis=2)
    half, step = 0.11, 0.0005
    c0 = np.asarray(prior_c[:2], dtype=float)
    Bt = bev(view, c0, half, step, z_top, w)
    Bg = bev(view, c0, half, step, z_top - 0.5 * MEAS["keytop"], w)
    Be = bev(view, c0, half, step, z_top - MEAS["keytop"], w)
    tops, gaps, edges = template(p)

    def scores(cands):
        out = np.full(len(cands), -1e9)
        for k0 in range(0, len(cands), 600):
            C = cands[k0:k0 + 600]
            c, s = np.cos(C[:, 2]), np.sin(C[:, 2])
            vals = []
            for P, B in ((tops, Bt), (gaps, Bg), (edges, Be)):
                X = C[:, None, 0] + c[:, None] * P[None, :, 0] - s[:, None] * P[None, :, 1]
                Y = C[:, None, 1] + s[:, None] * P[None, :, 0] + c[:, None] * P[None, :, 1]
                vals.append(gather(B, c0, half, step, np.stack([X, Y], axis=-1)))
            vt, vg, ve = vals
            dark = np.concatenate([vg, ve], axis=1)
            nt = np.sum(np.isfinite(vt), axis=1)
            nd = np.sum(np.isfinite(dark), axis=1)
            good = (nt > 0.6 * tops.shape[0]) & (nd > 0.5 * dark.shape[1])
            mt = np.nanmean(vt, axis=1)
            md = np.nanmean(dark, axis=1)
            out[k0:k0 + 600] = np.where(good, mt - md, -1e9)
        return out

    best = None
    per_hyp = []
    for k in range(4):
        y0 = prior_yaw + k * math.pi / 2.0
        o0 = c0 - np.array([[math.cos(y0), -math.sin(y0)], [math.sin(y0), math.cos(y0)]]) @ (KB_CENTRE_U * p)
        dy = np.radians(np.arange(-12.0, 12.01, 2.0))
        dd = np.arange(-0.014, 0.01401, 0.002)
        A, Bx, By = np.meshgrid(dy, dd, dd, indexing="ij")
        cands = np.column_stack([o0[0] + Bx.ravel(), o0[1] + By.ravel(), y0 + A.ravel()])
        sc = scores(cands)
        i = int(np.argmax(sc))
        per_hyp.append(float(sc[i]))
        if best is None or sc[i] > best[0]:
            best = (float(sc[i]), cands[i].copy())
    b = best[1]
    dy = np.radians(np.arange(-2.0, 2.01, 0.25))
    dd = np.arange(-0.002, 0.00201, 0.00025)
    A, Bx, By = np.meshgrid(dy, dd, dd, indexing="ij")
    cands = np.column_stack([b[0] + Bx.ravel(), b[1] + By.ravel(), b[2] + A.ravel()])
    sc = scores(cands)
    i = int(np.argmax(sc))
    o = cands[i]
    per_hyp.sort(reverse=True)
    return {"one": np.array([o[0], o[1]]), "yaw": float(o[2]), "score": float(sc[i]),
            "margin": float(per_hyp[0] - per_hyp[1]), "hyp_scores": [round(v, 1) for v in per_hyp]}


def depth_z(view, target_xy, radius, pct):
    if view.depth_m is None:
        return None
    H, W = view.depth_m.shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:int(V_FREE), :] = True
    P = view.points_from_mask(mask, stride=2)
    if P.shape[0] == 0:
        return None
    r = np.hypot(P[:, 0] - target_xy[0], P[:, 1] - target_xy[1])
    sel = P[r < radius]
    if sel.shape[0] < 30:
        return None
    return float(np.percentile(sel[:, 2], pct))


def disc_z(view, centres, radius):
    if view.depth_m is None:
        return None
    H, W = view.depth_m.shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:int(V_FREE), :] = True
    P = view.points_from_mask(mask, stride=1)
    if P.shape[0] == 0:
        return None
    C = np.asarray(centres, dtype=float).reshape(-1, 2)
    d = np.min(np.hypot(P[:, None, 0] - C[None, :, 0], P[:, None, 1] - C[None, :, 1]), axis=1)
    sel = P[d < radius]
    if sel.shape[0] < 20:
        return None
    return float(np.median(sel[:, 2]))


def cap_check(view, key, vmax=None):
    vlim = V_FREE if vmax is None else float(vmax)
    c, s_ = math.cos(key["yaw"]), math.sin(key["yaw"])
    R2 = np.array([[c, -s_], [s_, c]])
    img = view.rgb.astype(float)
    w = img.min(axis=2)
    def med(kx_, ky_):
        g = np.arange(-0.004, 0.0041, 0.002)
        P = np.array([key["xy"] + R2 @ np.array([kx_ + a, ky_ + b]) for a in g for b in g])
        P3 = np.column_stack([P, np.full(P.shape[0], key["z_case"] + MEAS["keytop"])])
        uv, ok = view.project_base_points(P3)
        ok = ok & (uv[:, 1] < vlim)
        if int(ok.sum()) < 10:
            return None
        u = np.round(uv[ok, 0]).astype(int)
        v = np.round(uv[ok, 1]).astype(int)
        return float(np.median(w[v, u]))
    one = med(0.0, 0.0)
    refs = [m_ for m_ in (med(MEAS["pitch"], 0.0), med(0.0, MEAS["pitch"]), med(MEAS["pitch"], MEAS["pitch"])) if m_ is not None]
    if one is None or len(refs) < 2:
        return {"one": one, "refs": refs, "present": None}
    ratio = one / max(1.0, float(np.median(refs)))
    return {"one": round(one, 1), "refs": [round(v_, 1) for v_ in refs], "ratio": round(ratio, 3),
            "present": True if ratio > 0.8 else (False if ratio < 0.6 else None)}


def ring_z(view, centre, r0, r1):
    if view.depth_m is None:
        return None
    H, W = view.depth_m.shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:int(V_FREE), :] = True
    P = view.points_from_mask(mask, stride=1)
    r = np.hypot(P[:, 0] - centre[0], P[:, 1] - centre[1])
    sel = P[(r >= r0) & (r < r1)]
    if sel.shape[0] < 30:
        return None
    return float(np.median(sel[:, 2]))


def find_handle(view, prior_xy, z_h):
    if view.depth_m is None:
        return {"depth": None}
    H, W = view.depth_m.shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:int(V_FREE), :] = True
    P = view.points_from_mask(mask, stride=1)
    c0 = np.asarray(prior_xy[:2], dtype=float)
    near = P[(np.abs(P[:, 0] - c0[0]) < 0.03) & (np.abs(P[:, 1] - c0[1]) < 0.03) & (np.abs(P[:, 2] - z_h) < 0.03)]
    if near.shape[0] < 100:
        return {"near": int(near.shape[0])}
    zt = float(np.percentile(near[:, 2], 99))
    top = near[near[:, 2] > zt - 0.004]
    c = np.median(top[:, :2], axis=0)
    for it_ in range(3):
        keep = top[np.hypot(top[:, 0] - c[0], top[:, 1] - c[1]) < 0.014]
        if keep.shape[0] < 60:
            return {"near": int(near.shape[0]), "top": int(top.shape[0]), "kept": int(keep.shape[0])}
        c = np.median(keep[:, :2], axis=0)
    X = keep[:, :2] - keep[:, :2].mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(X.T))
    ax = vecs[:, int(np.argmax(vals))]
    bx = np.array([-ax[1], ax[0]])
    pa, pb = X @ ax, X @ bx
    la = float(np.percentile(pa, 98) - np.percentile(pa, 2))
    lb = float(np.percentile(pb, 98) - np.percentile(pb, 2))
    cen = keep[:, :2].mean(axis=0) + ax * 0.5 * float(np.percentile(pa, 98) + np.percentile(pa, 2)) + bx * 0.5 * float(np.percentile(pb, 98) + np.percentile(pb, 2))
    return {"centre": cen, "yaw": math.atan2(float(ax[1]), float(ax[0])), "len": la, "wid": lb, "z_top": float(np.median(keep[:, 2])),
            "cells": int(keep.shape[0])}


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def puller_pick():
    if puller is not None:
        first = api.choose_arm(puller.mid)
        for side in [first] + [s for s in active if s != first]:
            pick = api.pick_object(side, queries,
                                   find={"size": (0.02, 0.12), "table_z": table, "z_tolerance": 0.05, "exclude": excl,
                                         "exclude_radius": 0.07},
                                   grasp={"depth_into": 0.02, "lift": 0.12, "above": 0.08})
            report["pick_" + side] = pick
            if pick["attempts"]:
                if pick["ok"]:
                    api.stage_done()
                break


def pick_and_pull():
    if len(active) > 1:
        side = None if puller is None else api.choose_arm(puller.mid)
        look_side = None if side is None else [s for s in active if s != side][0]
    else:
        side = look_side = active[0]
    start_q = {s: np.array(q, dtype=float) for s, q in api.robot_state()["arm"].items()}
    start_grasp = {s: api.grasp_pose(s).copy() for s in active}
    key = None
    look_kb = None
    if kb is not None and side is not None:
        kb_c = np.array([float(kb.mid[0]), float(kb.mid[1])])
        z_top0 = float(kb.top_z)
        looked = None
        y0 = wrist_yaw(look_side)
        for dy in (0.0, 0.6, -0.6, 1.2, -1.2, math.pi):
            T = look_pose(look_side, "hand_" + look_side, (kb_c[0], kb_c[1], z_top0), 0.15, y0 + dy)
            if T is None:
                continue
            try:
                m = api.motion()
                m.tcp(look_side, T, limits=FAST)
                m.send(depth_next=True, label="look_keyboard")
                looked = round(y0 + dy, 2)
                look_kb = {"target": (kb_c[0], kb_c[1], z_top0), "yaw": y0 + dy}
                break
            except MotionError as error:
                report.setdefault("look_kb_errors", []).append(f"{round(dy, 2)}: {str(error)[:70]}")
        report["look_kb_psi"] = looked
        if looked is not None:
            view = api.camera("hand_" + look_side)
            zt = depth_z(view, kb_c, 0.035, 85)
            report["z_top_depth"] = None if zt is None else round(zt, 4)
            z_top = zt if (zt is not None and abs(zt - z_top0) < 0.02) else z_top0
            prior_yaw = float(kb.principal_yaw()[0]) - math.pi / 2.0
            zc = depth_z(view, kb_c, 0.06, 10)
            MEAS["keytop"] = min(max(z_top - zc, 0.004), 0.03) if zc is not None else None
            mp = measure_pitch(view, kb_c, prior_yaw, z_top)
            MEAS["pitch"] = mp[0] if mp is not None else None
            report["kb_measured"] = {"keytop": None if MEAS["keytop"] is None else round(MEAS["keytop"], 4),
                                     "pitch": None if MEAS["pitch"] is None else round(MEAS["pitch"], 4),
                                     "pitch_info": None if mp is None else mp[1]}
            if MEAS["keytop"] is None or MEAS["pitch"] is None:
                return
            mk = match_keyboard(view, kb_c, prior_yaw, z_top, MEAS["pitch"])
            report["kb_match0"] = {"one": [round(float(v), 4) for v in mk["one"]], "yaw_deg": round(math.degrees(mk["yaw"]), 2),
                                   "score": round(mk["score"], 1), "margin": round(mk["margin"], 1), "hyp": mk["hyp_scores"]}
            Rk = np.array([[math.cos(mk["yaw"]), -math.sin(mk["yaw"])], [math.sin(mk["yaw"]), math.cos(mk["yaw"])]])
            cen1 = [mk["one"] + Rk @ (MEAS["pitch"] * np.array([x, y])) for x, y, wx, wy in LAYOUT_U if wx == 1 and wy == 1]
            z1 = disc_z(view, cen1, 0.003)
            report["z_top_disc"] = None if z1 is None else round(z1, 4)
            if z1 is not None and abs(z1 - z_top) < 0.01:
                z_top = z1
                mk = match_keyboard(view, kb_c, mk["yaw"], z_top, MEAS["pitch"])
            report["kb_match"] = {"one": [round(float(v), 4) for v in mk["one"]], "yaw_deg": round(math.degrees(mk["yaw"]), 2),
                                  "score": round(mk["score"], 1), "margin": round(mk["margin"], 1), "hyp": mk["hyp_scores"],
                                  "z_top": round(z_top, 4)}
            if mk["score"] > 25.0:
                key = {"xy": mk["one"], "yaw": mk["yaw"], "z_case": z_top - MEAS["keytop"]}

    handle = None
    rack = None
    if key is not None:
        p_c = np.array([float(puller.mid[0]), float(puller.mid[1])])
        z_h0 = float(puller.top_z)
        looked = None
        y0 = wrist_yaw(side)
        for dy in (0.0, 0.6, -0.6, 1.2, -1.2, math.pi):
            T = look_pose(side, "hand_" + side, (p_c[0], p_c[1], z_h0), 0.12, y0 + dy)
            if T is None:
                continue
            try:
                m = api.motion()
                if look_side != side:
                    m.tcp({side: T, look_side: start_grasp[look_side]}, limits=FAST)
                else:
                    m.tcp(side, T, gripper="open", limits=FAST)
                m.send(depth_next=True, label="look_puller")
                looked = round(y0 + dy, 2)
                break
            except MotionError as error:
                report.setdefault("look_puller_errors", []).append(f"{round(dy, 2)}: {str(error)[:70]}")
        report["look_puller_psi"] = looked
        if looked is not None:
            view = api.camera("hand_" + side)
            zh = depth_z(view, p_c, 0.020, 97)
            report["z_handle_depth"] = None if zh is None else round(zh, 4)
            z_h = zh if (zh is not None and abs(zh - z_h0) < 0.02) else z_h0
            fh = find_handle(view, p_c, z_h)
            report["handle"] = {k: ([round(float(x), 4) for x in v] if isinstance(v, np.ndarray) else (round(v, 4) if isinstance(v, float) else v))
                                for k, v in fh.items()}
            if "centre" in fh and abs(fh["z_top"] - z_h) < 0.01:
                z_h = fh["z_top"]
            if "centre" in fh:
                Pp = api.points("hand_" + side, stride=2)
                Pp = Pp[np.all(np.isfinite(Pp), axis=1)]
                cxy = np.asarray(fh["centre"], dtype=float)
                fl = Pp[(np.hypot(Pp[:, 0] - cxy[0], Pp[:, 1] - cxy[1]) < 0.045) & (Pp[:, 2] < z_h - 0.03)]
                if fl.shape[0] >= 30:
                    MEAS["l_puller"] = min(max(z_h - float(np.percentile(fl[:, 2], 5)), 0.03), 0.20)
                report["l_puller"] = None if MEAS["l_puller"] is None else round(MEAS["l_puller"], 4)
            if "centre" in fh and MEAS["l_puller"] is not None and fh["len"] > 1.15 * fh["wid"] and 0.008 < fh["len"] < 0.040 and 0.005 < fh["wid"] < 0.025:
                handle = {"xy": fh["centre"], "yaw": fh["yaw"], "z": z_h}
                rack = {"xy": fh["centre"], "yaw": fh["yaw"], "floor_z": z_h - MEAS["l_puller"]}

    held = False
    if handle is not None:
        zg = handle["z"] - DEPTH_INTO
        hx, hy = float(handle["xy"][0]), float(handle["xy"][1])
        cur = api.grasp_pose(side)
        cur_yaw = math.atan2(float(cur[1, 1]), float(cur[0, 1]))
        yaws = sorted([handle["yaw"], wrap(handle["yaw"] + math.pi)], key=lambda a: abs(wrap(a - cur_yaw)))
        for lin in (True, False):
            for yc in yaws:
                try:
                    m = api.motion()
                    m.tcp(side, top_down((hx, hy, zg + 0.06), yc), gripper="open", limits=FAST)
                    if lin:
                        m.tcp(side, top_down((hx, hy, zg), yc), linear=True, limits=Limits(linear_velocity=0.15))
                    else:
                        m.tcp(side, top_down((hx, hy, zg), yc), limits=Limits(joint_velocity=0.8))
                    m.hold(10)
                    m.gripper(side, "close")
                    m.hold(15)
                    if lin:
                        m.tcp(side, top_down((hx, hy, zg + 0.12), yc), linear=True, limits=Limits(linear_velocity=0.25))
                    else:
                        m.tcp(side, top_down((hx, hy, zg + 0.12), yc), limits=Limits(joint_velocity=1.2))
                    m.send(depth_next=True, label="pick_puller")
                    report["grasp"] = {"xy": [round(hx, 4), round(hy, 4)], "z": round(zg, 4), "yaw_deg": round(math.degrees(yc), 1),
                                       "line": lin}
                    break
                except MotionError as error:
                    report.setdefault("grasp_errors", []).append(f"{lin} {round(math.degrees(yc))}: {str(error)[:60]}")
            if "grasp" in report:
                break
        if "grasp" not in report:
            try:
                plan = api.top_grasp(side, center=(hx, hy, handle["z"] - 0.04), top_z=handle["z"], bottom_z=handle["z"] - MEAS["l_puller"],
                                     yaw=handle["yaw"], depth_into=DEPTH_INTO, lift=0.12, above=0.06)
                plan.send(depth_next=True, label="pick_puller_fallback")
                report["fallback"] = "top_grasp on the wrist-measured handle (no pull)"
                if api.holding(side, closed_below=0.008):
                    api.stage_done()
                report["holding"] = api.holding(side, closed_below=0.008)
            except MotionError as error:
                report["fallback"] = "puller pick (no grasp plan at all): " + str(error)[:60]
                puller_pick()
            return
        held = bool(api.holding(side, closed_below=0.010))
        report["holding"] = held
        if held:
            api.stage_done()
    else:
        report["fallback"] = "puller pick (no keyboard / puller from the wrist views)"
        puller_pick()
        return

    if held and key is not None:
        kx, ky = float(key["xy"][0]), float(key["xy"][1])
        z_tips = key["z_case"] + TIP_OVER_CASE
        zg_down = z_tips + MEAS["l_puller"] - DEPTH_INTO
        cur = api.grasp_pose(side)
        cur_yaw = math.atan2(float(cur[1, 1]), float(cur[0, 1]))
        yaws = sorted([key["yaw"], wrap(key["yaw"] + math.pi)], key=lambda a: abs(wrap(a - cur_yaw)))
        w0 = np.array(api.robot_state()["waist"], dtype=float)
        waists = [None]
        if w0.size == 5:
            sgn = -1.0 if ky < float(api.shoulder(side)[1]) else 1.0
            for dyaw in (0.2, 0.35, 0.5):
                w = w0.copy()
                w[4] += sgn * dyaw
                waists.append(w)
        presses = [None]
        def pull_motion(m, w, yc, pr, other=None):
            if other is None:
                m.tcp(side, top_down((kx, ky, zg_down + 0.025), yc), limits=FAST, waist=w)
            else:
                m.tcp({side: top_down((kx, ky, zg_down + 0.025), yc), other[0]: other[1]}, limits=FAST, waist=w)
            if pr is not None:
                m.tcp(look_side, pr[0], gripper="close", limits=FAST)
                m.tcp(look_side, pr[1], linear=True, limits=Limits(linear_velocity=0.05))
                m.hold(6)
            m.hold(4)
            m.tcp(side, top_down((kx, ky, zg_down), yc), linear=True, limits=Limits(linear_velocity=0.04))
            m.hold(10)
            if pr is not None:
                m.tcp(look_side, pr[0], linear=True, limits=Limits(linear_velocity=0.08))
            m.tcp(side, top_down((kx, ky, zg_down + GEN_LIFT), yc), linear=True, limits=Limits(linear_velocity=0.03))
            m.hold(10)
            return m

        plans = [(w, yc, pr) for pr in presses for w in waists for yc in yaws]
        plan_used = None
        for w, yc, pr in plans:
            try:
                m = pull_motion(api.motion(), w, yc, pr)
                m.send(depth_next=True, label="pull_key")
                report["pull"] = {"xy": [round(kx, 4), round(ky, 4)], "z_tips": round(z_tips, 4), "yaw_deg": round(math.degrees(yc), 1),
                                  "torso_yaw": None if w is None else round(float(w[4] - w0[4]), 2),
                                  "press_two": None if pr is None else [pr[2], pr[3]]}
                plan_used = (w, yc, pr)
                break
            except MotionError as error:
                if len(report.get("pull_errors", [])) < 12:
                    report.setdefault("pull_errors", []).append(f"{None if pr is None else pr[2:]} {None if w is None else round(float(w[4] - w0[4]), 2)} {round(math.degrees(yc))}: {str(error)[:50]}")
        if "pull" in report and look_side != side and look_kb is not None:
            ex_ = np.array([math.cos(key["yaw"]), math.sin(key["yaw"])])
            for attempt in (1,):
                cur = api.grasp_pose(side).copy()
                aside = cur.copy()
                aside[:2, 3] -= 0.07 * ex_
                aside[2, 3] += 0.04
                Tl = look_pose(look_side, "hand_" + look_side, look_kb["target"], 0.15, look_kb["yaw"])
                checked = False
                for tgt in (aside, None):
                    try:
                        m = api.motion()
                        if tgt is not None:
                            m.tcp({side: tgt, look_side: Tl}, limits=FAST)
                        else:
                            up = cur.copy()
                            up[2, 3] += 0.04
                            m.tcp(side, up, linear=True, limits=Limits(linear_velocity=0.1))
                            m.tcp(look_side, Tl, limits=FAST)
                        m.send(depth_next=True, label="check_key")
                        checked = True
                        break
                    except MotionError as error:
                        report.setdefault("check_errors", []).append(str(error)[:80])
                if not checked:
                    break
                cc = cap_check(api.camera("hand_" + look_side), key)
                report.setdefault("cap_check", []).append(cc)
                rt = cc.get("ratio")
                if rt is None or rt <= 0.4:
                    break
                w, yc, pr = plan_used
                try:
                    m = pull_motion(api.motion(), w, yc, pr, other=(look_side, start_grasp[look_side]))
                    m.send(depth_next=True, label="pull_key_again")
                    report["pull_again"] = True
                except MotionError as error:
                    report["pull_again"] = str(error)[:80]
        if "pull" in report and look_side == side and plan_used is not None:
            ex_ = np.array([math.cos(key["yaw"]), math.sin(key["yaw"])])
            cur = api.grasp_pose(side).copy()
            checked = False
            for dx_, dz_ in ((0.10, 0.06), (0.0, 0.08)):
                tgt = cur.copy()
                tgt[:2, 3] -= dx_ * ex_
                tgt[2, 3] += dz_
                try:
                    m = api.motion()
                    m.tcp(side, tgt, linear=True, limits=Limits(linear_velocity=0.10))
                    m.send(depth_next=True, label="check_key_head")
                    checked = True
                    break
                except MotionError as error:
                    report.setdefault("check_errors", []).append(str(error)[:80])
            if checked:
                head_view = api.camera("head")
                cc = cap_check(head_view, key, vmax=head_view.rgb.shape[0])
                report.setdefault("cap_check", []).append(dict(cc, camera="head"))
                rt = cc.get("ratio")
                if rt is not None and rt > 0.4:
                    w, yc, pr = plan_used
                    try:
                        m = pull_motion(api.motion(), w, yc, pr)
                        m.send(depth_next=True, label="pull_key_again")
                        report["pull_again"] = "head_check"
                    except MotionError as error:
                        report["pull_again"] = str(error)[:80]
        if "pull" in report:
            api.stage_done()


if report["embodiment"] not in ("g2_omnipicker", "arx"):
    puller_pick()
else:
    try:
        pick_and_pull()
    except EpisodeEnded:
        raise
    except Exception as error:
        report["pick_and_pull_exception"] = str(error)[:200]
        if not report.get("holding"):
            try:
                puller_pick()
            except EpisodeEnded:
                raise
            except Exception as error2:
                report["puller_pick_exception"] = str(error2)[:200]
report["budget"] = api.budget()["phases"]
api.end_episode()
