# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

seg0 = api.segment_text(0)
side = api.instructed_arm(seg0) or "left"
table = api.table_height("head")
report["side"] = side
report["table_z"] = round(table, 4)
router = api.find_object(["router", "network router", "grey box", "gray router box"], size=(0.20, 0.45))
report["router"] = None if router is None else router.summary()
exclude = [router.mid] if router is not None else []
Q = ["ethernet cable plug", "white network cable plug", "rj45 connector", "cable plug"]
FAST = Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30)
PORT_IN, PORT_ALONG, PORT_BELOW_TOP = 0.1599, 0.0944, 0.0241
PLUG_ANCHOR_X, ANCHOR_ABOVE_FLOOR = 0.0293, 0.0082
GRASP_BACK = 0.008


HALF_SIDE = 0.1783


def router_port(det, arm_side):
    view = api.camera(det.camera)
    cam = view.T_base_optical[:3, 3]
    P = api.points(det.camera, mask=np.asarray(det.mask, dtype=bool), stride=1)
    P = P[np.all(np.isfinite(P), axis=1)]
    zz = P[np.abs(P[:, 2] - float(det.mid[2])) < 0.02][:, 2] if P.shape[0] else np.zeros(0)
    if zz.size < 200:
        return None
    top = float(np.median(zz))
    xy = P[P[:, 2] > top - 0.008][:, :2]
    if xy.shape[0] < 150:
        return None
    best = None
    for a_deg in np.arange(0.0, 90.0, 0.25):
        a = math.radians(float(a_deg))
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-math.sin(a), math.cos(a)])
        su, sv = xy @ u, xy @ v
        area = (float(np.percentile(su, 99)) - float(np.percentile(su, 1))) * (float(np.percentile(sv, 99)) - float(np.percentile(sv, 1)))
        if best is None or area < best[0]:
            best = (area, a)
    a = best[1]
    for it in range(2):
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-math.sin(a), math.cos(a)])
        corr = []
        for axv, oth, sgn in ((u, v, -1.0), (v, u, 1.0)):
            s, o = xy @ axv, xy @ oth
            low_near = float(cam[:2] @ axv) < float(np.median(s))
            lo_o, hi_o = float(np.percentile(o, 15)), float(np.percentile(o, 85))
            bs, es = [], []
            b = lo_o
            while b < hi_o:
                sel = (o >= b) & (o < b + 0.005)
                if int(sel.sum()) >= 5:
                    bs.append(b + 0.0025)
                    es.append(float(np.percentile(s[sel], 2)) if low_near else float(np.percentile(s[sel], 98)))
                b += 0.005
            if len(bs) >= 8:
                corr.append(sgn * float(np.polyfit(np.array(bs), np.array(es), 1)[0]))
        if corr:
            a = a + float(np.mean(corr))
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-math.sin(a), math.cos(a)])
    c = np.zeros(2)
    for axv in (u, v):
        s = xy @ axv
        if float(cam[:2] @ axv) < float(np.median(s)):
            c = c + (float(np.percentile(s, 0.75)) + HALF_SIDE) * axv
        else:
            c = c + (float(np.percentile(s, 99.25)) - HALF_SIDE) * axv
    want = 1.0 if arm_side == "left" else -1.0
    faces = [n for n in (u, -u, v, -v) if float(n[0]) < -0.2]
    if not faces:
        return None
    n = max(faces, key=lambda f: want * float(f[1]))
    t = np.array([float(n[1]), -float(n[0])])
    port = c + PORT_IN * n + PORT_ALONG * t
    return {"centre": c, "yaw": a, "sides": (0.0, 0.0), "top": top, "n": n, "t": t,
            "port": np.array([float(port[0]), float(port[1]), top - PORT_BELOW_TOP]), "push": -n}


rp = router_port(router, side) if router is not None else None
if rp is not None:
    report["router_rect"] = {"centre": [round(float(v), 4) for v in rp["centre"]], "yaw_deg": round(math.degrees(rp["yaw"]), 1),
                             "sides": [round(float(v), 4) for v in rp["sides"]], "top": round(rp["top"], 4),
                             "port": [round(float(v), 4) for v in rp["port"]],
                             "push_deg": round(math.degrees(math.atan2(float(rp["push"][1]), float(rp["push"][0]))), 1)}


def plug_body(det):
    P = det.points
    c = np.median(P[:, :2], axis=0)
    xy = P[:, :2] - c
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    ax = vecs[:, int(np.argmax(vals))]
    perp = np.array([-ax[1], ax[0]])
    s, t = xy @ ax, xy @ perp
    lo, hi = float(np.percentile(s, 2)), float(np.percentile(s, 98))
    wide = []
    k = lo
    while k < hi:
        sel = (s >= k) & (s < k + 0.005)
        if int(sel.sum()) >= 4:
            w = float(np.percentile(t[sel], 95) - np.percentile(t[sel], 5))
            if w >= 0.009:
                wide.append((k + 0.0025, float(np.median(t[sel]))))
        k += 0.005
    if not wide:
        return None
    sm = float(np.mean([w[0] for w in wide]))
    tm = float(np.mean([w[1] for w in wide]))
    centre = c + ax * sm + perp * tm
    near = P[np.abs(s - sm) < 0.02]
    top = float(np.percentile(near[:, 2], 95)) if near.shape[0] > 5 else det.top_z
    return {"xy": centre, "yaw": math.atan2(float(ax[1]), float(ax[0])), "top": top, "bins": len(wide), "ax": ax}


def front_dir(det, cam_name, ax):
    view = api.camera(cam_name)
    mask = np.asarray(det.mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if ys.size < 40:
        return None
    img = view.rgb.astype(float)
    sat = img.max(axis=2) - img.min(axis=2)
    pts = np.column_stack([xs, ys]).astype(float)
    cpx = pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov((pts - cpx).T))
    pa = vecs[:, int(np.argmax(vals))]
    proj = (pts - cpx) @ pa
    half_a = proj > 0
    if int(half_a.sum()) < 15 or int((~half_a).sum()) < 15:
        return None
    sa = float(np.mean(sat[ys[half_a], xs[half_a]]))
    sb = float(np.mean(sat[ys[~half_a], xs[~half_a]]))
    ma = np.zeros(mask.shape, dtype=bool)
    ma[ys[half_a], xs[half_a]] = True
    mb = np.zeros(mask.shape, dtype=bool)
    mb[ys[~half_a], xs[~half_a]] = True
    pa3 = view.points_from_mask(ma)
    pb3 = view.points_from_mask(mb)
    if pa3.shape[0] < 5 or pb3.shape[0] < 5:
        return None
    d3 = np.median(pa3[:, :2], axis=0) - np.median(pb3[:, :2], axis=0)
    sign_a = 1.0 if float(d3 @ ax) > 0.0 else -1.0
    front = sign_a if sa > sb else -sign_a
    return {"sign": front, "sat": [round(sa, 1), round(sb, 1)]}


def cable_side(body_, cam_name):
    P = api.points(cam_name, stride=2)
    P = P[np.all(np.isfinite(P), axis=1)]
    c = np.asarray(body_["xy"], dtype=float)
    ax = np.asarray(body_["ax"], dtype=float)
    perp = np.array([-float(ax[1]), float(ax[0])])
    rel = P[:, :2] - c
    s = rel @ ax
    tt = rel @ perp
    rr = np.hypot(s, tt)
    ring = P[(rr > 0.05) & (rr < 0.09)]
    if ring.shape[0] < 100:
        return None
    floor = float(np.percentile(ring[:, 2], 30))
    raised = (P[:, 2] > floor + 0.002) & (P[:, 2] < floor + 0.012)
    counts = []
    for sg in (1.0, -1.0):
        counts.append(int(np.sum(raised & (sg * s > 0.045) & (sg * s < 0.08) & (np.abs(tt) < 0.012))))
    out = {"counts": counts, "floor": round(floor, 4)}
    hi_, lo_ = max(counts), min(counts)
    if hi_ >= 30 and hi_ >= 3 * max(lo_, 1):
        out["sign"] = 1.0 if counts[0] > counts[1] else -1.0
    return out


out = 1.0 if side == "left" else -1.0
for dx, dy, dz in ((0.0, 0.10, 0.06), (-0.06, 0.10, 0.0), (0.0, 0.14, -0.04)):
    try:
        up = api.motion()
        t = api.grasp_pose(side).copy()
        t[0, 3] += dx
        t[1, 3] += out * dy
        t[2, 3] += dz
        up.tcp(side, t)
        up.send(depth_next=True, label="clear_view")
        report["cleared"] = [dx, dy, dz]
        break
    except MotionError as error:
        report["clear_error"] = str(error)[:160]
det = api.find_object(Q, size=(0.008, 0.08), table_z=table, z_tolerance=0.04, exclude=exclude, exclude_radius=0.12, side=side)
body = plug_body(det) if det is not None else None
report["body"] = None if body is None else {"xy": [round(float(v), 4) for v in body["xy"]], "yaw_deg": round(math.degrees(body["yaw"]), 1),
                                             "top": round(body["top"], 4), "bins": body["bins"]}
fsign = None
if body is not None:
    try:
        hv = api.motion()
        hv.tcp(side, top_down((float(body["xy"][0]), float(body["xy"][1]), body["top"] + 0.10), body["yaw"] + math.pi / 2.0),
               gripper="open", free_axis=(0.0, 0.0, 1.0), limits=FAST)
        hv.send(depth_next=True, label="hover")
        wcam = f"hand_{side}"
        if api.has_depth(wcam):
            best = None
            for q in Q:
                for d in api.perceive(q, wcam, max_detections=4):
                    if d.points.shape[0] < 30 or float(np.linalg.norm(d.center[:2] - body["xy"])) > 0.07:
                        continue
                    if abs(float(d.bottom_z) - float(table)) > 0.05:
                        continue
                    if any(float(np.linalg.norm(d.center - api.tcp_pose(sd)[:3, 3])) < 0.05 for sd in ("left", "right")):
                        continue
                    wb = plug_body(d)
                    if wb is not None and (best is None or d.score > best[0]):
                        best = (d.score, wb, q, d)
                if best is not None:
                    break
            if best is not None:
                report["wrist_body"] = {"xy": [round(float(v), 4) for v in best[1]["xy"]], "yaw_deg": round(math.degrees(best[1]["yaw"]), 1),
                                        "top": round(best[1]["top"], 4), "bins": best[1]["bins"], "query": best[2]}
                body = best[1]
                fd = front_dir(best[3], wcam, body["ax"])
                cs = cable_side(body, wcam)
                report["front"] = fd
                report["cable"] = cs
                if cs is not None and "sign" in cs:
                    fsign = -cs["sign"]
                elif fd is not None and max(fd["sat"]) > 1.25 * max(min(fd["sat"]), 1.0):
                    fsign = fd["sign"]
    except MotionError as error:
        report["hover_error"] = str(error)[:160]
held = False
grasp_xy = None
if body is not None:
    fvec = None if fsign is None else fsign * np.asarray(body["ax"], dtype=float)
    gxy = np.asarray(body["xy"], dtype=float) - (GRASP_BACK * fvec if fvec is not None else 0.0)
    report["grasp_xy"] = [round(float(v), 4) for v in gxy]
    wfloor = float(report["cable"]["floor"]) if isinstance(report.get("cable"), dict) and "floor" in report["cable"] else None
    grips = ([("tips", 0.0)] if wfloor is not None else []) + [("depth", 0.008), ("depth", 0.012)]
    for gi, (kind, depth) in enumerate(grips):
        try:
            m = api.motion()
            if gi > 0:
                m.gripper(side, "open", rows=20)
            if kind == "tips":
                api.top_grasp(side, center=(float(gxy[0]), float(gxy[1]), body["top"]), top_z=body["top"],
                              bottom_z=wfloor, yaw=body["yaw"], tip_clearance=0.004, contact_width=0.021, lift=0.06, above=0.05,
                              motion=m, limits=FAST, settle_after_close=30)
            else:
                api.top_grasp(side, center=(float(gxy[0]), float(gxy[1]), body["top"]), top_z=body["top"],
                              bottom_z=table, yaw=body["yaw"], depth_into=depth, floor_clearance=0.010, lift=0.06, above=0.05,
                              motion=m, limits=FAST, settle_after_close=30)
            if m.rows + 8 > api.rows_left() - 30:
                break
            m.send(depth_next=True, label="pick_plug")
        except MotionError as error:
            report.setdefault("plan_errors", []).append(str(error)[:160])
            continue
        held = bool(api.holding(side))
        report.setdefault("tries", []).append([kind, held])
        if held:
            grasp_xy = gxy
            break
        if api.has_depth(f"hand_{side}"):
            again = None
            for q in Q:
                for dd in api.perceive(q, f"hand_{side}", max_detections=4):
                    if dd.points.shape[0] < 30 or float(np.linalg.norm(dd.center[:2] - gxy)) > 0.08:
                        continue
                    if abs(float(dd.bottom_z) - float(table)) > 0.05:
                        continue
                    wb2 = plug_body(dd)
                    if wb2 is not None and (again is None or dd.score > again[0]):
                        again = (dd.score, wb2, dd)
                if again is not None:
                    break
            if again is not None:
                nb = again[1]
                ax2 = np.asarray(nb["ax"], dtype=float)
                if fvec is not None:
                    f2 = ax2 if float(ax2 @ fvec) > 0.0 else -ax2
                    fvec = f2
                    fsign = 1.0 if float(ax2 @ f2) > 0.0 else -1.0
                body = nb
                gxy = np.asarray(nb["xy"], dtype=float) - (GRASP_BACK * fvec if fvec is not None else 0.0)
                report.setdefault("regrasp_xy", []).append([round(float(v), 4) for v in gxy])
if not held and body is None:
    pick = api.pick_object(side, Q, find={"size": (0.008, 0.08), "table_z": table, "z_tolerance": 0.04, "exclude": exclude,
                                          "exclude_radius": 0.12},
                           grasp={"depth_into": 0.008, "lift": 0.08, "floor_clearance": 0.010, "settle_after_close": 18})
    report["pick"] = pick
    held = bool(pick["ok"])
if held:
    api.stage_done()

SEGS = ((-0.0353, -0.0121, 0.0146), (-0.0121, -0.0057, 0.0146), (0.0069, 0.0264, 0.0125 - 0.0016))


def height_profile(band_, sx_):
    out = []
    for lo in np.arange(-0.040, 0.050, 0.005):
        sel = band_[(sx_ * band_[:, 0] >= lo) & (sx_ * band_[:, 0] < lo + 0.005)]
        if sel.shape[0] >= 5:
            out.append([round(float(lo), 3), int(sel.shape[0])] + [round(float(np.percentile(sel[:, 2], q)), 4) for q in (10, 50, 90)])
    return out


def plug_fit(prof_):
    rows_ = [(r[0] + 0.0025, r[2]) for r in prof_ if r[1] >= 5]
    best_ = None
    for u0 in np.arange(-0.040, 0.0101, 0.0005):
        for phi in np.arange(-0.24, 0.2401, 0.02):
            hs = []
            for uc, z in rows_:
                x = uc - u0
                for a, b, h in SEGS:
                    if a <= x < b:
                        hs.append(z + h - x * phi)
                        break
            if len(hs) < 5:
                continue
            hs = np.array(hs)
            hm = float(np.median(hs))
            cost = float(np.sum(np.minimum(np.abs(hs - hm), 0.003))) / len(hs) + 0.0004 * (8 - min(len(hs), 8))
            if best_ is None or cost < best_[3]:
                best_ = (float(u0), hm, float(phi), cost, len(hs))
    return best_


def in_hand_from_view(cam_name, side_, sx_):
    if not api.has_depth(cam_name):
        return None
    P_ = api.points(cam_name, stride=2)
    P_ = P_[np.all(np.isfinite(P_), axis=1)]
    G_ = api.grasp_pose(side_)
    q_ = (P_ - G_[:3, 3]) @ G_[:3, :3]
    band_ = q_[(np.abs(q_[:, 1]) < 0.005) & (q_[:, 2] > -0.03) & (q_[:, 2] < 0.03)]
    prof_ = height_profile(band_, sx_)
    f_ = plug_fit(prof_) if len(prof_) >= 6 else None
    if f_ is None:
        return {"profile": prof_}
    u0, hm, phi, cost, n_ = f_
    a_g = np.array([sx_ * (u0 + PLUG_ANCHOR_X), 0.0, hm - ANCHOR_ABOVE_FLOOR + PLUG_ANCHOR_X * phi])
    a_g[2] += 0.0023 if side_ == "right" else -0.0015
    ok_ = cost < 0.0016 and n_ >= 7 and 0.005 < u0 + PLUG_ANCHOR_X < 0.065 and abs(float(a_g[2])) < 0.03
    return {"profile": prof_, "a": a_g, "phi": phi, "ok": ok_,
            "fit": [round(u0, 4), round(hm, 4), round(math.degrees(phi), 1), round(cost * 1000, 2), n_]}


def roty(angle):
    c_, s_ = math.cos(angle), math.sin(angle)
    return np.array([[c_, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c_]])


if held and rp is not None and fsign is not None and grasp_xy is not None:
    G0 = api.grasp_pose(side)
    xg = G0[:3, 0]
    fvec3 = np.array([float(fvec[0]), float(fvec[1]), 0.0])
    front_is_x = float(fvec3 @ xg) > 0.0
    grasp_z_close = float(G0[2, 3]) - 0.06
    a_x = (PLUG_ANCHOR_X + GRASP_BACK) * (1.0 if front_is_x else -1.0)
    floor_ref = float(report["cable"]["floor"]) if isinstance(report.get("cable"), dict) and "floor" in report["cable"] else table
    a_z = grasp_z_close - (floor_ref + ANCHOR_ABOVE_FLOOR)
    report["in_hand"] = {"front_is_x": front_is_x, "a_x": round(a_x, 4), "a_z": round(a_z, 4), "grasp_z_close": round(grasp_z_close, 4)}
    d = np.asarray(rp["push"], dtype=float)
    theta = math.atan2(float(d[1]), float(d[0]))
    yaw_c = theta - math.pi / 2.0 if front_is_x else theta + math.pi / 2.0
    port = np.asarray(rp["port"], dtype=float).copy()
    a_vec = np.array([a_x, 0.0, a_z])
    sx = 1.0 if a_x > 0.0 else -1.0

    R_d = top_down((0.0, 0.0, 0.0), yaw_c)[:3, :3]
    rot = {"R": R_d}

    def grasp_for_anchor(anchor):
        T = np.eye(4)
        T[:3, :3] = rot["R"]
        T[:3, 3] = np.asarray(anchor, dtype=float) - rot["R"] @ a_vec
        return T

    waist0 = np.array(api.robot_state()["waist"], dtype=float)
    torsos = [None]
    if waist0.size == 5:
        for lean in (0.15, 0.30, 0.45):
            w = waist0.copy()
            w[2] += lean
            torsos.append(w)
    d3 = np.array([d[0], d[1], 0.0])
    up6 = np.array([0.0, 0.0, 0.006])
    chosen = None
    for w in torsos:
        try:
            sc = api.motion()
            if w is not None:
                sc.waist(w)
            sc.tcp(side, grasp_for_anchor(port - 0.020 * d3 + up6), limits=FAST)
            sc.tcp(side, grasp_for_anchor(port - 0.020 * d3), linear=True, limits=Limits(linear_velocity=0.06))
            sc.tcp(side, grasp_for_anchor(port + 0.012 * d3), linear=True, limits=Limits(linear_velocity=0.06))
            chosen = ("none" if w is None else w)
            break
        except MotionError as error:
            report.setdefault("plan_errors_insert", []).append(str(error)[:100])
    if chosen is not None:
        try:
            m = api.motion()
            if not isinstance(chosen, str):
                m.waist(chosen)
            m.tcp(side, grasp_for_anchor(port - 0.020 * d3 + up6), limits=Limits(joint_velocity=3.0, joint_acceleration=12.0))
            m.hold(6)
            m.send(depth_next=True, label="pre_insert")
            report["insert_waist"] = None if isinstance(chosen, str) else [round(float(v), 3) for v in chosen]
            wcam = f"hand_{side}"
            if api.has_depth(wcam):
                Pw = api.points(wcam, stride=2)
                Pw = Pw[np.all(np.isfinite(Pw), axis=1)]
                near = Pw[(np.abs(Pw[:, 2] - rp["top"]) < 0.012) & (np.linalg.norm(Pw[:, :2] - port[:2], axis=1) < 0.12)
                          & ((Pw[:, :2] - port[:2]) @ np.asarray(d, dtype=float) > 0.02)]
                report["wrist_top_pts"] = int(near.shape[0])
                if near.shape[0] > 150:
                    top_w = float(np.median(near[:, 2]))
                    report["wrist_top"] = round(top_w, 4)
                    if abs(top_w - rp["top"]) < 0.010:
                        port[2] = top_w - PORT_BELOW_TOP
                ih = in_hand_from_view(wcam, side, sx)
                if ih is not None:
                    report["plug_profile"] = ih.get("profile")
                    if "fit" in ih:
                        report["fit1"] = ih["fit"] + [bool(ih["ok"])]
                    if ih.get("ok"):
                        a_vec[0], a_vec[1], a_vec[2] = float(ih["a"][0]), 0.0, float(ih["a"][2])
                        report["fit1_phi_deg"] = round(math.degrees(float(ih["phi"])), 1)
            port[2] = float(port[2]) - 0.0006
            report["port_used"] = [round(float(v), 4) for v in port]
            report["a_used"] = [round(float(v), 4) for v in a_vec]
            offsets = (0.0, -0.0024, 0.0024)
            tries_log = []
            for k_try, dz_try in enumerate(offsets):
                tgt = port + np.array([0.0, 0.0, dz_try])
                if k_try > 0:
                    if api.rows_left() < 112:
                        break
                    m = api.motion()
                    m.tcp(side, offset(api.grasp_pose(side), dx=-0.012 * float(d[0]), dy=-0.012 * float(d[1])), linear=True,
                          limits=Limits(linear_velocity=0.10))
                    m.hold(2)
                    m.send(depth_next=True, label=f"back_off{k_try}")
                    ih2 = in_hand_from_view(wcam, side, sx)
                    if ih2 is not None and ih2.get("ok"):
                        a_vec[0], a_vec[1], a_vec[2] = float(ih2["a"][0]), 0.0, float(ih2["a"][2])
                        rot["R"] = R_d
                        tries_log.append({"refit": ih2["fit"]})
                planned = None
                for attempt_rot in ((rot["R"],) if rot["R"] is R_d else (rot["R"], R_d)):
                    rot["R"] = attempt_rot
                    try:
                        m = api.motion()
                        if k_try == 0:
                            m.tcp(side, grasp_for_anchor(tgt - 0.020 * d3), linear=True, limits=Limits(linear_velocity=0.06))
                            m.tcp(side, grasp_for_anchor(tgt + 0.012 * d3), linear=True, limits=Limits(linear_velocity=0.06))
                        else:
                            m.tcp(side, grasp_for_anchor(tgt - 0.020 * d3), linear=True, limits=Limits(linear_velocity=0.08))
                            m.tcp(side, grasp_for_anchor(tgt + 0.012 * d3), linear=True, limits=Limits(linear_velocity=0.06))
                        m.hold(30 if k_try == 0 else 26)
                        planned = m
                        break
                    except MotionError as error:
                        tries_log.append({"plan_error": str(error)[:80]})
                if planned is None:
                    break
                tries_log.append({"try": k_try, "dz": dz_try, "tilt": rot["R"] is not R_d, "a": [round(float(v), 4) for v in a_vec]})
                report["insert_tries"] = tries_log
                planned.send(depth_next=True, label=f"push{k_try}")
            report["insert_tries"] = tries_log
            api.stage_done()
            m = api.motion()
            m.gripper(side, "open", rows=15)
            m.hold(8)
            m.send(depth_next=True, label="release_plug")
        except MotionError as error:
            report["push_error"] = str(error)[:200]
report["budget"] = api.budget()["phases"]
api.end_episode()
