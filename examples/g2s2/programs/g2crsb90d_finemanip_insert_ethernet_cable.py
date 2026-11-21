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
GRASP_BACK = 0.008
SETTLE_ROWS = int(round(2.0 / float(api.task_info()["row_dt_s"])))


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
    ext = {}
    for axv, nm in ((u, "u"), (v, "v")):
        s_ = xy @ axv
        ext[nm] = (float(np.percentile(s_, 0.75)), float(np.percentile(s_, 99.25)))
    faces = []
    for n, nm, other, sg in ((u, "u", "v", 1.0), (-u, "u", "v", -1.0), (v, "v", "u", 1.0), (-v, "v", "u", -1.0)):
        if float(n[0]) >= -0.2:
            continue
        off = ext[nm][1] if sg > 0 else -ext[nm][0]
        t = np.array([float(n[1]), -float(n[0])])
        oth = u if other == "u" else v
        lo_o, hi_o = ext[other]
        t_lo, t_hi = (lo_o, hi_o) if float(t @ oth) > 0.0 else (-hi_o, -lo_o)
        faces.append({"n": n, "off": off, "t": t, "t_lo": t_lo, "t_hi": t_hi})
    if not faces:
        return None
    want = 1.0 if arm_side == "left" else -1.0
    faces.sort(key=lambda f_: -want * float(f_["n"][1]))
    c = u * 0.5 * (ext["u"][0] + ext["u"][1]) + v * 0.5 * (ext["v"][0] + ext["v"][1])
    return {"centre": c, "yaw": a, "top": top, "faces": faces}


rp = router_port(router, side) if router is not None else None
if rp is not None:
    report["router_rect"] = {"centre": [round(float(v), 4) for v in rp["centre"]], "yaw_deg": round(math.degrees(rp["yaw"]), 1),
                             "top": round(rp["top"], 4),
                             "faces": [{"n_deg": round(math.degrees(math.atan2(float(f_["n"][1]), float(f_["n"][0]))), 1),
                                        "off": round(f_["off"], 4), "t": [round(f_["t_lo"], 4), round(f_["t_hi"], 4)]} for f_ in rp["faces"]]}


def port_on_face(cam_name, face, z_lo, z_hi):
    view = api.camera(cam_name)
    if view.model is None or view.T_base_optical is None or view.depth_m is None:
        return None
    H_, W_ = view.depth_m.shape
    vv, uu = np.mgrid[0:H_:2, 0:W_:2]
    rays = view.model.pixel_rays(uu.ravel().astype(float), vv.ravel().astype(float))
    Rb = view.T_base_optical[:3, :3]
    ob = view.T_base_optical[:3, 3]
    dirs = rays @ Rb.T
    n = np.asarray(face["n"], dtype=float)
    t = np.asarray(face["t"], dtype=float)
    dn = dirs[:, 0] * n[0] + dirs[:, 1] * n[1]
    okn = np.abs(dn) > 1e-6
    lam = np.where(okn, (float(face["off"]) - float(ob[:2] @ n)) / np.where(okn, dn, 1.0), -1.0)
    Pp = ob + dirs * lam[:, None]
    tt = Pp[:, :2] @ t
    win = (lam > 0.03) & (tt > face["t_lo"] + 0.008) & (tt < face["t_hi"] - 0.008) & (Pp[:, 2] > z_lo + 0.003) & (Pp[:, 2] < z_hi - 0.003)
    dep = view.depth_m[vv.ravel(), uu.ravel()]
    good = np.isfinite(dep) & (dep > 0)
    facep = win & good & (np.abs(dep - lam) < 0.004)
    hole = win & ((good & (dep > lam + 0.006)) | ~good)
    out = {"face_px": int(facep.sum()), "hole_px": int(hole.sum())}
    if out["face_px"] < 300 or out["hole_px"] < 25:
        return out
    ht, hz = tt[hole], Pp[hole, 2]
    ft, fz = tt[facep], Pp[facep, 2]
    hd = dirs[hole]
    keys = np.floor(ht / 0.004).astype(int) * 100000 + np.floor(hz / 0.004).astype(int)
    vals, cnts = np.unique(keys, return_counts=True)
    tried = []
    for idx in np.argsort(-cnts)[:12]:
        if int(cnts[idx]) < 3:
            break
        kb = int(vals[idx])
        ct = (kb // 100000) * 0.004 + 0.002
        cz = (kb % 100000) * 0.004 + 0.002
        if any(abs(ct - r_["t"]) < 0.012 and abs(cz - r_["z_raw"]) < 0.012 for r_ in tried):
            continue
        sel = (np.abs(ht - ct) < 0.012) & (np.abs(hz - cz) < 0.012)
        if int(sel.sum()) < 25:
            continue
        for it_ in range(2):
            ct, cz = float(np.median(ht[sel])), float(np.median(hz[sel]))
            sel = (np.abs(ht - ct) < 0.020) & (np.abs(hz - cz) < 0.015)
        wt = float(np.percentile(ht[sel], 95) - np.percentile(ht[sel], 5))
        wz = float(np.percentile(hz[sel], 95) - np.percentile(hz[sel], 5))
        band = np.abs(fz - cz) < max(0.5 * wz, 0.003)
        left = int(np.count_nonzero(band & (ft < ct - 0.5 * wt - 0.002) & (ft > ct - 0.5 * wt - 0.025)))
        right = int(np.count_nonzero(band & (ft > ct + 0.5 * wt + 0.002) & (ft < ct + 0.5 * wt + 0.025)))
        above = int(np.count_nonzero((np.abs(ft - ct) < 0.5 * wt) & (fz > cz + 0.5 * wz + 0.002) & (fz < cz + 0.5 * wz + 0.020)))
        dm = hd[sel].mean(axis=0)
        horiz = max(float(-(dm[0] * n[0] + dm[1] * n[1])), 1e-3)
        tan_a = min(max(0.0, float(-dm[2]) / horiz), 2.0)
        cz_c = cz - 0.5 * 0.006 * tan_a
        rec = {"t": round(ct, 4), "z": round(cz_c, 4), "z_raw": round(cz, 4), "w": round(wt, 4), "h": round(wz, 4),
               "n_sel": int(sel.sum()), "lra": [left, right, above]}
        tried.append(rec)
        if not (0.006 < wt < 0.040 and 0.004 < wz < 0.030):
            continue
        if min(left, right) < 15 or above < 10:
            continue
        out.update(rec)
        out["cells_tried"] = len(tried)
        p_ = n * float(face["off"]) + t * ct
        out["port"] = np.array([float(p_[0]), float(p_[1]), cz_c])
        return out
    out["tried"] = tried[:4]
    return out


port_meas = None
push_dir = None
if rp is not None:
    z_lo_r, z_hi_r = float(table), float(rp["top"])
    looks = []
    for face in rp["faces"][:2]:
        n = np.asarray(face["n"], dtype=float)
        t = np.asarray(face["t"], dtype=float)
        tc = 0.5 * (face["t_lo"] + face["t_hi"])
        fp = np.array([n[0] * face["off"] + t[0] * tc, n[1] * face["off"] + t[1] * tc, 0.5 * (z_lo_r + z_hi_r)])
        sent = False
        for tilt_ in (0.6, 0.9, 0.3):
            for dist_ in (0.16, 0.20):
                a3 = np.array([-n[0] * math.cos(tilt_), -n[1] * math.cos(tilt_), -math.sin(tilt_)])
                y3 = np.array([t[0], t[1], 0.0])
                x3 = np.cross(y3, a3)
                T_ = np.eye(4)
                T_[:3, :3] = np.column_stack([x3, y3, a3])
                T_[:3, 3] = fp - a3 * dist_
                try:
                    m = api.motion()
                    m.tcp(side, T_, gripper="open", limits=FAST)
                    m.hold(4)
                    m.send(depth_next=True, label="look_port")
                    sent = True
                    break
                except MotionError as error:
                    looks.append({"plan": str(error)[:60]})
            if sent:
                break
        if not sent:
            continue
        found = port_on_face(f"hand_{side}", face, z_lo_r, z_hi_r) if api.has_depth(f"hand_{side}") else None
        looks.append({k: v for k, v in (found or {}).items() if k != "port"})
        if found is not None and "port" in found:
            port_meas = found["port"]
            push_dir = -n
            break
    report["port_looks"] = looks
    report["port"] = None if port_meas is None else [round(float(v), 4) for v in port_meas]


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
    half = 0.5 * (max(w[0] for w in wide) - min(w[0] for w in wide)) + 0.0025
    return {"xy": centre, "yaw": math.atan2(float(ax[1]), float(ax[0])), "top": top, "bins": len(wide), "ax": ax, "half": half}


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
    hb = float(body_.get("half") or 0.0)
    ring = P[(rr > hb + 0.015) & (rr < hb + 0.055)]
    if ring.shape[0] < 100:
        return None
    floor = float(np.percentile(ring[:, 2], 30))
    raised = (P[:, 2] > floor + 0.002) & (P[:, 2] < floor + 0.012)
    counts = []
    for sg in (1.0, -1.0):
        counts.append(int(np.sum(raised & (sg * s > hb + 0.010) & (sg * s < hb + 0.045) & (np.abs(tt) < 0.012))))
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
plug_w = None
plug_pts = None
lying_raw = None
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
                pw_ = float(best[3].principal_yaw()[2])
                plug_w = pw_ if 0.008 < pw_ < 0.04 else None
                report["plug_width"] = None if plug_w is None else round(plug_w, 4)
                plug_pts = best[3].points
                Pl = api.points(wcam, stride=1)
                Pl = Pl[np.all(np.isfinite(Pl), axis=1)]
                axl = np.asarray(body["ax"], dtype=float)
                rel_l = Pl[:, :2] - np.asarray(body["xy"], dtype=float)
                s_l = rel_l @ axl
                t_l = rel_l @ np.array([-float(axl[1]), float(axl[0])])
                fl_ = float(cs["floor"]) if cs is not None and "floor" in cs else float(table)
                h_l = Pl[:, 2] - fl_
                keep_l = (np.abs(t_l) < 0.005) & (np.abs(s_l) < 0.08) & (h_l > 0.0015) & (h_l < 0.05)
                lying_raw = (s_l[keep_l], h_l[keep_l])
                report["lying_pts"] = int(keep_l.sum())
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
                              bottom_z=wfloor, yaw=body["yaw"], tip_clearance=0.004, contact_width=plug_w, lift=0.06, above=0.05,
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

def height_profile(band_, sx_):
    out = []
    for lo in np.arange(-0.040, 0.050, 0.005):
        sel = band_[(sx_ * band_[:, 0] >= lo) & (sx_ * band_[:, 0] < lo + 0.005)]
        if sel.shape[0] >= 5:
            out.append([round(float(lo), 3), int(sel.shape[0])] + [round(float(np.percentile(sel[:, 2], q)), 4) for q in (10, 50, 90)])
    return out


def in_hand_front(cam_name, side_, sx_, half_h):
    if not api.has_depth(cam_name):
        return None
    P_ = api.points(cam_name, stride=2)
    P_ = P_[np.all(np.isfinite(P_), axis=1)]
    G_ = api.grasp_pose(side_)
    q_ = (P_ - G_[:3, 3]) @ G_[:3, :3]
    band_ = q_[(np.abs(q_[:, 1]) < 0.005) & (q_[:, 2] > -0.03) & (q_[:, 2] < 0.03)]
    prof_ = height_profile(band_, sx_)
    rows_ = [r for r in prof_ if r[1] >= 5]
    if len(rows_) < 4:
        return {"profile": prof_}
    front = max(r[0] for r in rows_) + 0.005
    near_ = [r for r in rows_ if r[0] >= front - 0.0175]
    top_ = float(np.median([r[2] for r in near_]))
    a_g = np.array([sx_ * front, 0.0, top_ + half_h])
    ok_ = 0.005 < front < 0.07 and abs(float(a_g[2])) < 0.03
    return {"profile": prof_, "a": a_g, "ok": ok_, "fit": [round(front, 4), round(top_, 4)]}


def fit_in_hand(cam_name, side_, sx_, lraw, fsign_, front_end):
    if lraw is None or lraw[0].size < 40 or not api.has_depth(cam_name):
        return None
    P_ = api.points(cam_name, stride=2)
    P_ = P_[np.all(np.isfinite(P_), axis=1)]
    G_ = api.grasp_pose(side_)
    q_ = (P_ - G_[:3, 3]) @ G_[:3, :3]
    band_ = q_[(np.abs(q_[:, 1]) < 0.005) & (q_[:, 2] > -0.03) & (q_[:, 2] < 0.04)]
    s_ih = sx_ * band_[:, 0]
    sl = fsign_ * lraw[0] + GRASP_BACK
    hl = lraw[1]
    cen, zt = [], []
    for lo in np.arange(-0.035, 0.050, 0.005):
        c_ = float(lo) + 0.0025
        if abs(c_) < 0.022:
            continue
        bi = (s_ih >= lo) & (s_ih < lo + 0.005)
        if int(bi.sum()) < 5:
            continue
        cen.append(c_)
        zt.append(float(np.percentile(band_[bi, 2], 10)))
    if len(cen) < 3 or sum(1 for c_ in cen if c_ > 0.0) < 2:
        return {"bins": len(cen)}
    grid = np.arange(-0.06, 0.0701, 0.0025)
    hg = np.full(grid.shape, np.nan)
    for i_, g_ in enumerate(grid):
        b_ = np.abs(sl - g_) < 0.0025
        if int(b_.sum()) >= 4:
            hg[i_] = float(np.percentile(hl[b_], 90))
    okg = np.isfinite(hg)
    if int(okg.sum()) < 8:
        return {"bins": len(cen), "lying_bins": int(okg.sum())}
    gx, gh = grid[okg], hg[okg]
    cen_a, zt_a = np.array(cen), np.array(zt)
    best = None
    for ds in np.arange(-0.008, 0.0081, 0.001):
        sh = cen_a - ds
        inside = (sh > gx[0]) & (sh < gx[-1])
        if int(inside.sum()) < 3:
            continue
        hv = np.interp(sh[inside], gx, gh)
        A = np.column_stack([np.ones(int(inside.sum())), cen_a[inside]])
        bvec = zt_a[inside] + hv
        sol = np.linalg.lstsq(A, bvec, rcond=None)[0]
        rms = float(np.sqrt(np.mean((bvec - A @ sol) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(ds), float(sol[0]), float(sol[1]), int(inside.sum()))
    if best is None:
        return {"bins": len(cen)}
    rms, ds, U, phi = best[:4]
    tip = (sl > front_end - 0.010) & (sl <= front_end + 0.001)
    h_tip = float(np.median(hl[tip])) if int(tip.sum()) >= 5 else None
    out = {"rms": round(rms, 4), "ds": round(ds, 4), "U": round(U, 4), "phi": round(phi, 3), "n": best[4],
           "h_tip": None if h_tip is None else round(h_tip, 4)}
    if h_tip is None:
        return out
    fe = front_end + ds
    out["a"] = np.array([sx_ * fe, 0.0, U + phi * fe - 0.5 * h_tip])
    out["ok"] = bool(rms < 0.002 and abs(phi) < 0.35 and 0.005 < fe < 0.07 and abs(float(out["a"][2])) < 0.04)
    return out


def roty(angle):
    c_, s_ = math.cos(angle), math.sin(angle)
    return np.array([[c_, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c_]])


if held and rp is not None and fsign is not None and grasp_xy is not None and port_meas is not None:
    G0 = api.grasp_pose(side)
    xg = G0[:3, 0]
    fvec3 = np.array([float(fvec[0]), float(fvec[1]), 0.0])
    front_is_x = float(fvec3 @ xg) > 0.0
    grasp_z_close = float(G0[2, 3]) - 0.06
    floor_ref = float(report["cable"]["floor"]) if isinstance(report.get("cable"), dict) and "floor" in report["cable"] else table
    if plug_pts is not None and plug_pts.shape[0] >= 20:
        fr = (plug_pts[:, :2] - np.asarray(grasp_xy, dtype=float)) @ np.asarray(fvec, dtype=float)
        front_off = float(np.percentile(fr, 98))
        plug_h = float(np.percentile(plug_pts[:, 2], 90)) - float(floor_ref)
    else:
        front_off, plug_h = GRASP_BACK, float(body["top"]) - float(floor_ref)
    half_h = 0.5 * min(max(plug_h, 0.004), 0.04)
    lying_front, h_tip0 = None, None
    if lying_raw is not None and lying_raw[0].size >= 40:
        sl0 = fsign * lying_raw[0] + GRASP_BACK
        lying_front = float(np.percentile(sl0, 99))
        tip0 = sl0 > lying_front - 0.010
        if int(tip0.sum()) >= 5:
            h_tip0 = float(np.median(lying_raw[1][tip0]))
            half_h = 0.5 * min(max(h_tip0, 0.004), 0.04)
        if 0.005 < lying_front < 0.07:
            front_off = lying_front
    report["lying"] = {"front": None if lying_front is None else round(lying_front, 4), "h_tip": None if h_tip0 is None else round(h_tip0, 4)}
    a_x = front_off * (1.0 if front_is_x else -1.0)
    a_z = grasp_z_close - (floor_ref + half_h)
    report["in_hand"] = {"front_is_x": front_is_x, "a_x": round(a_x, 4), "a_z": round(a_z, 4), "grasp_z_close": round(grasp_z_close, 4)}
    d = np.asarray(push_dir, dtype=float)
    theta = math.atan2(float(d[1]), float(d[0]))
    yaw_c = theta - math.pi / 2.0 if front_is_x else theta + math.pi / 2.0
    port = np.asarray(port_meas, dtype=float).copy()
    a_vec = np.array([a_x, 0.0, a_z])
    sx = 1.0 if a_x > 0.0 else -1.0

    R_d = top_down((0.0, 0.0, 0.0), yaw_c)[:3, :3]
    rot = {"R": R_d}

    def grasp_for_front(anchor):
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
            sc.tcp(side, grasp_for_front(port - 0.020 * d3 + up6), limits=FAST)
            sc.tcp(side, grasp_for_front(port - 0.020 * d3), linear=True, limits=Limits(linear_velocity=0.06))
            sc.tcp(side, grasp_for_front(port + 0.028 * d3), linear=True, limits=Limits(linear_velocity=0.06))
            chosen = ("none" if w is None else w)
            break
        except MotionError as error:
            report.setdefault("plan_errors_insert", []).append(str(error)[:100])
    if chosen is not None:
        try:
            m = api.motion()
            if not isinstance(chosen, str):
                m.waist(chosen)
            m.tcp(side, grasp_for_front(port - 0.020 * d3 + up6), limits=Limits(joint_velocity=3.0, joint_acceleration=12.0))
            m.hold(6)
            m.send(depth_next=True, label="pre_insert")
            report["insert_waist"] = None if isinstance(chosen, str) else [round(float(v), 3) for v in chosen]
            wcam = f"hand_{side}"
            if api.has_depth(wcam):
                Pw = api.points(wcam, stride=2)
                Pw = Pw[np.all(np.isfinite(Pw), axis=1)]
                ih = fit_in_hand(wcam, side, sx, lying_raw, fsign, lying_front) if lying_front is not None else None
                if ih is not None:
                    report["fit1"] = {k: v for k, v in ih.items() if k != "a"}
                    if ih.get("ok"):
                        a_vec[0], a_vec[1], a_vec[2] = float(ih["a"][0]), 0.0, float(ih["a"][2])
                        if abs(float(ih["phi"])) > 0.05:
                            rot["R"] = R_d @ roty(sx * math.atan(float(ih["phi"])))
                if ih is None or not ih.get("ok"):
                    ih_old = in_hand_front(wcam, side, sx, half_h)
                    if ih_old is not None:
                        report["plug_profile"] = ih_old.get("profile")
                        if ih_old.get("ok"):
                            a_vec[0], a_vec[1], a_vec[2] = float(ih_old["a"][0]), 0.0, float(ih_old["a"][2])
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
                    ih2 = fit_in_hand(wcam, side, sx, lying_raw, fsign, lying_front) if lying_front is not None else None
                    if ih2 is not None and ih2.get("ok"):
                        a_vec[0], a_vec[1], a_vec[2] = float(ih2["a"][0]), 0.0, float(ih2["a"][2])
                        rot["R"] = R_d @ roty(sx * math.atan(float(ih2["phi"]))) if abs(float(ih2["phi"])) > 0.05 else R_d
                        tries_log.append({"refit": {k: v for k, v in ih2.items() if k != "a"}})
                planned = None
                for attempt_rot in ((rot["R"],) if rot["R"] is R_d else (rot["R"], R_d)):
                    rot["R"] = attempt_rot
                    try:
                        m = api.motion()
                        if k_try == 0:
                            m.tcp(side, grasp_for_front(tgt - 0.020 * d3), linear=True, limits=Limits(linear_velocity=0.06))
                            m.tcp(side, grasp_for_front(tgt + 0.028 * d3), linear=True, limits=Limits(linear_velocity=0.06))
                        else:
                            m.tcp(side, grasp_for_front(tgt - 0.020 * d3), linear=True, limits=Limits(linear_velocity=0.08))
                            m.tcp(side, grasp_for_front(tgt + 0.028 * d3), linear=True, limits=Limits(linear_velocity=0.06))
                        m.hold(SETTLE_ROWS)
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
