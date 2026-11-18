# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

def coin_by_colour(cam_name):
    view = api.camera(cam_name)
    img = view.rgb.astype(float)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = img.max(axis=2)
    sat = mx - img.min(axis=2)
    red = (r > 90) & (r > 1.5 * g) & (r > 1.5 * b)
    if int(red.sum()) < 2000:
        return None
    inside = (np.cumsum(red, axis=1) > 0) & (np.cumsum(red[:, ::-1], axis=1)[:, ::-1] > 0) & \
             (np.cumsum(red, axis=0) > 0) & (np.cumsum(red[::-1, :], axis=0)[::-1, :] > 0)
    k = max(6, int(0.016 * img.shape[1]))
    shrunk = inside.copy()
    for dy, dx in ((k, 0), (-k, 0), (0, k), (0, -k)):
        shrunk &= np.roll(np.roll(inside, dy, axis=0), dx, axis=1)
    cand = shrunk & ~red & (sat < 60) & (mx > 80)
    ys, xs = np.nonzero(cand)
    if ys.size < 20:
        return None
    cell = max(8, int(0.03 * img.shape[1]))
    keys = (ys // cell) * 100000 + (xs // cell)
    vals, counts = np.unique(keys, return_counts=True)
    best = int(vals[int(np.argmax(counts))])
    cy, cx = (best // 100000) * cell + cell / 2.0, (best % 100000) * cell + cell / 2.0
    near = (np.abs(ys - cy) < 2.5 * cell) & (np.abs(xs - cx) < 2.5 * cell)
    mask = np.zeros(cand.shape, dtype=bool)
    mask[ys[near], xs[near]] = True
    pts = view.points_from_mask(mask)
    if pts.shape[0] < 15:
        return None
    c = np.median(pts, axis=0)
    xy = pts[:, :2] - c[:2]
    vals2, vecs2 = np.linalg.eigh(np.cov(xy.T))
    major = vecs2[:, int(np.argmax(vals2))]
    red_near = np.zeros(red.shape, dtype=bool)
    ry, rx = np.nonzero(red)
    sel = (np.abs(ry - cy) < 5 * cell) & (np.abs(rx - cx) < 5 * cell)
    red_near[ry[sel], rx[sel]] = True
    wall = view.points_from_mask(red_near, stride=2)
    walls_top = float(np.percentile(wall[:, 2], 90)) if wall.shape[0] > 50 else None
    return {"center": c, "top": float(np.percentile(pts[:, 2], 95)), "yaw": math.atan2(float(major[1]), float(major[0])),
            "pixels": int(near.sum()), "points": int(pts.shape[0]), "walls_top": walls_top,
            "length": float(np.ptp(xy @ major)), "width": float(np.ptp(xy @ np.array([-major[1], major[0]])))}


FAST = Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30)
seg0 = api.segment_text(0)
side = api.instructed_arm(seg0) or "left"
table = api.table_height("head")
report["side"] = side
report["table_z"] = round(table, 4)
box = api.find_object(["blue piggy bank", "piggy bank", "blue cylinder", "money box"], size=(0.06, 0.25))
report["box"] = None if box is None else box.summary()
exclude = [box.mid] if box is not None else []
TRAY_Q = ["red tray", "red coin box", "red coin tray", "red organizer tray", "red box"]
tray = api.find_object(TRAY_Q, size=(0.12, 0.40), side=side)
if tray is not None and float(min(tray.extent[0], tray.extent[1])) < 0.12:
    tray = None
if tray is None:
    out = 1.0 if side == "left" else -1.0
    for dx, dy, dz in ((0.0, 0.10, 0.08), (-0.06, 0.12, 0.02)):
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
    tray = api.find_object(TRAY_Q, size=(0.12, 0.40), side=side)
    if tray is not None and float(min(tray.extent[0], tray.extent[1])) < 0.12:
        tray = None
report["tray"] = None if tray is None else tray.summary()
pick = {"ok": False, "reason": "no tray"}
if tray is not None:
    hover = top_down((float(tray.mid[0]), float(tray.mid[1]), float(tray.top_z) + 0.14), 0.0)
    try:
        m = api.motion()
        m.tcp(side, hover, gripper="open", free_axis=(0.0, 0.0, 1.0), limits=FAST)
        m.send(depth_next=True, label="hover_tray")
        coin = None
        col = coin_by_colour(f"hand_{side}") if api.has_depth(f"hand_{side}") else None
        if col is not None and float(np.linalg.norm(col["center"][:2] - tray.mid[:2])) < 0.18 and col["length"] < 0.07:
            report["coin_colour"] = {"center": [round(float(v), 4) for v in col["center"]], "top": round(col["top"], 4),
                                     "yaw_deg": round(math.degrees(col["yaw"]), 1), "pixels": col["pixels"],
                                     "walls_top": None if col["walls_top"] is None else round(col["walls_top"], 4),
                                     "length": round(col["length"], 4), "width": round(col["width"], 4)}
            floor = col["walls_top"] if col["walls_top"] is not None and col["walls_top"] < col["top"] - 0.003 else col["top"] - 0.010
            try:
                yaw_c = float(col["yaw"]) + math.pi / 2.0
                cur = api.grasp_pose(side)
                cur_y = math.atan2(float(cur[1, 1]), float(cur[0, 1]))
                if math.cos(yaw_c - cur_y) < 0.0:
                    yaw_c += math.pi
                tips_z = float(floor) - 0.006
                gz = tips_z + 0.0090
                c3 = np.asarray(col["center"], dtype=float)
                plan = api.motion()
                pre = top_down((float(c3[0]), float(c3[1]), gz + 0.06), yaw_c)
                at = top_down((float(c3[0]), float(c3[1]), gz), yaw_c)
                plan.tcp(side, pre, gripper=0.27, limits=FAST)
                plan.tcp(side, at, linear=True, limits=Limits(linear_velocity=0.15))
                plan.hold(6)
                plan.gripper(side, "close", rows=30)
                plan.hold(20)
                plan.tcp(side, offset(at, dz=0.05), linear=True, limits=Limits(linear_velocity=0.15))
                report["pick_rows"] = plan.rows
                report["deep_grasp"] = {"tips_z": round(tips_z, 4), "grasp_z": round(gz, 4), "yaw_deg": round(math.degrees(yaw_c), 1)}
                plan.send(depth_next=True, label="pick_coin")
                pick = {"ok": bool(api.holding(side, closed_below=0.010)), "reason": "wrist colour"}
                again = coin_by_colour(f"hand_{side}") if api.has_depth(f"hand_{side}") else None
                pick["coin_left_in_tray"] = None if again is None else round(float(np.linalg.norm(again["center"][:2] - col["center"][:2])), 4)
                if not pick["ok"] and again is None:
                    pick["ok"] = True
                    pick["by"] = "tray empty"
            except MotionError as error:
                report["colour_pick_error"] = str(error)[:160]
        for q in (() if pick.get("reason") == "wrist colour" else ("silver coin", "coin", "metal coin standing upright")):
            dets = [d for d in api.perceive(q, camera=f"hand_{side}", max_detections=4)
                    if d.score >= 0.05 and d.points.shape[0] >= 15 and float(max(d.extent[0], d.extent[1])) < 0.06
                    and float(np.linalg.norm(d.center[:2] - tray.mid[:2])) < 0.15 and d.top_z > tray.top_z - 0.01]
            if dets:
                coin = max(dets, key=lambda d: d.score)
                break
        report["coin"] = None if coin is None else coin.summary()
        if coin is not None:
            plan = api.top_grasp(side, coin, depth_into=0.010, lift=0.05, above=0.05, limits=FAST)
            report["pick_rows"] = plan.rows
            plan.send(depth_next=True, label="pick_coin")
            pick = {"ok": bool(api.holding(side)), "reason": "wrist"}
    except MotionError as error:
        pick = {"ok": False, "reason": str(error)[:200]}
report["pick"] = pick


def slot_from_wrist(cam_name, centre_xy, lid_top):
    view = api.camera(cam_name)
    if view.model is None or view.T_base_optical is None or view.depth_m is None:
        return None
    H, W = view.depth_m.shape
    vv, uu = np.mgrid[0:H:2, 0:W:2]
    u1 = uu.ravel().astype(float)
    v1 = vv.ravel().astype(float)
    rays = view.model.pixel_rays(u1, v1)
    Rb = view.T_base_optical[:3, :3]
    ob = view.T_base_optical[:3, 3]
    dirs = rays @ Rb.T
    dz = dirs[:, 2]
    okz = np.abs(dz) > 1e-6
    dplane = np.where(okz, (lid_top - ob[2]) / np.where(okz, dz, 1.0), -1.0)
    P = ob + dirs * dplane[:, None]
    dep = view.depth_m[vv.ravel(), uu.ravel()]
    rr = np.hypot(P[:, 0] - centre_xy[0], P[:, 1] - centre_xy[1])
    disc = (dplane > 0.03) & (rr < 0.050)
    hole = disc & ((dep > dplane + 0.02) | (dep <= 0.0) | ~np.isfinite(dep))
    lid = disc & (np.abs(dep - dplane) < 0.008)
    if int(hole.sum()) < 25 or int(lid.sum()) < 200:
        return {"hole": int(hole.sum()), "lid": int(lid.sum())}
    pts = P[hole][:, :2]
    c0 = np.median(pts, axis=0)
    xy = pts - c0
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    ax = vecs[:, int(np.argmax(vals))]
    pp = np.array([-ax[1], ax[0]])
    sa, sp = xy @ ax, xy @ pp
    lo, hi = float(np.percentile(sa, 3)), float(np.percentile(sa, 97))
    cen = c0 + ax * (0.5 * (lo + hi)) + pp * float(np.median(sp))
    return {"centre": cen, "yaw": math.atan2(float(ax[1]), float(ax[0])), "length": hi - lo,
            "width": float(np.percentile(sp, 95) - np.percentile(sp, 5)), "hole": int(hole.sum()), "lid": int(lid.sum())}


def coin_in_hand(cam_name, side_):
    view = api.camera(cam_name)
    img = view.rgb.astype(float)
    mx = img.max(axis=2)
    sat = mx - img.min(axis=2)
    cand = (sat < 45) & (mx > 90)
    G_ = api.grasp_pose(side_)
    P_ = view.points_from_mask(cand, stride=2)
    if P_.shape[0] < 20:
        return None
    q_ = (P_ - G_[:3, 3]) @ G_[:3, :3]
    sel = q_[(np.abs(q_[:, 1]) < 0.003) & (q_[:, 2] > -0.012) & (q_[:, 2] < 0.035) & (np.abs(q_[:, 0]) < 0.03)]
    if sel.shape[0] < 15:
        return None
    lo, hi = float(np.percentile(sel[:, 0], 2)), float(np.percentile(sel[:, 0], 98))
    off = G_[:3, :3] @ np.array([0.5 * (lo + hi), 0.0, 0.0])
    return {"x_mid": 0.5 * (lo + hi), "span": hi - lo, "z_top": float(np.percentile(sel[:, 2], 3)), "n": int(sel.shape[0]),
            "off_x": float(off[0]), "off_y": float(off[1]), "yaw_at": math.atan2(float(G_[1, 1]), float(G_[0, 1]))}


if pick["ok"] and box is not None:
    api.stage_done()
    top = float(box.top_z)
    bxy = np.array([float(box.mid[0]), float(box.mid[1])])
    slot_h = api.find_object(["coin slot", "slot", "black slot on top"], size=(0.01, 0.08),
                             region=lambda c: float(np.linalg.norm(c[:2] - bxy)) < 0.07 and abs(float(c[2]) - top) < 0.03)
    report["slot_head"] = None if slot_h is None else slot_h.summary()
    yaw_s = float(slot_h.principal_yaw()[0]) if slot_h is not None else 0.0
    held_pose = api.grasp_pose(side)
    grasp_y = math.atan2(float(held_pose[1, 1]), float(held_pose[0, 1]))

    def closing_for(slot_yaw):
        cl = slot_yaw + math.pi / 2.0
        if math.cos(cl - grasp_y) < 0.0:
            cl += math.pi
        return cl

    R_COIN = 0.01935
    TILTS = (0.0, -0.26, 0.26, -0.44, 0.44, -0.61, 0.61)

    def coin_pose(cxy, cz, cl_, tilt_, cg_):
        R_ = top_down((0.0, 0.0, 0.0), cl_, tilt=tilt_, tilt_toward=cl_ - math.pi / 2.0)[:3, :3]
        T_ = np.eye(4)
        T_[:3, :3] = R_
        T_[:3, 3] = np.array([float(cxy[0]), float(cxy[1]), float(cz)]) - R_ @ np.asarray(cg_, dtype=float)
        return T_

    cg = np.array([0.0, 0.0, 0.0051])
    waist0 = np.array(api.robot_state()["waist"], dtype=float)
    torsos = [None]
    if waist0.size == 5:
        for lean in (0.15, 0.30, 0.45):
            w = waist0.copy()
            w[2] += lean
            torsos.append(w)
    looked = None
    tilt_used = 0.0
    for w in torsos:
        for tl in TILTS:
            try:
                depth = 0.014 if tl == 0.0 else 0.010
                hov = coin_pose(bxy, top + 0.045 + R_COIN, closing_for(yaw_s), tl, cg)
                sc = api.motion()
                if w is not None:
                    sc.waist(w)
                sc.tcp(side, hov, limits=FAST)
                for cl_try in (closing_for(yaw_s), closing_for(yaw_s + math.pi / 2.0)):
                    sc2 = api.motion()
                    if w is not None:
                        sc2.waist(w)
                    sc2.tcp(side, coin_pose(bxy, top + 0.012 + R_COIN, cl_try, tl, cg), limits=FAST)
                    sc2.tcp(side, coin_pose(bxy, top - depth - 0.002 + R_COIN, cl_try, tl, cg), linear=True,
                            limits=Limits(linear_velocity=0.05))
                m = api.motion()
                if w is not None:
                    m.waist(w)
                m.tcp(side, hov, limits=FAST)
                m.hold(4)
                m.send(depth_next=True, label="over_box")
                looked = "none" if w is None else w
                tilt_used = tl
                report["box_waist"] = None if w is None else [round(float(v), 3) for v in w]
                report["tilt_deg"] = round(math.degrees(tl), 1)
                break
            except MotionError as error:
                report.setdefault("hover_errors", []).append(f"tilt {round(math.degrees(tl))}: {str(error)[:70]}")
        if looked is not None:
            break
    if looked is not None:
        wcam = f"hand_{side}"
        sl = slot_from_wrist(wcam, bxy, top) if api.has_depth(wcam) else None
        report["slot_wrist"] = None if sl is None else {k: (round(float(v), 4) if not isinstance(v, np.ndarray) else [round(float(x), 4) for x in v]) for k, v in sl.items()}
        target = bxy.copy()
        yaw_use = yaw_s
        if sl is not None and "centre" in sl and 0.025 < sl["length"] < 0.060 and sl["width"] < 0.010 and float(np.linalg.norm(sl["centre"] - bxy)) < 0.02:
            target = np.asarray(sl["centre"], dtype=float)
            yaw_use = float(sl["yaw"])
        cl = closing_for(yaw_use)
        ch = coin_in_hand(wcam, side) if api.has_depth(wcam) else None
        report["coin_in_hand"] = None if ch is None else {k: round(float(v), 4) for k, v in ch.items()}
        if ch is not None and 0.028 < ch["span"] < 0.045 and abs(ch["x_mid"]) < 0.012:
            cg[0] = float(ch["x_mid"])
        if ch is not None and 0.0 < ch["z_top"] + 0.0387 < 0.045:
            cg[2] = float(ch["z_top"]) + R_COIN
        report["coin_below"] = round(float(cg[2]) + R_COIN, 4)
        depth = 0.014 if tilt_used == 0.0 else 0.010
        c_in = top - depth + R_COIN
        c_pre = top + 0.012 + R_COIN
        c_lift = top + 0.003 + R_COIN
        wnow = np.array(api.robot_state()["waist"], dtype=float)
        tors2 = [None]
        if wnow.size == 5:
            for lean in (0.12, 0.24, 0.36):
                w2 = wnow.copy()
                w2[2] += lean
                tors2.append(w2)
        sent_ins = False
        z_cmd = None
        tol_saved = (api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad)
        for tol in ((0.0004, math.radians(0.3)), tol_saved):
            if sent_ins:
                break
            api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = tol
            report["ik_tol_mm"] = round(1000 * tol[0], 2)
            for w2 in tors2:
                try:
                    m = api.motion()
                    if w2 is not None:
                        m.waist(w2)
                    pre = coin_pose(target, c_pre, cl, tilt_used, cg)
                    dn = coin_pose(target, c_in, cl, tilt_used, cg)
                    m.tcp(side, pre, linear=w2 is None, limits=Limits(linear_velocity=0.10) if w2 is None else FAST)
                    m.tcp(side, dn, linear=True, limits=Limits(linear_velocity=0.05))
                    m.hold(8)
                    m.send(depth_next=True, label="coin_down")
                    z_cmd = float(dn[2, 3])
                    report["inserted_at"] = [round(float(target[0]), 4), round(float(target[1]), 4), round(c_in, 4), round(math.degrees(cl), 1)]
                    report["insert_waist"] = None if w2 is None else [round(float(v), 3) for v in w2]
                    sent_ins = True
                    break
                except MotionError as error:
                    report.setdefault("insert_errors", []).append(str(error)[:100])
        if sent_ins:
            api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = 0.0004, math.radians(0.3)
            across = np.array([math.cos(cl), math.sin(cl)])
            along = np.array([-math.sin(cl), math.cos(cl)])
            cur = np.array(target, dtype=float)

            def above_target():
                return float(api.grasp_pose(side)[2, 3]) - z_cmd

            def phase_rows():
                for ph in api.budget()["phases"]:
                    if ph.get("phase") == 1:
                        return int(ph.get("rows") or 0)
                return 0

            up_now = above_target()
            tries = [round(1000 * up_now, 1)]
            steps = [(0.0008, 0.0), (-0.0008, 0.0), (0.0016, 0.0), (-0.0016, 0.0), (0.0, 0.0012), (0.0, -0.0012),
                     (0.0024, 0.0), (-0.0024, 0.0)]
            k = 0
            while up_now > 0.004 and k < len(steps) and phase_rows() < 450 - 80 - 45 - 15:
                da, dl = steps[k]
                k += 1
                p = np.array(target, dtype=float) + da * across + dl * along
                try:
                    m = api.motion()
                    m.tcp(side, coin_pose(cur, c_lift, cl, tilt_used, cg), linear=True, limits=Limits(linear_velocity=0.08))
                    m.tcp(side, coin_pose(p, c_lift, cl, tilt_used, cg), linear=True, limits=Limits(linear_velocity=0.04))
                    m.tcp(side, coin_pose(p, c_in, cl, tilt_used, cg), linear=True, limits=Limits(linear_velocity=0.06))
                    m.hold(4)
                    m.send(depth_next=True, label=f"coin_retry{k}")
                    cur = p
                    up_now = above_target()
                    tries.append([round(1000 * da, 1), round(1000 * dl, 1), round(1000 * up_now, 1)])
                except MotionError as error:
                    tries.append(str(error)[:60])
            report["coin_tries"] = tries
            api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = tol_saved
            m = api.motion()
            m.gripper(side, "open", rows=15)
            m.hold(25)
            m.send(depth_next=True, label="coin_release")
            api.stage_done()
        api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = tol_saved
report["budget"] = api.budget()["phases"]
api.end_episode()
