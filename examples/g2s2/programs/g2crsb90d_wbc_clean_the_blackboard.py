# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

table = api.table_height("head")
report["table_z"] = round(table, 4)
ERASER_Q = ["blackboard eraser", "eraser", "board eraser"]


def eraser_width():
    det = api.find_object(ERASER_Q, size=(0.03, 0.20), table_z=table, z_tolerance=0.05)
    return None if det is None else float(det.principal_yaw()[2])


def eraser_wrist_look():
    det = api.find_object(ERASER_Q, size=(0.03, 0.20), table_z=table, z_tolerance=0.05)
    if det is None:
        return None, None, None
    sent = False
    for h_look, dx_l in ((0.15, 0.0), (0.18, 0.0), (0.15, -0.04), (0.20, -0.04), (0.12, 0.0)):
        try:
            m = api.motion()
            m.tcp("right", top_down((float(det.mid[0]) + dx_l, float(det.mid[1]), float(det.top_z) + h_look), 0.0), gripper="open",
                  free_axis=(0.0, 0.0, 1.0))
            m.send(depth_next=True, label="look_eraser")
            sent = True
            break
        except MotionError as error:
            report["look_eraser_error"] = str(error)[:100]
    if not sent:
        return None, None, None
    if not api.has_depth("hand_right"):
        return None, None, None
    P = api.points("hand_right", stride=2)
    P = P[np.all(np.isfinite(P), axis=1)]
    r = np.hypot(P[:, 0] - float(det.mid[0]), P[:, 1] - float(det.mid[1]))
    ring = P[(r > 0.06) & (r < 0.12) & (np.abs(P[:, 2] - float(table)) < 0.05)]
    core = P[(r < 0.025) & (P[:, 2] < float(table) + 0.12)]
    if ring.shape[0] < 50:
        return None, None, None
    desk = float(np.percentile(ring[:, 2], 30))
    h = float(np.percentile(core[:, 2], 95)) - desk if core.shape[0] >= 30 else None
    F = P[(r < 0.08) & (P[:, 2] > desk + 0.01) & (P[:, 2] < desk + 0.10)]
    w = None
    support = None
    if F.shape[0] >= 50:
        cF = np.median(F[:, :2], axis=0)
        xy = F[:, :2] - cF
        ev_, V_ = np.linalg.eigh(np.cov(xy.T))
        u1, u2 = V_[:, int(np.argmax(ev_))], V_[:, int(np.argmin(ev_))]
        s1, s2 = xy @ u1, xy @ u2
        w = float(np.percentile(s2, 98) - np.percentile(s2, 2))
        h1 = 0.5 * float(np.percentile(s1, 98) - np.percentile(s1, 2))
        h2 = 0.5 * w
        q_ = P[:, :2] - cF
        b1 = np.abs(q_ @ u1) - h1
        b2 = np.abs(q_ @ u2) - h2
        bd = np.maximum(b1, b2)
        sr = P[(bd > 0.008) & (bd < 0.025) & (P[:, 2] < desk + 0.05) & (P[:, 2] > desk - 0.02)]
        if sr.shape[0] >= 30:
            support = float(np.median(sr[:, 2]))
    report["eraser_wrist"] = {"desk": round(desk, 4), "h": None if h is None else round(h, 4), "w": None if w is None else round(w, 4),
                              "support": None if support is None else round(support, 4)}
    return h, w, support


def eraser_in_hand():
    if not api.has_depth():
        api.hold(1, depth_next=True, label="observe_eraser")
    G = api.grasp_pose("right")
    best = None
    for q in ERASER_Q:
        for d in api.perceive(q, max_detections=4):
            if d.points.shape[0] < 40 or float(np.linalg.norm(d.mid[:3] - G[:3, 3])) > 0.12:
                continue
            if best is None or d.score > best.score:
                best = d
        if best is not None:
            break
    if best is None:
        return None, None
    qg = (best.points - G[:3, 3]) @ G[:3, :3]
    qg = qg[np.linalg.norm(qg, axis=1) < 0.10]
    if qg.shape[0] < 40:
        return None, None
    return float(np.percentile(qg[:, 2], 97)), float(np.percentile(qg[:, 1], 97) - np.percentile(qg[:, 1], 3))


er_w = eraser_width()
er_hw, er_ww, er_sup = eraser_wrist_look()
pick = api.pick_object("right", ERASER_Q,
                       find={"size": (0.03, 0.20), "table_z": table, "z_tolerance": 0.05},
                       grasp={"depth_into": 0.025, "lift": 0.15})
report["pick"] = pick

STANDOFF = 0.78
SWEET_Y = -0.12
FELT = None
PRESS = 0.006
STEP = None


def board_plane():
    board = api.find_object(["whiteboard", "white board", "writing board"], size=(0.4, 3.0), away_from_grippers=0.0)
    if board is None:
        return None
    pts = board.points
    c = np.median(pts, axis=0)
    A = np.column_stack([pts[:, 1] - c[1], np.ones(len(pts))])
    coef, res, rank, sv = np.linalg.lstsq(A, pts[:, 0] - c[0], rcond=None)
    n = np.array([1.0, -float(coef[0]), 0.0])
    n = n / float(np.linalg.norm(n))
    d = float(np.dot(n, c))
    lo = np.percentile(pts, 5, axis=0)
    hi = np.percentile(pts, 95, axis=0)
    return {"det": board, "n": n, "d": d, "c": c, "lo": lo, "hi": hi}


def dark_strokes(bp2):
    n, d = bp2["n"], bp2["d"]
    lateral = np.array([-n[1], n[0], 0.0])
    view = api.camera("head")
    det = bp2["det"]
    if view.depth_m is None or det.mask is None:
        return None
    lum = view.rgb.astype(float).mean(axis=2)
    mask = det.mask.astype(bool)
    if int(mask.sum()) < 500:
        return None
    board_lum = float(np.median(lum[mask]))
    P = view.points_from_mask(mask & (lum < 0.55 * board_lum), stride=1)
    if P.shape[0] < 50:
        return None
    P = P[(np.abs(P @ n - d) < 0.015) & (P[:, 2] >= float(bp2["lo"][2]) + 0.03)]
    if P.shape[0] < 50:
        return None
    return np.column_stack([P @ lateral, P[:, 2]])


def marked_area(bp2):
    n, d = bp2["n"], bp2["d"]
    lateral = np.array([-n[1], n[0], 0.0])
    ds = dark_strokes(bp2)
    if ds is not None:
        report.setdefault("dark_samples", []).append(int(ds.shape[0]))
    for q in ("marker writing on the whiteboard", "handwriting", "scribbles"):
        dets = [x for x in api.perceive(q, max_detections=4) if x.points.shape[0] > 30 and abs(float(np.dot(n, x.center)) - d) < 0.05]
        if dets:
            allp = np.concatenate([x.points for x in dets])
            box = [float(np.percentile(allp @ lateral, 2)), float(np.percentile(allp @ lateral, 98)),
                   float(np.percentile(allp[:, 2], 2)), float(np.percentile(allp[:, 2], 98))]
            if ds is not None:
                box = [min(box[0], float(np.percentile(ds[:, 0], 1))), max(box[1], float(np.percentile(ds[:, 0], 99))),
                       min(box[2], float(np.percentile(ds[:, 1], 1))), max(box[3], float(np.percentile(ds[:, 1], 99)))]
                return (box[0], box[1], box[2], box[3], q + " + dark strokes")
            return (box[0], box[1], box[2], box[3], q)
    if ds is not None:
        return (float(np.percentile(ds[:, 0], 1)), float(np.percentile(ds[:, 0], 99)),
                float(np.percentile(ds[:, 1], 1)), float(np.percentile(ds[:, 1], 99)), "dark strokes")
    bl = bp2["det"].points @ lateral
    l0, l1 = np.percentile(bl, 5), np.percentile(bl, 95)
    z0, z1 = float(bp2["lo"][2]), float(bp2["hi"][2])
    return (float(l0 + 0.15 * (l1 - l0)), float(l1 - 0.15 * (l1 - l0)), z0 + 0.15 * (z1 - z0), z1 - 0.15 * (z1 - z0), "board")


if pick["ok"]:
    api.stage_done()
    gn_ = (api.notes.get("top_grasp_right") or {}).get("grasp")
    if gn_ is not None and er_sup is not None:
        felt_w = float(gn_[2]) - er_sup
    else:
        felt_w = (er_hw - 0.025) if er_hw is not None else None
    felt_m, width_m = (None, None) if (felt_w is not None and 0.004 < felt_w < 0.06) else eraser_in_hand()
    if felt_w is not None and 0.004 < felt_w < 0.06:
        FELT = felt_w
    elif felt_m is not None and 0.004 < felt_m < 0.06:
        FELT = felt_m
    if er_ww is not None and 0.02 < er_ww < 0.08:
        STEP = er_ww
    elif width_m is not None and 0.02 < width_m < 0.08:
        STEP = width_m
    elif er_w is not None and 0.02 < er_w < 0.08:
        STEP = er_w
    report["eraser_measured"] = {"felt_wrist": None if felt_w is None else round(felt_w, 4),
                                 "felt_in_hand": None if felt_m is None else round(felt_m, 4),
                                 "width_in_hand": None if width_m is None else round(width_m, 4),
                                 "width_head": None if er_w is None else round(er_w, 4),
                                 "felt": None if FELT is None else round(FELT, 4), "step": None if STEP is None else round(STEP, 4)}
    try:
        m = api.motion()
        vertical = top_down((0.0, 0.0, 0.0), 0.0)[:3, :3]
        for tx, ty, tz in ((0.42, -0.30, 0.03), (0.42, -0.20, 0.03), (0.36, -0.40, 0.03), (0.48, -0.30, 0.03), (0.42, -0.30, 0.12),
                           (0.48, -0.25, 0.12), (0.36, -0.30, 0.03)):
            try:
                trial = m.copy()
                near = api.grasp_pose("right").copy()
                near[:3, :3] = vertical
                near[0, 3], near[1, 3] = tx, ty
                near[2, 3] = near[2, 3] + tz
                trial.tcp("right", near, free_axis=(0.0, 0.0, 1.0))
                m = trial
                report["tuck"] = [tx, ty, tz]
                break
            except MotionError as error:
                report.setdefault("tuck_errors", []).append(str(error)[:80])
        m.base(0.0, 0.0, math.pi, yaw_speed=0.8)
        m.hold(60)
        m.send(depth_next=True, label="turn_around")
    except MotionError as error:
        report["turn_error"] = str(error)[:200]
    bp = board_plane()
    report["board_seen"] = None if bp is None else {"c": [round(float(v), 3) for v in bp["c"]], "n": [round(float(v), 3) for v in bp["n"]]}
    if bp is not None:
        n = bp["n"]
        heading = math.atan2(float(n[1]), float(n[0]))
        mid = bp["det"].mid
        dist_to_plane = float(np.dot(n, mid)) - STANDOFF
        lateral = np.array([-n[1], n[0], 0.0])
        lat = float(np.dot(lateral, mid)) - SWEET_Y
        tx = float(n[0] * dist_to_plane + lateral[0] * lat)
        ty = float(n[1] * dist_to_plane + lateral[1] * lat)
        try:
            m = api.motion()
            m.base(tx, ty, heading, speed=0.3, yaw_speed=0.6)
            m.hold(60)
            m.send(depth_next=True, label="to_board")
            report["to_board"] = [round(tx, 3), round(ty, 3), round(math.degrees(heading), 1)]
            api.stage_done()
        except MotionError as error:
            report["nav_error"] = str(error)[:200]
        bp2 = board_plane()
        if bp2 is not None and bp2["d"] > STANDOFF + 0.07:
            try:
                m = api.motion()
                m.base(float(bp2["d"] - STANDOFF), 0.0, 0.0, speed=0.15)
                m.hold(45)
                m.send(depth_next=True, label="to_board_2")
                report["to_board_2"] = round(float(bp2["d"] - STANDOFF), 3)
                bp2 = board_plane() or bp2
            except MotionError as error:
                report["nav2_error"] = str(error)[:200]
        if bp2 is not None and FELT is not None and STEP is not None:
            lat_lo, lat_hi, z_lo, z_hi, source = marked_area(bp2)
            report["marks"] = {"source": source, "area": [round(v, 3) for v in (lat_lo, lat_hi, z_lo, z_hi)]}
            shift = 0.5 * (lat_lo + lat_hi) - SWEET_Y
            if abs(shift) > 0.05:
                try:
                    nn = bp2["n"]
                    lat_v = np.array([-nn[1], nn[0], 0.0])
                    m = api.motion()
                    m.base(float(lat_v[0] * shift), float(lat_v[1] * shift), 0.0, speed=0.15)
                    m.hold(30)
                    m.send(depth_next=True, label="side_step")
                    report["side_step"] = round(float(shift), 3)
                    bp3 = board_plane()
                    if bp3 is not None:
                        bp2 = bp3
                        lat_lo, lat_hi, z_lo, z_hi, source = marked_area(bp2)
                        report["marks_2"] = {"source": source, "area": [round(v, 3) for v in (lat_lo, lat_hi, z_lo, z_hi)]}
                except MotionError as error:
                    report["side_step_error"] = str(error)[:200]
            n, d = bp2["n"], bp2["d"]
            report["board_near"] = {"d": round(d, 3), "n": [round(float(v), 3) for v in n],
                                    "lo": [round(float(v), 3) for v in bp2["lo"]], "hi": [round(float(v), 3) for v in bp2["hi"]]}
            lateral = np.array([-n[1], n[0], 0.0])
            lat_lo, lat_hi = lat_lo - 0.04, lat_hi + 0.04
            z_lo, z_hi = max(z_lo - 0.03, 0.95), min(z_hi + 0.03, 1.60)
            z_axis = n
            y_axis = np.array([0.0, 0.0, 1.0])
            x_axis = np.cross(y_axis, z_axis)
            rot = np.column_stack([x_axis, y_axis, z_axis])
            free = (0.0, 0.0, 1.0)
            lat_p = float(lateral @ np.array([0.0, SWEET_Y, 0.0]))
            probe_errs = []
            for z_p in (z_hi - 0.02, z_lo + 0.02, 0.5 * (z_lo + z_hi)):
                try:
                    mp = api.motion()
                    far_p = n * (d - 0.32) + lateral * lat_p
                    p0 = n * (d - FELT - 0.03) + lateral * lat_p
                    p1 = n * (d - FELT + 0.04) + lateral * lat_p
                    mp.tcp("right", pose((float(far_p[0]), float(far_p[1]), z_p), rot), free_axis=free,
                           limits=Limits(linear_velocity=0.12, joint_velocity=0.6, joint_acceleration=1.5))
                    mp.tcp("right", pose((float(p0[0]), float(p0[1]), z_p), rot), linear=True, free_axis=free, limits=Limits(linear_velocity=0.12))
                    mp.tcp("right", pose((float(p1[0]), float(p1[1]), z_p), rot), linear=True, free_axis=free, limits=Limits(linear_velocity=0.03))
                    mp.hold(10)
                except MotionError as error:
                    probe_errs.append(f"z {z_p:.2f}: {str(error)[:80]}")
                    continue
                mp.send(depth_next=True, label="probe_felt")
                s_meas = float(api.grasp_pose("right")[:3, 3] @ n)
                s_cmd = float(p1 @ n)
                report["probe"] = {"z": round(z_p, 3), "felt_before": round(FELT, 4), "short_mm": round(1000.0 * (s_cmd - s_meas), 1),
                                   "felt_touch": round(d - s_meas, 4)}
                if s_cmd - s_meas > 0.008 and 0.003 < d - s_meas < 0.08:
                    FELT = d - s_meas
                back_p = n * (d - FELT - 0.10) + lateral * lat_p
                for lin in (True, False):
                    try:
                        mb = api.motion()
                        mb.tcp("right", pose((float(back_p[0]), float(back_p[1]), z_p), rot), linear=lin, free_axis=free,
                               limits=Limits(linear_velocity=0.2, joint_velocity=0.6, joint_acceleration=1.5))
                        mb.send(depth_next=True, label="probe_back")
                        break
                    except MotionError as error:
                        probe_errs.append(f"back: {str(error)[:80]}")
                break
            if probe_errs:
                report["probe_error"] = probe_errs[:4]
            depth = d - FELT + PRESS
            zs = list(np.arange(z_hi, z_lo - 1e-6, -STEP))
            if not zs or zs[-1] > z_lo + 0.01:
                zs.append(z_lo)
            wipe = Limits(linear_velocity=0.40, joint_velocity=2.2)
            slow = Limits(linear_velocity=0.25, joint_velocity=2.2)
            brisk = Limits(linear_velocity=0.2, joint_velocity=1.2, joint_acceleration=3.0)
            gentle = Limits(linear_velocity=0.12, joint_velocity=0.6, joint_acceleration=1.5)
            tol = (api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad)
            api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = 0.003, math.radians(1.5)
            WIN = 0.46
            n_win = max(1, int(math.ceil((lat_hi - lat_lo) / WIN - 1e-6)))
            edges = [lat_lo + (lat_hi - lat_lo) * k / n_win for k in range(n_win + 1)]
            windows = [(edges[k], edges[k + 1]) for k in range(n_win)]
            windows.sort(key=lambda w: abs(0.5 * (w[0] + w[1]) - SWEET_Y))
            report["windows"] = [[round(w[0], 3), round(w[1], 3)] for w in windows]
            offset_lat = 0.0
            first = True
            win_log = []
            for wi, (w_lo, w_hi) in enumerate(windows):
                mid_w = 0.5 * (w_lo + w_hi) - offset_lat
                step_lat = mid_w - SWEET_Y
                if abs(step_lat) > 0.04:
                    try:
                        sm = api.motion()
                        if not first:
                            back = sm.grasp_pose("right").copy()
                            back[:3, 3] = back[:3, 3] - n * 0.10
                            t_saved = api.ik_position_tolerance_m
                            api.ik_position_tolerance_m = 0.006
                            try:
                                trial_b = sm.copy()
                                trial_b.tcp("right", back, linear=True, free_axis=free)
                                sm = trial_b
                            except MotionError:
                                try:
                                    trial_b = sm.copy()
                                    trial_b.tcp("right", back, free_axis=free, limits=brisk)
                                    sm = trial_b
                                except MotionError as error:
                                    win_log.append({"window": wi, "back_error": str(error)[:80]})
                            finally:
                                api.ik_position_tolerance_m = t_saved
                        sm.base(float(lateral[0] * step_lat), float(lateral[1] * step_lat), 0.0, speed=0.2)
                        sm.hold(25)
                        sm.send(depth_next=True, label=f"side_step_w{wi}")
                        offset_lat += step_lat
                    except MotionError as error:
                        win_log.append({"window": wi, "step_error": str(error)[:100]})
                        continue
                    bpw = board_plane()
                    if bpw is not None:
                        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(bpw["n"] @ n)))))
                        if ang < 12.0 and abs(float(bpw["d"]) - d) < 0.15:
                            win_log.append({"window": wi, "replane": [round(float(bpw["d"]) - d, 3), round(ang, 1)]})
                            n, d = bpw["n"], float(bpw["d"])
                            lateral = np.array([-n[1], n[0], 0.0])
                            x_axis = np.cross(y_axis, n)
                            rot = np.column_stack([x_axis, y_axis, n])
                            depth = d - FELT + PRESS
                        else:
                            win_log.append({"window": wi, "replane_rejected": [round(float(bpw["d"]) - d, 3), round(ang, 1)]})
                a_w, b_w = w_lo - offset_lat, w_hi - offset_lat
                m = api.motion()
                done_passes = 0
                started = False
                n_rest = len(windows) - wi
                share = (api.rows_left() - 60 - 400 * (n_rest - 1)) / max(1, n_rest)
                win_log.append({"window": wi, "share_rows": int(share)})
                for i, z in enumerate(zs):
                    if api.reply_elapsed() > 50.0:
                        win_log.append({"window": wi, "reply_cap_at_pass": i})
                        break
                    if started and m.rows > share:
                        win_log.append({"window": wi, "cut_at_pass": i})
                        break
                    a, b = (a_w, b_w) if i % 2 == 0 else (b_w, a_w)
                    third = (b - a) / 3.0
                    options = [([(a, b)], "line"), ([(a + (b - a) / 6.0, b - (b - a) / 6.0)], "line"),
                               ([(a, 0.5 * (a + b)), (0.5 * (a + b), b)], "line"), ([(a, b)], "reseed"),
                               ([(a, a + third), (a + third, a + 2 * third), (a + 2 * third, b)], "reseed")]
                    for k, (pieces, mode) in enumerate(options):
                        trial = m.copy()
                        got = 0
                        if mode == "reseed":
                            t_saved = (api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad)
                            api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = 0.006, math.radians(2.5)
                        for (aa, bb) in pieces:
                            t2 = trial.copy()
                            try:
                                p0 = n * depth + lateral * aa
                                p1 = n * depth + lateral * bb
                                if not started:
                                    far = n * (d - (0.32 if first else 0.10)) + lateral * aa
                                    start = n * (d - 0.10) + lateral * aa
                                    t2.tcp("right", pose((float(far[0]), float(far[1]), float(z)), rot), free_axis=free, limits=gentle if first else brisk)
                                    if first:
                                        t2.tcp("right", pose((float(start[0]), float(start[1]), float(z)), rot), linear=True, free_axis=free, limits=gentle)
                                if mode == "reseed":
                                    t2.tcp("right", pose((float(p0[0]), float(p0[1]), float(z)), rot), free_axis=free, limits=brisk)
                                    t2.tcp("right", pose((float(p1[0]), float(p1[1]), float(z)), rot), linear=True, limits=slow, free_axis=free)
                                else:
                                    t2.tcp("right", pose((float(p0[0]), float(p0[1]), float(z)), rot), linear=True, limits=wipe, free_axis=free)
                                    t2.tcp("right", pose((float(p1[0]), float(p1[1]), float(z)), rot), linear=True, limits=wipe, free_axis=free)
                            except MotionError as error:
                                report.setdefault("wipe_errors", []).append(f"w{wi} z {z:.2f} option {k}: {str(error)[:50]}")
                                continue
                            trial = t2
                            started = True
                            got += 1
                        if mode == "reseed":
                            api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = t_saved
                        if got:
                            if started and done_passes > 0 and trial.rows > share + 120:
                                got = 0
                                break
                            m = trial
                            done_passes += 1
                            break
                if started:
                    if wi == len(windows) - 1:
                        try:
                            back = m.grasp_pose("right").copy()
                            back[:3, 3] = back[:3, 3] - n * 0.10
                            m.tcp("right", back, linear=True, free_axis=free)
                        except MotionError as error:
                            report["wipe_back_error"] = str(error)[:120]
                    m.send(depth_next=True, label=f"wipe_w{wi}")
                    first = False
                win_log.append({"window": wi, "lat": [round(a_w, 3), round(b_w, 3)], "passes": [done_passes, len(zs)], "rows": m.rows if started else 0})
                if api.rows_left() < 400:
                    break
            api.ik_position_tolerance_m, api.ik_rotation_tolerance_rad = tol
            report["wipe_windows"] = win_log
report["budget"] = api.budget()["phases"]
api.end_episode()
