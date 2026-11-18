# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

pen_side = api.instructed_arm(api.segment_text(0)) or "right"
cap_side = api.instructed_arm(api.segment_text(1)) or ("left" if pen_side == "right" else "right")
table = api.table_height("head")
FAST = Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30)
report["pen_side"], report["cap_side"] = pen_side, cap_side
cam = api.camera("head")
img = cam.rgb.astype(float)


def axis_info(det, kind):
    P = det.points
    c = np.median(P, axis=0)
    xy = P[:, :2] - c[:2]
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    a = vecs[:, int(np.argmax(vals))]
    a3 = np.array([float(a[0]), float(a[1]), 0.0])
    pr = xy @ a
    lo, hi = float(np.percentile(pr, 2)), float(np.percentile(pr, 98))
    ends = {"lo": c + a3 * lo, "hi": c + a3 * hi}
    uv, ok = cam.project_base_points(np.vstack([ends["lo"], ends["hi"]]))
    ys, xs = np.nonzero(det.mask)
    pix = np.column_stack([xs, ys]).astype(float)
    d_lo = np.linalg.norm(pix - uv[0], axis=1)
    d_hi = np.linalg.norm(pix - uv[1], axis=1)
    L = float(np.linalg.norm(uv[1] - uv[0])) + 1e-6
    stats = {}
    for name, dist, other in (("lo", d_lo, d_hi), ("hi", d_hi, d_lo)):
        sel = (dist < 0.30 * L) & (dist < other)
        if int(sel.sum()) < 5:
            sel = dist < other
        col = img[ys[sel], xs[sel]].mean(axis=0)
        stats[name] = {"bright": float(col.mean()), "sat": float(col.max() - col.min())}
    if kind == "pen":
        special = "lo" if stats["lo"]["bright"] < stats["hi"]["bright"] else "hi"
    else:
        closed_score = {k: (255.0 - v["bright"]) + 2.0 * v["sat"] for k, v in stats.items()}
        special = "lo" if closed_score["lo"] < closed_score["hi"] else "hi"
    other = "hi" if special == "lo" else "lo"
    u = ends[special] - ends[other]
    u = u / max(1e-6, float(np.linalg.norm(u)))
    h = 0.5 * (float(np.percentile(P[:, 2], 95)) + float(table))
    return {"c": c, "end": ends[special], "u": u, "stats": stats, "special": special, "h": h}


def grasp_frame(note):
    if not note or abs(float(note.get("tilt_deg", 0.0))) > 1.0 or note.get("waist") is not None:
        return None
    g = np.array(note["grasp"], dtype=float)
    return top_down((float(g[0]), float(g[1]), float(g[2])), math.radians(float(note["closing_yaw_deg"])))


cap_seen = api.find_object(["pen cap", "cap"], size=(0.01, 0.08), table_z=table, z_tolerance=0.03, clip_z=(table - 0.01, table + 0.03))
report["cap_seen"] = None if cap_seen is None else cap_seen.summary()
pen_find = {"size": (0.06, 0.25), "table_z": table, "z_tolerance": 0.03, "clip_z": (table - 0.01, table + 0.03)}
pen_det = api.find_object(["pen", "ballpoint pen", "marker pen"], **pen_find)
pen_ax = axis_info(pen_det, "pen") if pen_det is not None else None
pen = api.pick_object(pen_side, ["pen", "ballpoint pen", "marker pen"], find=pen_find,
                      grasp={"depth_into": 0.05, "lift": 0.08, "above": 0.06, "floor_clearance": 0.014}, limits=FAST, clear_view=0.15)
report["pen"] = pen
G_pen = grasp_frame(api.notes.get(f"top_grasp_{pen_side}")) if pen["ok"] else None
if pen["ok"]:
    api.stage_done()
    if not api.has_depth():
        api.hold(1, depth_next=True, label="observe")
    cap_find = {"size": (0.01, 0.08), "table_z": table, "z_tolerance": 0.03, "side": cap_side, "clip_z": (table - 0.01, table + 0.03)}
    cap_det = api.find_object(["pen cap", "cap"], **cap_find)
    cap_ax = axis_info(cap_det, "cap") if cap_det is not None else None
    cap = api.pick_object(cap_side, ["pen cap", "cap"], find=cap_find,
                          grasp={"depth_into": 0.05, "lift": 0.06, "above": 0.06, "floor_clearance": 0.015}, limits=FAST, clear_view=0.15)
    report["cap"] = cap
    G_cap = grasp_frame(api.notes.get(f"top_grasp_{cap_side}")) if cap["ok"] else None
    if cap["ok"]:
        api.stage_done()
    if cap["ok"] and G_pen is not None and G_cap is not None and pen_ax is not None and cap_ax is not None:
        Gp_inv, Gc_inv = np.linalg.inv(G_pen), np.linalg.inv(G_cap)

        def in_hand(G_inv, ax):
            end = ax["end"].copy()
            end[2] = ax["h"]
            p = (G_inv @ np.append(end, 1.0))[:3]
            raw = (G_inv @ np.append(ax["end"], 1.0))[:3]
            p[1] = 0.0
            d = G_inv[:3, :3] @ ax["u"]
            d[1] = 0.0
            d[2] = 0.0
            d = d / max(1e-6, float(np.linalg.norm(d)))
            return p, d, raw

        tip_l, tipdir_l, tip_raw = in_hand(Gp_inv, pen_ax)
        mouth_l, opendir_l, mouth_raw = in_hand(Gc_inv, cap_ax)
        report["ends"] = {"pen": pen_ax["stats"], "cap": cap_ax["stats"], "tip_l": [round(float(v), 4) for v in tip_l],
                          "mouth_l": [round(float(v), 4) for v in mouth_l], "tip_raw": [round(float(v), 4) for v in tip_raw],
                          "mouth_raw": [round(float(v), 4) for v in mouth_raw]}
        s = 1.0 if pen_side == "right" else -1.0

        def yaw_for(u_local, want_xy):
            best = None
            for k in range(360):
                yv = math.radians(k)
                R = top_down((0.0, 0.0, 0.0), yv)[:3, :3]
                w = R @ u_local
                err = float(np.linalg.norm(w[:2] - want_xy))
                if best is None or err < best[0]:
                    best = (err, yv)
            return best[1]

        yaw_p = yaw_for(tipdir_l, np.array([0.0, s]))
        yaw_c = yaw_for(opendir_l, np.array([0.0, -s]))
        Rp = top_down((0.0, 0.0, 0.0), yaw_p)[:3, :3]
        Rc = top_down((0.0, 0.0, 0.0), yaw_c)[:3, :3]
        done = False
        for T in ((0.50, 0.0, 0.95), (0.46, 0.0, 0.93), (0.54, 0.0, 0.97), (0.48, 0.03 * s, 0.95), (0.48, -0.03 * s, 0.95)):
            T = np.array(T)
            axis_y = np.array([0.0, s, 0.0])
            Gp = pose(T - Rp @ tip_l, Rp)
            Gc_pre = pose(T + axis_y * 0.035 - Rc @ mouth_l, Rc)
            Gc_ins = pose(T - axis_y * (0.0315 + 0.010) - Rc @ mouth_l, Rc)
            try:
                m = api.motion()
                m.tcp({pen_side: Gp, cap_side: Gc_pre})
                m.hold(20)
                m.send(depth_next=True, label="align_coaxial")
            except MotionError as error:
                report.setdefault("align_errors", []).append(str(error)[:120])
                continue
            report["align_T"] = [round(float(v), 3) for v in T]
            wcam = f"hand_{cap_side}"
            mouth_w = T + axis_y * 0.035
            pen_pts, cap_pts = None, None
            if api.has_depth(wcam):
                for q in ("marker pen", "pen"):
                    for d in api.perceive(q, wcam, max_detections=4):
                        P = d.points
                        along = (P[:, 1] - float(T[1])) * s
                        sel = (along < 0.0) & (along > -0.06) & (np.abs(P[:, 0] - float(T[0])) < 0.025) & (np.abs(P[:, 2] - float(T[2])) < 0.03)
                        if int(sel.sum()) >= 30:
                            pen_pts = P[sel]
                            break
                    if pen_pts is not None:
                        break
                for q in ("pen cap", "white cap", "cap"):
                    for d in api.perceive(q, wcam, max_detections=4):
                        P = d.points
                        along = (P[:, 1] - float(mouth_w[1])) * s
                        sel = (along > -0.005) & (along < 0.045) & (np.abs(P[:, 0] - float(T[0])) < 0.025) & (np.abs(P[:, 2] - float(T[2])) < 0.03)
                        if int(sel.sum()) >= 30:
                            cap_pts = P[sel]
                            break
                    if cap_pts is not None:
                        break
            dx = 0.0
            dz = 0.0
            if pen_pts is not None and cap_pts is not None:
                dx_seen = float(np.median(pen_pts[:, 0]) - np.median(cap_pts[:, 0]))
                dz_seen = float(np.percentile(pen_pts[:, 2], 90) - 0.0088) - float(np.percentile(cap_pts[:, 2], 90) - 0.0097)
                report["wrist_check"] = {"dx": round(dx_seen, 4), "dz": round(dz_seen, 4), "pen_pts": int(pen_pts.shape[0]),
                                         "cap_pts": int(cap_pts.shape[0])}
                if 0.0015 < abs(dx_seen) < 0.010:
                    dx = dx_seen
                if 0.002 < abs(dz_seen) < 0.009:
                    dz = dz_seen
            else:
                report["wrist_check"] = {"pen": pen_pts is not None, "cap": cap_pts is not None}
            if dx != 0.0 or dz != 0.0:
                Gc_pre = Gc_pre.copy()
                Gc_pre[0, 3] += dx
                Gc_pre[2, 3] += dz
                Gc_ins = Gc_ins.copy()
                Gc_ins[0, 3] += dx
                Gc_ins[2, 3] += dz
                report["cap_shift"] = [round(dx, 4), round(dz, 4)]
            try:
                m = api.motion()
                if dx != 0.0 or dz != 0.0:
                    m.tcp(cap_side, Gc_pre, linear=True, limits=Limits(linear_velocity=0.03))
                m.tcp(cap_side, Gc_ins, linear=True, limits=Limits(linear_velocity=0.05))
                m.hold(12)
                m.gripper(cap_side, "open", rows=15)
                back = Gc_ins.copy()
                back[:3, 3] = back[:3, 3] + axis_y * 0.05 + np.array([0.0, 0.0, 0.06])
                down = Gp.copy()
                down[2, 3] = table + 0.016
                m.tcp({cap_side: back, pen_side: offset(down, dz=0.04)}, limits=FAST)
                m.tcp(pen_side, down, linear=True, limits=Limits(linear_velocity=0.2))
                m.gripper(pen_side, "open", rows=15)
                m.tcp(pen_side, offset(down, dz=0.05), linear=True)
                m.send(depth_next=False, label="cap_on_and_place")
                report["inserted_and_placed"] = True
                done = True
            except MotionError as error:
                report["insert_error"] = str(error)[:200]
            break
report["budget"] = api.budget()["phases"]
api.end_episode()
