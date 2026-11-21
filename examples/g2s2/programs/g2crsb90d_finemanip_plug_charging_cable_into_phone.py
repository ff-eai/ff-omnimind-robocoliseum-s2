# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

table = api.table_height("head")
plug_seen = api.find_object(["white charging plug", "usb plug", "charger plug"], size=(0.008, 0.08))
report["plug_seen"] = None if plug_seen is None else plug_seen.summary()


def cable_end():
    view = api.camera("head")
    if view.depth_m is None:
        return None
    rgb = view.rgb.astype(np.float64) / 255.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    green = (rgb[:, :, 1] >= rgb[:, :, 0]) & (rgb[:, :, 1] > rgb[:, :, 2] + 0.08) & (sat > 0.25) & (mx > 0.35)
    if int(green.sum()) < 40:
        return None
    P = view.points_from_mask(green, stride=1)
    if P.shape[0] < 40:
        return None
    Q = P[(P[:, 2] > table - 0.01) & (P[:, 2] < table + 0.03)]
    for it_ in range(2):
        if Q.shape[0] < 30:
            return None
        med = np.median(Q[:, :2], axis=0)
        Q = Q[np.linalg.norm(Q[:, :2] - med, axis=1) < 0.20]
    if Q.shape[0] < 30:
        return None
    c = np.median(Q[:, :2], axis=0)
    dv = Q[:, :2] - c
    w_, V_ = np.linalg.eigh(np.cov(dv.T))
    ax = V_[:, int(np.argmax(w_))]
    s_ = dv @ ax
    e1, e2 = Q[int(np.argmin(s_))], Q[int(np.argmax(s_))]
    box = api.find_object(["white box", "power strip storage box", "white charger box"], size=(0.05, 0.40))
    if box is not None:
        bxy = box.mid[:2]
        far = e1 if np.linalg.norm(e1[:2] - bxy) > np.linalg.norm(e2[:2] - bxy) else e2
    else:
        far = e1 if np.linalg.norm(e1[:2]) < np.linalg.norm(e2[:2]) else e2
    near = Q[np.linalg.norm(Q[:, :2] - far[:2], axis=1) < 0.04]
    dirv = far[:2] - np.median(near[:, :2], axis=0)
    if float(np.linalg.norm(dirv)) < 1e-4:
        return None
    dirv = dirv / float(np.linalg.norm(dirv))
    white = (sat < 0.15) & (mx > 0.70)
    W = view.points_from_mask(white, stride=1) if int(white.sum()) >= 10 else np.zeros((0, 3))
    plug_xy = far[:2]
    top_w = float(far[2])
    if W.shape[0] >= 10:
        W = W[(W[:, 2] > table - 0.01) & (W[:, 2] < table + 0.03)]
        rel = W[:, :2] - far[:2]
        along = rel @ dirv
        across = np.abs(rel @ np.array([-dirv[1], dirv[0]]))
        Wc = W[(along > -0.005) & (along < 0.04) & (across < 0.012)]
        if Wc.shape[0] >= 8:
            plug_xy = np.median(Wc[:, :2], axis=0)
            top_w = float(np.percentile(Wc[:, 2], 95))
    return {"xy": plug_xy, "dir": dirv, "len": float(s_.max() - s_.min()), "end": far, "top": top_w}


cable_start = cable_end()
report["cable_start"] = None if cable_start is None else {"xy": [round(float(v), 4) for v in cable_start["xy"]],
                                                          "dir": [round(float(v), 3) for v in cable_start["dir"]],
                                                          "len": round(cable_start["len"], 3)}
try:
    up = api.motion()
    t = api.grasp_pose("left").copy()
    t[2, 3] += 0.08
    t[1, 3] += 0.08
    up.tcp("left", t)
    up.send(depth_next=True, label="clear_view")
except MotionError as error:
    report["clear_error"] = str(error)[:200]
phone = api.find_object(["black smartphone", "black mobile phone", "phone"], size=(0.04, 0.20), table_z=table, z_tolerance=0.10,
                        brightness=(0.0, 110.0))
if phone is None:
    phone = api.find_object(["black smartphone", "smartphone on a stand", "phone"], size=(0.04, 0.22), table_z=table, z_tolerance=0.12)
    report["phone_second_look"] = phone is not None
report["phone_seen"] = None if phone is None else phone.summary()
held = False
if phone is not None:
    P = phone.points
    topz = float(phone.top_z)
    edge = P[P[:, 2] > topz - 0.025]
    if edge.shape[0] < 20:
        edge = P
    c = np.median(edge, axis=0)
    xy = edge[:, :2] - c[:2]
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    along = vecs[:, int(np.argmax(vals))]
    edge_yaw = math.atan2(float(along[1]), float(along[0]))
    report["phone_edge"] = {"c": [round(float(v), 4) for v in c], "top": round(topz, 4), "yaw_deg": round(math.degrees(edge_yaw), 1)}
    P0 = phone.points
    c0 = P0.mean(axis=0)
    ev, V = np.linalg.eigh(np.cov((P0 - c0).T))
    nrm = V[:, 0]
    keep_p = np.abs((P0 - c0) @ nrm) < 0.010
    if int(keep_p.sum()) > 50:
        c0 = P0[keep_p].mean(axis=0)
        ev, V = np.linalg.eigh(np.cov((P0[keep_p] - c0).T))
        nrm = V[:, 0]
    upz = np.array([0.0, 0.0, 1.0]) - float(nrm[2]) * nrm
    lng = upz / max(float(np.linalg.norm(upz)), 1e-6)
    cam_h = api.camera("head").T_base_optical[:3, 3]
    if float(nrm @ (c0 - cam_h)) < 0:
        nrm = -nrm
    sl = (P0 - c0) @ lng
    top_s = float(np.percentile(sl, 98))
    strip = (P0[sl > top_s - 0.006] - c0) @ nrm
    half_t = min(max(float(np.median(strip)), 0.002), 0.010) if strip.shape[0] >= 15 else 0.002
    mid_top = c0 + lng * top_s + nrm * half_t
    report["phone_half_thickness"] = round(half_t, 4)
    lean = math.degrees(math.acos(min(1.0, abs(float(lng[2])))))
    report["phone_plane"] = {"normal": [round(float(v), 3) for v in nrm], "long": [round(float(v), 3) for v in lng],
                             "lean_deg": round(lean, 1), "top": [round(float(v), 4) for v in mid_top],
                             "extent_mm": [round(1000 * float(np.sqrt(max(e, 0.0)) * 4.0), 1) for e in ev]}
    side_ax = np.cross(nrm, lng)
    def face_grasp(sign, depth_, turn, ease):
        d_ = math.cos(turn) * (-lng) + math.sin(turn) * side_ax
        d_ = (1.0 - ease) * d_ + ease * np.array([0.0, 0.0, -1.0])
        zg = d_ / np.linalg.norm(d_)
        yg = nrm - float(nrm @ zg) * zg
        yg = sign * yg / np.linalg.norm(yg)
        xg = np.cross(yg, zg)
        R_ = np.column_stack([xg, yg, zg])
        at = mid_top + zg * depth_
        return pose(at, R_), pose(at - zg * (depth_ + 0.07), R_), pose(at - zg * (depth_ + 0.07) + np.array([0.0, 0.0, 0.05]), R_)
    tries = [(sg, 0.03, tn, es) for es in (0.0, 0.35) for tn in (0.0, 0.5, -0.5, 1.0, -1.0) for sg in (1.0, -1.0)]
    for attempt, (sg, dp, tn, es) in enumerate(tries):
        if api.rows_left() < 260 or api.reply_elapsed() > 35.0:
            break
        g_t, pre_t, up_t = face_grasp(sg, dp, tn, es)
        try:
            m = api.motion()
            m.gripper("left", "open", rows=10)
            m.tcp("left", pre_t, limits=Limits(joint_velocity=3.0, joint_acceleration=12.0))
            m.tcp("left", g_t, linear=True, limits=Limits(linear_velocity=0.15))
            m.gripper("left", "close", rows=20)
            m.hold(5)
            m.tcp("left", up_t, linear=True, limits=Limits(linear_velocity=0.15))
            if m.rows > 340:
                raise MotionError(f"face grasp needs {m.rows} rows (stage 0 StepOut 450)")
        except MotionError as error:
            errs_p = report.setdefault("plan_errors", [])
            if len(errs_p) < 8:
                errs_p.append(f"sign {sg} turn {tn} ease {es}: {str(error)[:60]}")
            continue
        m.send(depth_next=True, label=f"pick_phone_face{attempt}")
        held = bool(api.holding("left"))
        report.setdefault("phone_tries", []).append(["face", sg, tn, es, held])
        if held:
            break
        break
    if not held and api.rows_left() >= 260:
        for attempt in range(2):
            try:
                m = api.motion()
                if attempt:
                    m.gripper("left", "open", rows=20)
                api.top_grasp("left", center=(float(c[0]), float(c[1]), topz), top_z=topz, bottom_z=topz - 0.06, yaw=edge_yaw,
                              depth_into=0.02 + 0.01 * attempt, lift=0.10, above=0.07, motion=m)
                m.send(depth_next=True, label=f"pick_phone{attempt}")
            except MotionError as error:
                report.setdefault("plan_errors", []).append(str(error)[:200])
                break
            held = bool(api.holding("left"))
            report.setdefault("phone_tries", []).append(["top", held])
            if held or api.rows_left() < 260:
                break
if held:
    api.stage_done()
    if not api.has_depth():
        api.hold(1, depth_next=True, label="observe")
    cab = cable_end() or cable_start
    plug = {"ok": False}
    if cab is not None:
        report["cable_used"] = {"xy": [round(float(v), 4) for v in cab["xy"]], "dir": [round(float(v), 3) for v in cab["dir"]],
                                "len": round(cab["len"], 3)}
        yaw_c = math.atan2(float(cab["dir"][1]), float(cab["dir"][0]))
        top_c = float(cab.get("top") if cab.get("top") is not None else cab["end"][2])
        try:
            plan = api.top_grasp("right", center=(float(cab["xy"][0]), float(cab["xy"][1]), top_c), top_z=top_c, bottom_z=table,
                                 yaw=yaw_c, depth_into=0.012, floor_clearance=0.010, lift=0.06, above=0.06,
                                 limits=Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30))
            plan.send(depth_next=True, label="pick_plug_cable_end")
            plug = {"ok": bool(api.holding("right")), "source": "cable_end"}
        except MotionError as error:
            report["cable_pick_error"] = str(error)[:160]
    report["plug_cable"] = plug
    if not plug.get("ok"):
        plug = api.pick_object("right", ["white charging plug", "usb plug", "charger plug", "plug"],
                           find={"size": (0.008, 0.08), "side": "right"},
                           grasp={"depth_into": 0.012, "lift": 0.06, "above": 0.06, "floor_clearance": 0.010},
                           limits=Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30))
    report["plug"] = plug
    if plug["ok"]:
        api.stage_done()
report["budget"] = api.budget()["phases"]
api.end_episode()
