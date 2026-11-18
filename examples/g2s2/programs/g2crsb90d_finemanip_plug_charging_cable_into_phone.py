# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

table = api.table_height("head")
plug_seen = api.find_object(["white charging plug", "usb plug", "charger plug"], size=(0.008, 0.08))
report["plug_seen"] = None if plug_seen is None else plug_seen.summary()
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
    mid_top = c0 + lng * top_s + nrm * 0.0059
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
    plug = api.pick_object("right", ["white charging plug", "usb plug", "charger plug", "plug"],
                           find={"size": (0.008, 0.08), "side": "right"},
                           grasp={"depth_into": 0.012, "lift": 0.06, "above": 0.06, "floor_clearance": 0.010},
                           limits=Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30))
    report["plug"] = plug
    if plug["ok"]:
        api.stage_done()
report["budget"] = api.budget()["phases"]
api.end_episode()
