# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
report["embodiment"] = info["embodiment"]
active = info["active_arms"]
start_arm = {s: np.array(q, dtype=float) for s, q in api.robot_state()["arm"].items()}

table = api.table_height("head")
report["table_z"] = round(table, 4)
pedal = api.find(["foot pedal", "trash can foot pedal", "small red pedal"], size=(0.015, 0.10), table_z=table, z_tolerance=0.05)
ball = api.find(["paper ball", "white paper ball", "crumpled white paper"], size=(0.02, 0.10), table_z=table, z_tolerance=0.06)
can = api.find(["trash can", "red trash can", "pedal trash can"], size=(0.08, 0.35), table_z=table, z_tolerance=0.08)
report["pedal"] = None if pedal is None else pedal.summary()
report["can"] = None if can is None else can.summary()
report["ball"] = None if ball is None else ball.summary()

if len(active) > 1:
    pick_side = "right" if (ball is None or float(ball.mid[1]) < 0.0) else "left"
    press_order = [s for s in active if s != pick_side] + [pick_side]
else:
    pick_side = active[0]
    press_order = list(active)
report["pick_side"] = pick_side


def outward(det):
    p = det.mid
    if can is not None:
        v = np.array([float(p[0] - can.mid[0]), float(p[1] - can.mid[1])])
        if float(np.linalg.norm(v)) > 0.03:
            return v / float(np.linalg.norm(v)), "can"
    v = -np.array([float(p[0]), float(p[1])])
    return v / float(np.linalg.norm(v)), "robot"


waist_options = [None]


def press_motion(side, det, free_yaw, way_opt=None):
    n, source = outward(det)
    phi = math.atan2(float(n[1]), float(n[0]))
    p = det.mid
    tip = np.array([float(p[0]) + 0.008 * float(n[0]), float(p[1]) + 0.008 * float(n[1]), det.top_z - 0.02])
    tip_in_grasp = (np.linalg.inv(api.grasp_pose(side)) @ api.tcp_pose(side))[:3, 3]
    report["press_geometry"] = {"outward_deg": round(math.degrees(phi), 1), "outward_from": source,
                                "tip": [round(float(v), 4) for v in tip]}
    errors = []
    if free_yaw:
        candidates = [(t, None) for t in (0.55, 0.7, 0.4, 0.85)]
    else:
        candidates = [(t, y) for t in (0.7, 0.55, 0.85, 0.4, 1.0) for y in (phi + math.pi / 2.0, phi - math.pi / 2.0)]
    g1 = str(info["embodiment"]) == "g1_omnipicker"
    if not g1:
        way_opt = None
    lid_z = float(can.top_z) if can is not None else None
    if lid_z is None:
        way_opt = None
    for once in (0,):
        for waist in waist_options:
            for tilt, yaw in candidates:
                if api.reply_elapsed() > 55.0:
                    errors.append("planning stopped at the 55 s reply cap")
                    report.setdefault("press_errors", []).extend(errors[-4:])
                    return None
                trial = api.motion().gripper(side, "close", rows=20)
                press = top_down(tip, phi + math.pi / 2.0 if yaw is None else yaw, tilt=tilt, tilt_toward=phi)
                press[:3, 3] = tip - press[:3, :3] @ tip_in_grasp
                above = offset(press, dz=0.07)
                try:
                    if waist is not None:
                        trial.waist(waist)
                    fast = Limits(joint_velocity=2.4, joint_acceleration=9.0)
                    fa = {"free_axis": (0, 0, 1)} if yaw is None else {}
                    if way_opt is not None:
                        way = above.copy()
                        way[0, 3] += way_opt[0] * float(n[0])
                        way[1, 3] += way_opt[0] * float(n[1])
                        way[2, 3] = max(float(above[2, 3]), lid_z + way_opt[1])
                        trial.tcp(side, way, limits=fast, **fa)
                        trial.tcp(side, above, linear=True, limits=Limits(linear_velocity=0.15), **fa)
                        report["press_way"] = [round(float(v), 3) for v in way[:3, 3]] + [way_opt[0], way_opt[1]]
                    else:
                        report["press_way"] = None
                        trial.tcp(side, above, limits=fast, **fa)
                    trial.tcp(side, press, linear=True, limits=Limits(linear_velocity=0.12), **fa)
                    trial.hold(20)
                    report["press_tilt_deg"] = round(math.degrees(tilt), 1)
                    report["press_yaw_deg"] = None if yaw is None else round(math.degrees(yaw), 1)
                    report["press_side"] = side
                    report["press_waist"] = None if waist is None else [round(float(v), 3) for v in waist]
                    return trial
                except MotionError as error:
                    errors.append(f"{side} way {way_opt} tilt {tilt:.2f} yaw {yaw} waist {waist}: {error}")
    report.setdefault("press_errors", []).extend(errors[-4:])
    return None


pressed = False
press_side = press_order[0]


def lid_top():
    d = api.find(["trash can", "red trash can", "pedal trash can"], size=(0.08, 0.35), table_z=table, z_tolerance=0.08)
    return None if d is None else float(d.top_z)


lid0 = None if can is None else float(can.top_z)
if pedal is not None:
    way_options = [(0.10, 0.05), (0.06, 0.02), None] if str(info["embodiment"]) == "g1_omnipicker" else [None]
    for way_opt in way_options:
        for free_yaw in (False, True):
            for side in press_order:
                if api.reply_elapsed() > 55.0:
                    report["press_capped"] = True
                    break
                m = press_motion(side, pedal, free_yaw, way_opt)
                if m is not None:
                    press_side = side
                    m.send(depth_next=True, label="press_pedal")
                    pressed = True
                    break
            if pressed:
                break
        if pressed:
            break
    if pressed and lid0 is not None:
        lt = lid_top()
        report["lid_rise"] = None if lt is None else round(lt - lid0, 4)
        if lt is not None and lt - lid0 < 0.03 and api.rows_left() > 200:
            try:
                deeper = api.motion()
                deeper.tcp(press_side, offset(api.grasp_pose(press_side), dz=-0.015), linear=True, limits=Limits(linear_velocity=0.06))
                deeper.hold(15)
                deeper.send(depth_next=True, label="press_deeper")
                lt2 = lid_top()
                report["lid_rise_2"] = None if lt2 is None else round(lt2 - lid0, 4)
                if lt2 is not None and lt2 - lid0 < 0.02 and api.rows_left() > 150:
                    n_p = outward(pedal)[0]
                    g_now = api.grasp_pose(press_side)
                    up_ = offset(g_now, dz=0.03)
                    over = offset(up_, dx=-0.010 * float(n_p[0]), dy=-0.010 * float(n_p[1]))
                    down = offset(over, dz=-0.045)
                    again = api.motion()
                    again.tcp(press_side, up_, linear=True, limits=Limits(linear_velocity=0.10))
                    again.tcp(press_side, over, linear=True, limits=Limits(linear_velocity=0.10))
                    again.tcp(press_side, down, linear=True, limits=Limits(linear_velocity=0.06))
                    again.hold(15)
                    again.send(depth_next=True, label="press_retry")
                    lt3 = lid_top()
                    report["lid_rise_3"] = None if lt3 is None else round(lt3 - lid0, 4)
            except MotionError as error:
                report["press_deeper_error"] = str(error)[:200]
    if pressed:
        api.stage_done()

if ball is not None and (pressed or pedal is None):
    try:
        if pressed and press_side == pick_side:
            quick = Limits(joint_velocity=2.5, joint_acceleration=10.0)
            plan = api.motion()
            try:
                plan.tcp(pick_side, offset(api.grasp_pose(pick_side), dz=-0.06, frame="tool"), limits=quick)
            except MotionError as error:
                report["retreat_error"] = str(error)
            plan.joints(pick_side, start_arm[pick_side], limits=quick)
            plan.send(depth_next=True, label="leave_pedal")
        ball_h = min(0.06, max(0.03, float(ball.extent[2])))
        BALL_Q = ["paper ball", "white paper ball", "crumpled white paper"]
        tip_frame = "omnipicker" in str(info["embodiment"])
        report["ball_depth"] = round(ball_h if tip_frame else min(0.035, 0.5 * ball_h + 0.008), 4)
        pk = api.pick_object(pick_side, BALL_Q,
                             find={"size": (0.02, 0.10), "table_z": table, "z_tolerance": 0.06},
                             grasp={"depth_into": report["ball_depth"], "floor_clearance": 0.008, "lift": 0.10, "wrist_floor": tip_frame},
                             refine_wrist=0.10)
        report["pick"] = {k: pk.get(k) for k in ("ok", "reason", "attempts", "refined")}
        report["holding"] = bool(pk.get("ok"))
        if pk.get("ok"):
            api.stage_done()
    except MotionError as error:
        report["pick_error"] = str(error)
    if report.get("holding") and pressed and press_side != pick_side and can is not None:
        rim = can.top_z
        n_out = np.array([-1.0, 0.0])
        if pedal is not None:
            n_out = outward(pedal)[0]
        Pc = can.points
        body = Pc[(Pc[:, 2] > table + 0.06) & (Pc[:, 2] < float(can.top_z) - 0.01)]
        if body.shape[0] > 50:
            Pc = body
        side_ax = np.array([-float(n_out[1]), float(n_out[0])])
        along = Pc[:, :2] @ np.asarray(n_out, dtype=float)
        across = Pc[:, :2] @ side_ax
        r_can = 0.5 * float(np.percentile(across, 97) - np.percentile(across, 3))
        near_face = float(np.percentile(along, 97))
        c_along = near_face - r_can
        c_across = 0.5 * float(np.percentile(across, 97) + np.percentile(across, 3))
        cxy = c_along * np.asarray(n_out, dtype=float) + c_across * side_ax
        cx, cy = float(cxy[0]), float(cxy[1])
        report["can_centre"] = {"mid": [round(float(can.mid[0]), 4), round(float(can.mid[1]), 4)], "centre": [round(cx, 4), round(cy, 4)], "r": round(r_can, 4)}
        side_dir = np.array([n_out[1], -n_out[0]]) if pick_side == "right" else np.array([-n_out[1], n_out[0]])
        released = False
        for tilt_, toward_ in ((0.0, 0.0), (0.4, math.pi), (0.4, math.atan2(float(side_dir[1]), float(side_dir[0])))):
            rot_ = top_down((0.0, 0.0, 0.0), 0.0, tilt=tilt_, tilt_toward=toward_)[:3, :3]
            for f_, h_ in ((0.25 * r_can, 0.04), (0.10 * r_can, 0.045), (0.0, 0.05), (0.40 * r_can, 0.035)):
                tgt = np.array([cx + f_ * float(n_out[0]), cy + f_ * float(n_out[1]), rim + h_])
                for app in (n_out, side_dir):
                    pre = tgt + np.array([0.06 * float(app[0]), 0.06 * float(app[1]), 0.02])
                    try:
                        m = api.motion()
                        m.tcp(pick_side, pose(pre, rot_), free_axis=(0.0, 0.0, 1.0), limits=Limits(joint_velocity=2.2, joint_acceleration=8.0))
                        m.tcp(pick_side, pose(tgt, rot_), linear=True, free_axis=(0.0, 0.0, 1.0), limits=Limits(linear_velocity=0.18))
                        m.send(depth_next=True, label="over_can")
                        shift = np.zeros(2)
                        cam = f"hand_{pick_side}"
                        if api.has_depth(cam):
                            Pw = api.points(cam, stride=2)
                            Pw = Pw[np.all(np.isfinite(Pw), axis=1)]
                            ins = Pw[(np.linalg.norm(Pw[:, :2] - tgt[:2], axis=1) < 0.10) & (Pw[:, 2] < rim - 0.03) & (Pw[:, 2] > table + 0.005)]
                            report.setdefault("opening", []).append(int(ins.shape[0]))
                            if ins.shape[0] > 200:
                                oc = 0.5 * (np.percentile(ins[:, :2], 5, axis=0) + np.percentile(ins[:, :2], 95, axis=0))
                                shift = oc - tgt[:2]
                                if float(np.linalg.norm(shift)) > 0.03:
                                    shift = shift * (0.03 / float(np.linalg.norm(shift)))
                                report["opening_shift"] = [round(float(v), 4) for v in shift]
                        m = api.motion()
                        try:
                            m.tcp(pick_side, pose(tgt + np.array([float(shift[0]), float(shift[1]), 0.0]), rot_), linear=True,
                                  free_axis=(0.0, 0.0, 1.0), limits=Limits(linear_velocity=0.15))
                        except MotionError:
                            pass
                        m.gripper(pick_side, "open", rows=12)
                        m.hold(12)
                        up_ = m.grasp_pose(pick_side).copy()
                        up_[2, 3] += 0.06
                        try:
                            m.tcp(pick_side, up_, linear=True, free_axis=(0.0, 0.0, 1.0))
                        except MotionError:
                            pass
                        m.send(depth_next=True, label="drop_ball")
                        report["dropped_at"] = [round(float(v), 4) for v in tgt] + [tilt_, round(math.degrees(math.atan2(float(app[1]), float(app[0]))), 0)]
                        api.stage_done()
                        released = True
                        break
                    except MotionError as error:
                        report.setdefault("drop_errors", []).append(str(error)[:70])
                if released:
                    break
            if released:
                break
report["notes"] = api.notes
report["budget"] = api.budget()["phases"]
api.end_episode()
