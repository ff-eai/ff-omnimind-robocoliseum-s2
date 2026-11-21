# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
active = info["active_arms"]
report["embodiment"] = info["embodiment"]
seg0 = api.segment_text(0)
side = api.instructed_arm(seg0) or ("left" if "left" in active else active[0])
table = api.table_height("head")
cyl = api.find_object(["graduated cylinder", "measuring cylinder", "tall glass cylinder"], size=(0.02, 0.15))
jug = api.find_object(["measuring jug", "measuring cup with handle", "plastic jug", "beaker"], size=(0.04, 0.20),
                      table_z=table, z_tolerance=0.05, exclude=[cyl.mid] if cyl is not None else [], exclude_radius=0.05)
report["cylinder"] = None if cyl is None else cyl.summary()
report["jug"] = None if jug is None else jug.summary()
handle = None
if jug is not None:
    for d in api.perceive("white handle", max_detections=3) + api.perceive("white plastic handle", max_detections=4) + \
             api.perceive("jug handle", max_detections=4):
        e = d.extent
        if d.points.shape[0] < 40 or float(e[2]) < 0.015 or float(max(e[0], e[1])) > 0.06:
            continue
        if float(np.linalg.norm(d.mid[:2] - jug.mid[:2])) > 0.12:
            continue
        handle = d
        break
report["handle"] = None if handle is None else handle.summary()
sent = False
OMNI = info["embodiment"] in ("g2_omnipicker", "g1_omnipicker")
if OMNI and handle is not None and jug is not None:
    P_h = handle.points
    rr = np.hypot(P_h[:, 0] - float(jug.mid[0]), P_h[:, 1] - float(jug.mid[1]))
    outer = P_h[rr >= np.percentile(rr, 70)] if P_h.shape[0] >= 20 else P_h
    if outer.shape[0] >= 8:
        bar = np.median(outer[:, :2], axis=0)
        z_tcp = 0.5 * (float(np.percentile(outer[:, 2], 10)) + float(np.percentile(outer[:, 2], 90)))
    else:
        bar = np.asarray(handle.mid[:2], dtype=float).copy()
        z_tcp = 0.5 * (float(handle.top_z) + float(handle.bottom_z))
    u = bar - np.asarray(jug.mid[:2], dtype=float)
    u = u / max(1e-6, float(np.linalg.norm(u)))
    report["grip_bar"] = [round(float(bar[0]), 4), round(float(bar[1]), 4), round(z_tcp, 4)]
    cur_y = api.motion().grasp_pose(side)[:3, 1]
    w0 = api.motion().waist_end
    wyaws = (0.0, 0.2, 0.35) if (w0 is not None and len(w0) == 5) else (0.0,)
    sh = api.shoulder(side)
    wsign = -1.0 if float(bar[1]) < float(sh[1]) else 1.0
    cands = [(wy, dd, bb) for wy in wyaws for dd in (0.0, 15.0, 30.0, 45.0) for bb in (0.0, 25.0)]
    for wy, down_deg, beta_deg in cands:
        dn = math.radians(down_deg)
        a0 = -u
        best_a = a0
        if beta_deg > 0.0:
            bt = math.radians(beta_deg)
            ra = np.array([math.cos(bt) * a0[0] - math.sin(bt) * a0[1], math.sin(bt) * a0[0] + math.cos(bt) * a0[1]])
            rb = np.array([math.cos(-bt) * a0[0] - math.sin(-bt) * a0[1], math.sin(-bt) * a0[0] + math.cos(-bt) * a0[1]])
            best_a = ra if ra[0] > rb[0] else rb
        tang = np.array([-best_a[1], best_a[0], 0.0])
        if float(tang @ cur_y) < 0.0:
            tang = -tang
        za = np.array([best_a[0] * math.cos(dn), best_a[1] * math.cos(dn), -math.sin(dn)])
        xa = np.cross(tang, za)
        grasp = pose((float(bar[0]), float(bar[1]), z_tcp), np.column_stack([xa, tang, za]))
        pre = grasp.copy()
        pre[:3, 3] = grasp[:3, 3] - za * 0.10
        high = offset(pre, dz=0.08)
        up = offset(grasp, dz=0.06)
        try:
            m = api.motion()
            if wy > 0.0:
                wt = np.asarray(w0, dtype=np.float64).copy()
                wt[4] += wsign * wy
                m.tcp(side, high, gripper="open", waist=wt)
            else:
                m.tcp(side, high, gripper="open")
            m.tcp(side, pre, linear=True)
            m.tcp(side, grasp, linear=True, limits=Limits(joint_velocity=api.limits.joint_velocity,
                                                          joint_acceleration=api.limits.joint_acceleration,
                                                          linear_velocity=0.10, angular_velocity=api.limits.angular_velocity))
            m.hold(10)
            m.gripper(side, "close")
            m.hold(20)
            m.tcp(side, up, linear=True)
        except MotionError as error:
            report.setdefault("plan_errors", []).append(f"grip bar w{wy} d{down_deg:.0f} b{beta_deg:.0f}: {str(error)[:80]}")
            continue
        m.send(depth_next=True, label="pick_jug_grip_bar")
        report["grip"] = f"grip_bar_w{wy}_d{down_deg:.0f}_b{beta_deg:.0f}"
        report["holding"] = api.holding(side)
        if report["holding"]:
            api.stage_done()
        sent = True
        break
if handle is not None and not sent:
    hy, hl, hw = handle.principal_yaw()
    for yaw_try in (hy, hy + math.pi / 2.0):
        try:
            plan = api.top_grasp(side, center=handle.mid, top_z=handle.top_z, bottom_z=handle.bottom_z, yaw=yaw_try,
                                 depth_into=0.02, lift=0.06, above=0.07, settle_after_close=15)
        except MotionError as error:
            report.setdefault("plan_errors", []).append(f"handle: {str(error)[:160]}")
            continue
        plan.send(depth_next=True, label="pick_jug_handle")
        report["grip"] = "handle"
        report["holding"] = api.holding(side)
        if report["holding"]:
            api.stage_done()
        sent = True
        break
if jug is not None and not sent:
    c = jug.mid
    radius = min(max(0.5 * float(min(jug.extent[0], jug.extent[1])), 0.02), 0.07)
    sh = api.shoulder(side)
    to_robot = np.array([float(sh[0] - c[0]), float(sh[1] - c[1])])
    to_robot = to_robot / max(1e-6, float(np.linalg.norm(to_robot)))
    for ang in (0.0, 0.6, -0.6, 1.2, -1.2):
        d = np.array([math.cos(ang) * to_robot[0] - math.sin(ang) * to_robot[1], math.sin(ang) * to_robot[0] + math.cos(ang) * to_robot[1]])
        rim = np.array([float(c[0] + d[0] * (radius - 0.004)), float(c[1] + d[1] * (radius - 0.004)), jug.top_z])
        radial = math.atan2(float(d[1]), float(d[0]))
        try:
            plan = api.top_grasp(side, center=rim, top_z=jug.top_z, bottom_z=table, yaw=radial + math.pi / 2.0,
                                 depth_into=0.03, lift=0.06, above=0.08, settle_after_close=15)
        except MotionError as error:
            report.setdefault("plan_errors", []).append(f"{ang}: {str(error)[:160]}")
            continue
        plan.send(depth_next=True, label="pick_jug_rim")
        report["rim_angle"] = ang
        report["holding"] = api.holding(side)
        if report["holding"]:
            api.stage_done()
        sent = True
        break
report["budget"] = api.budget()["phases"]
api.end_episode()
