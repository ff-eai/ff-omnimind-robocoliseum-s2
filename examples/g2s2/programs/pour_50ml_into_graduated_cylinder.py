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
if handle is not None:
    hy, hl, hw = handle.principal_yaw()
    for yaw_try in (hy, hy + math.pi / 2.0):
        try:
            plan = api.top_grasp(side, center=handle.mid, top_z=handle.top_z, bottom_z=handle.top_z - 0.05, yaw=yaw_try,
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
