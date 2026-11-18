# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
active = info["active_arms"]
table = api.table_height("head")
report["embodiment"] = info["embodiment"]
whole = api.find_object(["green scraper", "green squeegee", "scraper"], size=(0.06, 0.30), table_z=table, z_tolerance=0.06)
handle = api.find_object(["green squeegee handle", "scraper handle"], size=(0.03, 0.20), table_z=table, z_tolerance=0.06)
report["whole"] = None if whole is None else whole.summary()
report["handle"] = None if handle is None else handle.summary()
done = False
if handle is not None:
    H = handle.points
    hc = np.median(H, axis=0)
    xy = H[:, :2] - hc[:2]
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    axis = vecs[:, int(np.argmax(vals))]
    proj = xy @ axis
    lo, hi = float(np.percentile(proj, 3)), float(np.percentile(proj, 97))
    junction_hi = True
    if whole is not None:
        wc = whole.mid[:2] - hc[:2]
        junction_hi = float(wc @ axis) > 0.5 * (lo + hi)
    s_j, s_t = (hi, lo) if junction_hi else (lo, hi)
    s_g = s_j + 0.4 * (s_t - s_j)
    g_xy = hc[:2] + axis * s_g
    near = H[np.abs(proj - s_g) < 0.015]
    top = float(np.percentile(near[:, 2], 90)) if near.shape[0] > 10 else handle.top_z
    stem_yaw = math.atan2(float(axis[1]), float(axis[0]))
    report["grasp_xy"] = [round(float(v), 4) for v in g_xy]
    report["stem_yaw_deg"] = round(math.degrees(stem_yaw), 1)
    first = api.choose_arm(g_xy)
    floor = table
    try:
        hv = api.motion()
        hv.tcp(first, top_down((float(g_xy[0]), float(g_xy[1]), top + 0.12), stem_yaw + math.pi / 2.0),
               gripper="open", free_axis=(0.0, 0.0, 1.0))
        hv.send(depth_next=True, label="hover_stem")
        if api.has_depth(f"hand_{first}"):
            wf = api.table_height(f"hand_{first}")
            if math.isfinite(wf) and abs(wf - float(top)) < 0.04:
                floor = wf
                report["wrist_floor"] = round(wf, 4)
    except MotionError as error:
        report["hover_error"] = str(error)[:160]
    for side in [first] + [sd for sd in active if sd != first]:
        try:
            plan = api.top_grasp(side, center=(float(g_xy[0]), float(g_xy[1]), top), top_z=top, bottom_z=floor,
                                 yaw=stem_yaw, depth_into=0.05, floor_clearance=0.003, lift=0.10, above=0.07)
        except MotionError as error:
            report.setdefault("plan_errors", []).append(f"{side}: {str(error)[:160]}")
            continue
        plan.send(depth_next=True, label="pick_scraper")
        report["side"] = side
        report["holding"] = api.holding(side)
        done = True
        break
    if done and report.get("holding"):
        api.stage_done()
report["budget"] = api.budget()["phases"]
api.end_episode()
