# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
report["embodiment"] = info["embodiment"]
side = "right" if "right" in info["active_arms"] else info["active_arms"][0]
table = api.table_height("head")
holder = api.find_object(["pen holder", "pen cup", "pencil holder"], size=(0.03, 0.15), table_z=table, z_tolerance=0.06)
report["holder"] = None if holder is None else holder.summary()
pen = None
if holder is not None and float(np.linalg.norm(api.grasp_pose(side)[:2, 3] - holder.mid[:2])) < 0.22:
    out = 1.0 if side == "left" else -1.0
    for dx, dy, dz in ((0.0, 0.12, 0.05), (-0.06, 0.12, 0.0), (-0.10, 0.08, -0.05), (0.0, 0.15, -0.08)):
        try:
            up = api.motion()
            t = api.grasp_pose(side).copy()
            t[0, 3] += dx
            t[1, 3] += out * dy
            t[2, 3] += dz
            up.tcp(side, t)
            up.send(depth_next=True, label="clear_view")
            again = api.find_object(["pen holder", "pen cup", "pencil holder"], size=(0.03, 0.15), table_z=table, z_tolerance=0.06)
            holder = again if again is not None else holder
            report["cleared"] = [dx, dy, dz]
            break
        except MotionError as error:
            report["clear_error"] = str(error)[:200]
if holder is not None:
    hxy = holder.mid[:2]
    rim = float(holder.top_z)
    near = lambda c: float(np.linalg.norm(c[:2] - hxy)) < 0.05
    for q in ("pen in the pen holder", "pen", "marker pen"):
        cands = []
        for d in api.perceive(q, max_detections=5):
            if d.points.shape[0] < 20 or not near(d.center):
                continue
            pts = d.points[(d.points[:, 2] > rim - 0.01) & (d.points[:, 2] < rim + 0.10)]
            if pts.shape[0] < 10:
                continue
            cands.append((d, pts))
        if cands:
            d, pts = max(cands, key=lambda dp: float(np.percentile(dp[1][:, 2], 95)))
            pen = {"xy": np.median(pts[:, :2], axis=0), "top": float(np.percentile(pts[:, 2], 95)), "query": q}
            break
report["pen"] = None if pen is None else {"xy": [round(float(v), 4) for v in pen["xy"]], "top": round(pen["top"], 4), "query": pen["query"]}
if pen is not None:
    sent = False
    for depth in (0.02, 0.03):
        try:
            plan = api.top_grasp(side, center=(float(pen["xy"][0]), float(pen["xy"][1]), pen["top"]), top_z=pen["top"],
                                 bottom_z=pen["top"] - 0.06, depth_into=depth, lift=0.05, above=0.06,
                                 tilts=(20.0, -20.0, 35.0, -35.0, 50.0, -50.0, 65.0, -65.0, 80.0, -80.0))
        except MotionError as error:
            report.setdefault("plan_errors", []).append(str(error)[:200])
            continue
        plan.send(depth_next=True, label="pick_pen")
        report["holding"] = api.holding(side)
        if report["holding"]:
            api.stage_done()
        sent = True
        break
report["budget"] = api.budget()["phases"]
api.end_episode()
