# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
text = (" ".join(info["segments"]) + " " + str(info.get("prompt", ""))).lower()
direction = -1.0 if "to the right side" in text else 1.0
report["direction"] = "right" if direction < 0 else "left"
table = api.table_height("head")
BOX_Q = ["package box", "cardboard box", "brown box"]
cands = []
for q in BOX_Q:
    for d in api.plausible(api.perceive(q, "head", max_detections=5), size=(0.10, 0.60), table_z=table, z_tolerance=0.06):
        if d.points.shape[0] >= 2500 and float(d.extent[2]) <= 0.25 and float(min(d.extent[0], d.extent[1])) >= 0.08:
            cands.append(d)
box = max(cands, key=lambda d: d.points.shape[0]) if cands else api.find(BOX_Q, size=(0.10, 0.60), table_z=table, z_tolerance=0.06)
report["box_candidates"] = len(cands)
report["table_z"] = round(table, 4)
report["box"] = None if box is None else box.summary()
pts = api.points("head")
surf = pts[(np.abs(pts[:, 2] - table) < 0.015) & (pts[:, 0] > 0.25)]
span = None
if surf.shape[0] > 200:
    y_lo, y_hi = float(np.percentile(surf[:, 1], 1)), float(np.percentile(surf[:, 1], 99))
    span = y_hi - y_lo
    report["table_y"] = [round(y_lo, 3), round(y_hi, 3)]


def squeeze_targets(c, half_y, z, squeeze):
    out = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        approach = np.array([0.0, -sign, 0.0])
        x_axis = np.array([-1.0, 0.0, 0.0])
        y_axis = np.cross(approach, x_axis)
        face = np.array([float(c[0]), float(c[1]) + sign * half_y, z])
        out[side] = pose(face + approach * (0.045 - squeeze), np.column_stack([x_axis, y_axis, approach]))
    return out


if box is not None:
    c = box.mid
    half_y = 0.5 * float(box.extent[1])
    bottom = min(box.bottom_z, table)
    z = bottom + 0.40 * (box.top_z - bottom)
    targets = squeeze_targets(c, half_y, z, 0.032)
    waist0 = np.array(api.robot_state()["waist"], dtype=float)
    slow = Limits(joint_velocity=0.6, joint_acceleration=2.0, linear_velocity=0.10)
    lifted = False
    for dyaw in (0.0, -0.15, -0.30, -0.45, 0.15, -0.60):
        trial = api.motion()
        try:
            if dyaw != 0.0:
                w = waist0.copy()
                w[4] += dyaw
                trial.waist(w)
            trial.tcp({s: offset(t, dy=0.10 * (1 if s == "left" else -1)) for s, t in targets.items()}, free_axis=(0, 0, 1))
            trial.hand("left", "open", rows=10).hand("right", "open", rows=10)
            trial.tcp(targets, free_axis=(0, 0, 1))
            trial.hand("left", wuji_power(0.5), rows=25).hand("right", wuji_power(0.5), rows=25)
            for k in (1, 2, 3, 4):
                trial.tcp({s: offset(trial.grasp_pose(s), dz=0.025) for s in targets}, limits=slow)
        except MotionError as error:
            report.setdefault("lift_errors", []).append(f"{dyaw}: {str(error)[:160]}")
            continue
        trial.send(depth_next=True, label="squeeze_lift")
        report["lift_waist_dyaw"] = dyaw
        api.stage_done()
        lifted = True
        break
    if lifted:
        need = (span / 3.0 + 0.04) if span is not None else 0.40
        if span is not None:
            centre_y = 0.5 * (y_lo + y_hi)
            need = max(need, direction * (centre_y - float(c[1])) + 0.09)
        report["carry_need_m"] = round(need, 3)
        kin = api.episode.kin
        axis_link = kin.model.joints["idx05_body_joint5"].child
        axis_xy = kin.link_pose(axis_link, api.episode.commanded_positions())[:3, 3][:2]
        r = float(np.linalg.norm(np.array([float(c[0]), float(c[1])]) - axis_xy))
        dpsi = min(1.2, need / max(0.3, r))
        report["carry_yaw"] = [round(r, 3), round(dpsi, 3)]
        w = np.array(api.robot_state()["waist"], dtype=float)
        try:
            m = api.motion()
            wk = w.copy()
            wk[4] += direction * dpsi
            m.waist(wk, limits=Limits(joint_velocity=0.3, joint_acceleration=0.6))
            for k in (1, 2, 3):
                m.tcp({s: offset(m.grasp_pose(s), dz=-0.0317) for s in ("left", "right")}, limits=slow)
            m.hand("left", "open", rows=20).hand("right", "open", rows=20)
            m.tcp({s: offset(m.grasp_pose(s), dy=0.07 * (1 if s == "left" else -1)) for s in ("left", "right")})
            m.send(depth_next=False, label="carry_left")
            api.stage_done()
        except MotionError as error:
            report["carry_error"] = str(error)[:300]
report["budget"] = api.budget()["phases"]
api.end_episode()
