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
        if (d.points.shape[0] >= 2500 and float(d.extent[2]) <= 0.25 and float(min(d.extent[0], d.extent[1])) >= 0.08
                and float(d.top_z) - table >= 0.06):
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


BIG_SHIFT = 0.06
BIG_Z = 0.03


def big_box_fit(points, table_z):
    P = points[(points[:, 2] > table_z + 0.02) & (points[:, 2] < float(np.percentile(points[:, 2], 99)) + 0.03)]
    top = float(np.percentile(P[:, 2], 97))
    T = P[P[:, 2] > top - 0.015]
    if T.shape[0] < 200:
        T = P
    xy = T[:, :2] - np.median(T[:, :2], axis=0)
    best = None
    for k in np.arange(0.0, 90.0, 1.0):
        a = math.radians(float(k))
        su, sv = xy @ np.array([math.cos(a), math.sin(a)]), xy @ np.array([-math.sin(a), math.cos(a)])
        area = float(np.percentile(su, 99) - np.percentile(su, 1)) * float(np.percentile(sv, 99) - np.percentile(sv, 1))
        if best is None or area < best[0]:
            best = (area, k)
    phi = best[1] if best[1] < 45 else best[1] - 90
    return (float(np.percentile(P[:, 0], 1)), float(np.percentile(T[:, 0], 99.5)), float(np.percentile(T[:, 1], 2)),
            float(np.percentile(T[:, 1], 98)), top, float(phi))


if box is not None:
    c = box.mid
    half_y = 0.5 * float(box.extent[1])
    bottom = min(box.bottom_z, table)
    z = bottom + 0.40 * (box.top_z - bottom)
    x_near, x_far, yb_lo, yb_hi, top_u, phi = big_box_fit(api.camera("head").points_from_mask(box.mask), table)
    big = ((x_far - x_near) >= 0.30 or (top_u - table) >= 0.19) and abs(phi) <= 12.0
    report["big"] = {"big": bool(big), "x": [round(x_near, 3), round(x_far, 3)], "y": [round(yb_lo, 3), round(yb_hi, 3)],
                     "top": round(top_u, 3), "phi_deg": phi}
    if big:
        c = np.array([0.5 * (x_near + x_far) + BIG_SHIFT, 0.5 * (yb_lo + yb_hi), float(c[2])])
        half_y = 0.5 * (yb_hi - yb_lo)
        z = 0.5 * (table + top_u) + BIG_Z
    targets = squeeze_targets(c, half_y, z, 0.0)
    waist0 = np.array(api.robot_state()["waist"], dtype=float)
    slow = Limits(joint_velocity=0.6, joint_acceleration=2.0, linear_velocity=0.10)
    plan = [(d, 0.0) for d in (0.0, -0.15, -0.30, -0.45, 0.15, -0.60)]
    if big:
        kin0 = api.episode.kin
        axis0 = kin0.link_pose(kin0.model.joints["idx05_body_joint5"].child, api.episode.commanded_positions())[:3, 3][:2]
        bearing = math.atan2(float(c[1]) - float(axis0[1]), float(c[0]) - float(axis0[0]))
        yaw0 = max(-0.60, min(0.60, round(0.5 * bearing / 0.15) * 0.15))
        yaws = sorted({round(yaw0 + d, 2) for d in (0.0, 0.15, -0.15, 0.30, -0.30)} | {0.0}, key=lambda v: (abs(v - yaw0), abs(v)))
        plan = [(y, 0.0) for y in yaws[:4]] + [(y, 0.15) for y in yaws[:3]] + [(y, 0.30) for y in yaws[:2]]
    need = None
    if span is not None:
        edge = y_hi if direction > 0 else y_lo
        target_y = edge - direction * (half_y + 0.05)
        need = max(0.0, direction * (target_y - float(c[1])))
    report["carry_need_m"] = None if need is None else round(need, 3)
    kin = api.episode.kin
    axis_link = kin.model.joints["idx05_body_joint5"].child
    axis_xy = kin.link_pose(axis_link, api.episode.commanded_positions())[:3, 3][:2]
    r = float(np.linalg.norm(np.array([float(c[0]), float(c[1])]) - axis_xy))
    dpsi = 0.0 if need is None else min(1.2, need / max(0.3, r))
    report["carry_yaw"] = [round(r, 3), round(dpsi, 3)]
    CARRIES = (("full", 1.0, True), ("no_back_off", 1.0, False), ("turn_85", 0.85, False)) if need is not None else ()

    def build(dyaw, lean, carry):
        m = api.motion()
        w = waist0.copy()
        if dyaw != 0.0 or lean != 0.0:
            w[4] += dyaw
            w[2] += lean
            m.waist(w)
        m.tcp({s: offset(t, dy=0.10 * (1 if s == "left" else -1)) for s, t in targets.items()}, free_axis=(0, 0, 1))
        m.hand("left", "open", rows=10).hand("right", "open", rows=10)
        m.tcp(targets, free_axis=(0, 0, 1))
        m.hand("left", wuji_power(0.5), rows=25).hand("right", wuji_power(0.5), rows=25)
        for k in (1, 2, 3, 4):
            m.tcp({s: offset(m.grasp_pose(s), dz=0.025) for s in targets}, limits=slow)
        if carry is not None:
            name, turn_share, back_off = carry
            wk = w.copy()
            wk[4] += direction * dpsi * turn_share
            m.waist(wk, limits=Limits(joint_velocity=0.3, joint_acceleration=0.6))
            for k in (1, 2, 3):
                m.tcp({s: offset(m.grasp_pose(s), dz=-0.0317) for s in ("left", "right")}, limits=slow)
            m.hand("left", "open", rows=20).hand("right", "open", rows=20)
            if back_off:
                m.tcp({s: offset(m.grasp_pose(s), dy=0.07 * (1 if s == "left" else -1)) for s in ("left", "right")})
        return m

    chosen = None
    for dyaw, lean in plan:
        try:
            build(dyaw, lean, None)
        except MotionError as error:
            report.setdefault("lift_errors", []).append(f"{dyaw}/{lean}: {str(error)[:140]}")
            continue
        chosen = (dyaw, lean)
        break
    if chosen is not None:
        dyaw, lean = chosen
        report["lift_waist_dyaw"] = dyaw
        report["lift_lean"] = lean
        motion = None
        for carry in CARRIES:
            try:
                motion = build(dyaw, lean, carry)
            except MotionError as error:
                report.setdefault("carry_errors", []).append(f"{carry[0]}: {str(error)[:160]}")
                continue
            report["carry_variant"] = carry[0] + " (one reply)"
            break
        if motion is not None:
            motion.send(depth_next=False, label="pick_carry")
            api.stage_done()
        else:
            build(dyaw, lean, None).send(depth_next=True, label="squeeze_lift")
            api.stage_done()
            w = np.array(api.robot_state()["waist"], dtype=float)
            for variant, turn_share, back_off in CARRIES:
                try:
                    m = api.motion()
                    wk = w.copy()
                    wk[4] += direction * dpsi * turn_share
                    m.waist(wk, limits=Limits(joint_velocity=0.3, joint_acceleration=0.6))
                    for k in (1, 2, 3):
                        m.tcp({s: offset(m.grasp_pose(s), dz=-0.0317) for s in ("left", "right")}, limits=slow)
                    m.hand("left", "open", rows=20).hand("right", "open", rows=20)
                    if back_off:
                        m.tcp({s: offset(m.grasp_pose(s), dy=0.07 * (1 if s == "left" else -1)) for s in ("left", "right")})
                    m.send(depth_next=False, label="carry_left")
                    api.stage_done()
                    report["carry_variant"] = variant + " (second reply)"
                    break
                except MotionError as error:
                    report.setdefault("carry_errors", []).append(f"{variant} (second reply): {str(error)[:160]}")
report["budget"] = api.budget()["phases"]
api.end_episode()
