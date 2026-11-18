# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

seg0 = api.segment_text(0)
side = api.instructed_arm(seg0) or "left"
table = api.table_height("head")
report["side"] = side
text = seg0.lower()
for lead in ("pick up the ", "pick up ", "grasp the ", "take the "):
    if text.startswith(lead):
        text = text[len(lead):]
for tail in (" with the left gripper", " with the right gripper", " with the left arm", " with the right arm"):
    text = text.replace(tail, "")
phrase = text.strip().rstrip(".")
words = phrase.split()
queries = [phrase]
if len(words) >= 2:
    queries.append(words[0] + " " + words[-2] + " " + words[-1] if len(words) > 3 else " ".join(words[-2:]))
queries.append(" ".join(words[:1] + ["block"]))
report["queries"] = queries
FOLD = 5 if ("star" in phrase or "pentagon" in phrase) else (6 if "hexagon" in phrase else (4 if ("square" in phrase or "cross" in phrase) else (3 if "triangle" in phrase else 1)))
report["fold"] = FOLD


def shape_angle(xy, fold):
    c = np.median(xy, axis=0)
    v = xy - c
    r = np.hypot(v[:, 0], v[:, 1])
    th = np.arctan2(v[:, 1], v[:, 0])
    w = r ** 4
    a = math.atan2(float(np.sum(w * np.sin(fold * th))), float(np.sum(w * np.cos(fold * th)))) / fold
    period = 2 * math.pi / fold
    return a % period, c


box = api.find_object(["shape sorter box", "red box", "wooden box"], size=(0.08, 0.30))
report["box"] = None if box is None else box.summary()
in_box = (lambda c: float(np.linalg.norm(c[:2] - box.mid[:2])) < 0.09) if box is not None else None
block = api.find_object(queries, size=(0.01, 0.07), table_z=table, z_tolerance=0.04, side=side,
                        clip_z=(table - 0.01, table + 0.06), region=(lambda c: not in_box(c)) if in_box else None)
block_angle = None
if block is not None and FOLD > 1:
    topz = block.top_z
    face = block.points[block.points[:, 2] > topz - 0.006]
    if face.shape[0] > 30:
        block_angle, bc = shape_angle(face[:, :2], FOLD)
        report["block_angle_deg"] = round(math.degrees(block_angle), 1)
if block is not None and FOLD > 1:
    try:
        mh = api.motion()
        hov = top_down((float(block.center[0]), float(block.center[1]), float(block.top_z) + 0.12), 0.0)
        mh.tcp(side, hov, gripper="open", free_axis=(0.0, 0.0, 1.0), limits=Limits(joint_velocity=2.4, joint_acceleration=9.0))
        mh.send(depth_next=True, label="look_block")
        wcam = f"hand_{side}"
        best = None
        if api.has_depth(wcam):
            for q in queries:
                for d in api.perceive(q, wcam, max_detections=4):
                    if d.points.shape[0] < 60:
                        continue
                    if float(np.linalg.norm(d.center[:2] - block.center[:2])) > 0.04:
                        continue
                    if not (0.01 < float(max(d.extent[0], d.extent[1])) < 0.08):
                        continue
                    if best is None or d.score > best.score:
                        best = d
                if best is not None:
                    break
        if best is not None:
            wtop = float(best.top_z)
            wface = best.points[best.points[:, 2] > wtop - 0.004]
            report["block_wrist_points"] = [int(best.points.shape[0]), int(wface.shape[0])]
            if wface.shape[0] > 80:
                aw, cw = shape_angle(wface[:, :2], FOLD)
                report["block_angle_head_deg"] = report.get("block_angle_deg")
                report["block_angle_wrist_deg"] = round(math.degrees(aw), 1)
                block_angle = aw
                report["block_angle_deg"] = round(math.degrees(aw), 1)
    except MotionError as error:
        report["look_block_error"] = str(error)[:120]
pick = api.pick_object(side, queries,
                       find={"size": (0.01, 0.07), "table_z": table, "z_tolerance": 0.04, "clip_z": (table - 0.01, table + 0.06),
                             "region": (lambda c: not in_box(c)) if in_box else None},
                       grasp={"depth_into": 0.015, "lift": 0.10, "floor_clearance": 0.010},
                       limits=Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30), clear_view=0.15)
report["pick"] = pick
if pick["ok"] and box is not None:
    api.stage_done()
    note = api.notes.get(f"top_grasp_{side}") or {}
    close_yaw = math.radians(float(note.get("closing_yaw_deg", 0.0)))
    if not api.has_depth():
        api.hold(1, depth_next=True, label="observe")
    shape = phrase.replace("block", "").strip()
    shape_words = [w for w in shape.split() if w not in ("purple", "red", "blue", "green", "yellow", "orange", "pink", "cyan")]
    hole_q = [" ".join(shape_words) + " shaped hole", " ".join(shape_words[-2:]) + " hole", "star shaped hole"]
    box_now = api.find_object(["shape sorter box", "red box", "wooden box"], size=(0.08, 0.30), away_from_grippers=0.0)
    top = (box_now or box).top_z
    hole = api.find_object(hole_q, size=(0.01, 0.08), region=in_box, away_from_grippers=0.0)
    report["hole"] = None if hole is None else hole.summary()
    target_xy = None
    hole_angle = None
    if hole is not None:
        cam = api.camera("head").T_base_optical[:3, 3]
        p = hole.points
        dz = p[:, 2] - cam[2]
        ok = np.abs(dz) > 1e-3
        t = (top - cam[2]) / dz[ok]
        foot = cam[None, :] + (p[ok] - cam[None, :]) * t[:, None]
        if foot.shape[0] > 20:
            if FOLD > 1:
                hole_angle, hc = shape_angle(foot[:, :2], FOLD)
                report["hole_angle_deg"] = round(math.degrees(hole_angle), 1)
            target_xy = np.median(foot[:, :2], axis=0)
            report["hole_centre"] = [round(float(v), 4) for v in target_xy]
    if target_xy is None:
        target_xy = (hole.mid if hole is not None else box.mid)[:2]
    rot = None
    if block_angle is not None and hole_angle is not None:
        period = 2 * math.pi / FOLD
        want = close_yaw + (hole_angle - block_angle)
        turn = (want - close_yaw + math.pi) % (2 * math.pi) - math.pi
        turn = (turn + period / 2.0) % period - period / 2.0
        new_yaw = close_yaw + turn
        report["turn_deg"] = round(math.degrees(new_yaw - close_yaw), 1)
        rot = top_down((0.0, 0.0, 0.0), new_yaw)[:3, :3]
    h = float(block.extent[2]) if block is not None else 0.033
    tcp_z = top + 0.003 + max(0.0, h - 0.015)
    vs = {}
    if FOLD > 1:
        try:
            wcam = f"hand_{side}"
            held0 = api.grasp_pose(side)
            view = held0.copy()
            if rot is not None:
                view[:3, :3] = rot
            toward = -np.array([float(target_xy[0]), float(target_xy[1])])
            toward = toward / max(1e-6, float(np.linalg.norm(toward)))
            view[0, 3] = float(target_xy[0]) + 0.06 * float(toward[0])
            view[1, 3] = float(target_xy[1]) + 0.06 * float(toward[1])
            view[2, 3] = tcp_z + 0.10
            mv = api.motion()
            mv.tcp(side, view)
            mv.send(depth_next=True, label="view_hole")
            if api.has_depth(wcam):
                camc = api.camera(wcam).T_base_optical[:3, 3]
                hole_w = api.find_object(hole_q, camera=wcam, size=(0.01, 0.08), away_from_grippers=0.0,
                                         region=lambda c: float(np.linalg.norm(c[:2] - np.asarray(target_xy)[:2])) < 0.06)
                if hole_w is not None:
                    p = hole_w.points
                    dz_ = p[:, 2] - camc[2]
                    ok_ = np.abs(dz_) > 1e-3
                    t_ = (top - camc[2]) / dz_[ok_]
                    foot = camc[None, :] + (p[ok_] - camc[None, :]) * t_[:, None]
                    if foot.shape[0] > 30:
                        ah, hc = shape_angle(foot[:, :2], FOLD)
                        vs["hole"] = [round(float(hc[0]), 4), round(float(hc[1]), 4), round(math.degrees(ah), 1)]
                blk_w = None
                for q in queries:
                    for d in api.perceive(q, wcam, max_detections=4):
                        P = d.points
                        P = P[P[:, 2] > top + 0.01]
                        if P.shape[0] < 30:
                            continue
                        zt = float(np.percentile(P[:, 2], 95))
                        face = P[P[:, 2] > zt - 0.006]
                        if face.shape[0] > 20:
                            blk_w = face
                            break
                    if blk_w is not None:
                        break
                if blk_w is not None and "hole" in vs:
                    ab, bc = shape_angle(blk_w[:, :2], FOLD)
                    vs["block"] = [round(float(bc[0]), 4), round(float(bc[1]), 4), round(math.degrees(ab), 1)]
                    period = 2 * math.pi / FOLD
                    dang = (ah - ab + period / 2.0) % period - period / 2.0
                    G = api.grasp_pose(side)
                    off = np.array([float(bc[0]) - float(G[0, 3]), float(bc[1]) - float(G[1, 3])])
                    vs["dang_raw_deg"] = round(math.degrees(dang), 1)
                    if abs(math.degrees(dang)) >= 8.0:
                        dang = 0.0
                    c_, s_ = math.cos(dang), math.sin(dang)
                    off_new = np.array([c_ * off[0] - s_ * off[1], s_ * off[0] + c_ * off[1]])
                    new_xy = np.array([float(hc[0]) - off_new[0], float(hc[1]) - off_new[1]])
                    shift = float(np.linalg.norm(new_xy - np.asarray(target_xy, dtype=float)[:2]))
                    vs["shift"] = round(shift, 4)
                    if abs(math.degrees(dang)) < 25.0 and float(np.linalg.norm(off)) < 0.04 and shift < 0.03:
                        Rz = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
                        rot = Rz @ G[:3, :3]
                        target_xy = new_xy
                        vs["dang_deg"] = round(math.degrees(dang), 1)
                        vs["target"] = [round(float(target_xy[0]), 4), round(float(target_xy[1]), 4)]
        except MotionError as error:
            vs["error"] = str(error)[:160]
    report["visual_servo"] = vs
    try:
        held = api.grasp_pose(side)
        R0 = rot if rot is not None else held[:3, :3]
        def at(z, dyaw):
            c, s_ = math.cos(dyaw), math.sin(dyaw)
            Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
            return pose((float(target_xy[0]), float(target_xy[1]), float(z)), Rz @ R0)
        slow = Limits(linear_velocity=0.02, angular_velocity=0.3)
        waist0 = np.array(api.robot_state()["waist"], dtype=float)
        torsos = [None]
        if waist0.size == 5:
            for dyw in (0.2, -0.2, 0.35, -0.35):
                w = waist0.copy()
                w[4] += dyw
                torsos.append(w)
        m = None
        errs = []
        for w in torsos:
            try:
                trial = api.motion()
                if w is not None:
                    trial.waist(w)
                trial.tcp(side, at(tcp_z + 0.05, 0.0))
                trial.tcp(side, at(tcp_z - 0.001, 0.0), linear=True, limits=Limits(linear_velocity=0.08))
                for dz, dyaw in ((-0.004, math.radians(4.0)), (-0.007, math.radians(-4.0)), (-0.010, 0.0)):
                    trial.tcp(side, at(tcp_z + dz, dyaw), linear=True, limits=slow)
                trial.gripper(side, "open", rows=15)
                trial.tcp(side, at(tcp_z + 0.06, 0.0), linear=True)
                m = trial
                report["press_waist"] = None if w is None else round(float(w[4] - waist0[4]), 2)
                break
            except MotionError as error:
                errs.append(str(error)[:80])
        if m is None:
            raise MotionError("press: " + " | ".join(errs[-2:]))
        m.send(depth_next=True, label="press_block")
        report["pressed_at"] = [round(float(target_xy[0]), 4), round(float(target_xy[1]), 4), round(tcp_z, 4)]
        api.stage_done()
    except MotionError as error:
        report["press_error"] = str(error)[:200]
        try:
            api.place_held(side, (float(target_xy[0]), float(target_xy[1]), tcp_z), above=0.06, retreat=0.08,
                           rotation=rot, label="insert_block", descend_rows=60)
            report["released_at"] = [round(float(target_xy[0]), 4), round(float(target_xy[1]), 4), round(tcp_z, 4)]
            api.stage_done()
        except MotionError as error2:
            report["place_error"] = str(error2)[:400]
report["budget"] = api.budget()["phases"]
api.end_episode()
