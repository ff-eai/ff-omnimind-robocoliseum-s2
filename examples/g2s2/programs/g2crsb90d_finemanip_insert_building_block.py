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


def sorter_frame(det):
    P = det.points
    zt = float(np.percentile(P[:, 2], 50))
    Q = P[np.abs(P[:, 2] - zt) < 0.004][:, :2]
    if Q.shape[0] < 300:
        return None

    def spread(th):
        c, s = math.cos(th), math.sin(th)
        u = Q[:, 0] * c + Q[:, 1] * s
        v = -Q[:, 0] * s + Q[:, 1] * c
        return u, v, float(np.percentile(u, 99.5) - np.percentile(u, 0.5) + np.percentile(v, 99.5) - np.percentile(v, 0.5))
    best_th = None
    for th in np.arange(0.0, math.pi / 2.0, math.radians(0.5)):
        e = spread(th)[2]
        if best_th is None or e < best_th[0]:
            best_th = (e, float(th))
    for th in np.arange(best_th[1] - math.radians(0.5), best_th[1] + math.radians(0.51), math.radians(0.05)):
        e = spread(th)[2]
        if e < best_th[0]:
            best_th = (e, float(th))
    th = best_th[1]
    u, v, e = spread(th)
    eu = float(np.percentile(u, 99.5) - np.percentile(u, 0.5))
    evv = float(np.percentile(v, 99.5) - np.percentile(v, 0.5))
    if not (0.08 < eu < 0.30 and 0.08 < evv < 0.30):
        return {"bad_extent": [round(eu, 4), round(evv, 4)]}
    um = 0.5 * float(np.percentile(u, 99.5) + np.percentile(u, 0.5))
    vm = 0.5 * float(np.percentile(v, 99.5) + np.percentile(v, 0.5))
    c, s = math.cos(th), math.sin(th)
    return {"centre": (um * c - vm * s, um * s + vm * c), "yaw90": th, "lid_z": float(np.median(P[np.abs(P[:, 2] - zt) < 0.004][:, 2])),
            "extent": [round(eu, 4), round(evv, 4)]}


sorter = sorter_frame(box) if box is not None else None
if sorter is not None and "centre" in sorter:
    report["sorter"] = {"centre": [round(float(sorter["centre"][0]), 4), round(float(sorter["centre"][1]), 4)],
                        "yaw90_deg": round(math.degrees(sorter["yaw90"]), 2), "lid_z": round(sorter["lid_z"], 4), "extent": sorter["extent"]}
elif sorter is not None:
    report["sorter"] = sorter
block = api.find_object(queries, size=(0.01, 0.07), table_z=table, z_tolerance=0.04, side=side,
                        clip_z=(table - 0.01, table + 0.06), region=(lambda c: not in_box(c)) if in_box else None)
block_angle = None
if block is not None and FOLD > 1:
    topz = block.top_z
    face = block.points[block.points[:, 2] > topz - 0.006]
    if face.shape[0] > 30:
        block_angle, bc = shape_angle(face[:, :2], FOLD)
        report["block_angle_deg"] = round(math.degrees(block_angle), 1)
best = None
wtable = None
if block is not None and FOLD > 1:
    try:
        mh = api.motion()
        hov = top_down((float(block.center[0]), float(block.center[1]), float(block.top_z) + 0.12), 0.0)
        mh.tcp(side, hov, gripper="open", free_axis=(0.0, 0.0, 1.0), limits=Limits(joint_velocity=2.4, joint_acceleration=9.0))
        mh.send(depth_next=True, label="look_block")
        wcam = f"hand_{side}"
        best = None
        if api.has_depth(wcam):
            tw_ = api.table_height(wcam)
            wtable = float(tw_) if (math.isfinite(tw_) and abs(float(tw_) - float(table)) < 0.03) else None
            report["wrist_table"] = None if wtable is None else round(wtable, 4)
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
                if block_angle is None:
                    block_angle = aw
                    report["block_angle_deg"] = round(math.degrees(aw), 1)
    except MotionError as error:
        report["look_block_error"] = str(error)[:120]
lying = None
if best is not None and FOLD == 5:
    P = best.points
    zr = float(np.percentile(P[:, 2], 99.5))
    strip = P[P[:, 2] > zr - 0.005]
    if strip.shape[0] > 40:
        sc = np.mean(strip[:, :2], axis=0)
        ev, evec = np.linalg.eigh(np.cov((strip[:, :2] - sc).T))
        ratio = float(ev[1] / max(float(ev[0]), 1e-12))
        u = evec[:, 1]
        nrm = np.array([-float(u[1]), float(u[0])])
        along = (strip[:, :2] - sc) @ u
        lo_a, hi_a = float(np.percentile(along, 3)), float(np.percentile(along, 97))
        crest = P[P[:, 2] > zr - 0.002]
        across = float(np.mean((crest[:, :2] - sc) @ nrm)) if crest.shape[0] > 5 else 0.0
        ctr = sc + 0.5 * (lo_a + hi_a) * u + across * nrm
        report["ridge"] = [int(strip.shape[0]), round(ratio, 1), round(hi_a - lo_a, 4), round(zr, 4)]
        if ratio > 8.0 and 0.022 < hi_a - lo_a < 0.05:
            lying = {"centre": ctr, "axis_yaw": math.atan2(float(u[1]), float(u[0])), "top": zr, "thick": hi_a - lo_a}
            report["lying"] = {"centre": [round(float(ctr[0]), 4), round(float(ctr[1]), 4)],
                               "axis_deg": round(math.degrees(lying["axis_yaw"]), 1), "top": round(zr, 4)}
if lying is None and best is None and block is not None and FOLD == 5:
    P = block.points
    zr = float(np.percentile(P[:, 2], 99.0))
    strip = P[P[:, 2] > zr - 0.005]
    if strip.shape[0] >= 8:
        sc = np.mean(strip[:, :2], axis=0)
        ev, evec = np.linalg.eigh(np.cov((strip[:, :2] - sc).T))
        ratio = float(ev[1] / max(float(ev[0]), 1e-12))
        u = evec[:, 1]
        along = (strip[:, :2] - sc) @ u
        lo_a, hi_a = float(np.percentile(along, 3)), float(np.percentile(along, 97))
        report["ridge_head"] = [int(strip.shape[0]), round(ratio, 1), round(hi_a - lo_a, 4), round(zr, 4)]
        if ratio > 5.0 and 0.015 < hi_a - lo_a < 0.05:
            ctr = sc + 0.5 * (lo_a + hi_a) * u
            lying = {"centre": ctr, "axis_yaw": math.atan2(float(u[1]), float(u[0])), "top": zr - 0.003, "thick": hi_a - lo_a}
            report["lying"] = {"centre": [round(float(ctr[0]), 4), round(float(ctr[1]), 4)], "from": "head",
                               "axis_deg": round(math.degrees(lying["axis_yaw"]), 1), "top": round(zr - 0.003, 4)}
if lying is not None:
    lying["R"] = min(max((float(lying["top"]) - float(wtable if wtable is not None else table)) / 1.809, 0.008), 0.06)
    report["lying"]["R"] = round(lying["R"], 4)
lying_pick = None
if lying is not None:
    try:
        mp = api.motion()
        api.top_grasp(side, center=(float(lying["centre"][0]), float(lying["centre"][1]), float(lying["top"])),
                      top_z=float(lying["top"]), bottom_z=table, yaw=lying["axis_yaw"], depth_into=0.691 * lying["R"], floor_clearance=0.008,
                      above=0.10, lift=0.10, tilts=(), motion=mp,
                      limits=Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30))
        gnote = api.notes.get(f"top_grasp_{side}") or {}
        mp.send(depth_next=True, label="pick_lying")
        lying_pick = {"held": api.holding(side), "closing_yaw_deg": gnote.get("closing_yaw_deg"), "grasp": gnote.get("grasp"),
                      "waist": gnote.get("waist")}
    except MotionError as error:
        lying_pick = {"error": str(error)[:200]}
    report["pick_lying"] = lying_pick
if lying_pick is not None and "error" not in lying_pick and lying_pick.get("held") is not False:
    pick = {"ok": True, "lying": True, "holding": lying_pick.get("held")}
else:
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
        lid_p = float(sorter["lid_z"]) if (sorter is not None and "lid_z" in sorter) else float(top)
        t = (lid_p - cam[2]) / dz[ok]
        foot = cam[None, :] + (p[ok] - cam[None, :]) * t[:, None]
        if foot.shape[0] > 20:
            if FOLD > 1:
                hole_angle, hc = shape_angle(foot[:, :2], FOLD)
                report["hole_angle_deg"] = round(math.degrees(hole_angle), 1)
            target_xy = np.median(foot[:, :2], axis=0)
            report["hole_centre"] = [round(float(v), 4) for v in target_xy]
    if target_xy is None:
        target_xy = (hole.mid if hole is not None else box.mid)[:2]
    hole_box = None
    lid_z = float(sorter["lid_z"]) if (sorter is not None and "lid_z" in sorter) else float(top)
    rot = None
    if block_angle is not None and hole_angle is not None:
        period = 2 * math.pi / FOLD
        want = close_yaw + (hole_angle - block_angle)
        turn = (want - close_yaw + math.pi) % (2 * math.pi) - math.pi
        turn = (turn + period / 2.0) % period - period / 2.0
        new_yaw = close_yaw + turn
        report["turn_deg"] = round(math.degrees(new_yaw - close_yaw), 1)
        rot = top_down((0.0, 0.0, 0.0), new_yaw)[:3, :3]
    pdet = (pick.get("detection") or {}) if isinstance(pick, dict) else {}
    h = float(block.extent[2]) if block is not None else (float(pdet["extent"][2]) if pdet.get("extent") else 0.0)
    tcp_z = top + 0.003 + max(0.0, h - 0.015)
    vs = {}
    if FOLD > 1 and not pick.get("lying"):
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
                    t_ = (lid_z - camc[2]) / dz_[ok_]
                    foot = camc[None, :] + (p[ok_] - camc[None, :]) * t_[:, None]
                    if foot.shape[0] > 30:
                        ah, hc = shape_angle(foot[:, :2], FOLD)
                        vs["hole"] = [round(float(hc[0]), 4), round(float(hc[1]), 4), round(math.degrees(ah), 1)]
                if hole_box is not None:
                    if "hole" in vs:
                        vs["hole_wrist"] = vs["hole"]
                    hc = np.asarray(hole_box[0], dtype=float)
                    ah = hole_box[1]
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
    if pick.get("lying"):
        lv = {}
        hole_xy = np.asarray(target_xy, dtype=float)[:2]
        H = hole_angle
        if H is None:
            H = 0.0
            lv["no_hole_angle"] = True
        gn = (report.get("pick_lying") or {}).get("grasp")
        star_c = float(lying["top"]) - lying["R"]
        d_in = min(0.5 * lying["R"], max(0.0, (float(gn[2]) - star_c) if gn else 0.309 * lying["R"]))
        period = 2 * math.pi / FOLD
        pref = -1.66 if side == "left" else 1.4
        cands = []
        for k in range(FOLD):
            psi = H + math.pi + k * period
            dpsi = (psi - pref + math.pi) % (2 * math.pi) - math.pi
            cands.append((abs(dpsi), pref + dpsi))
        cands.sort()
        z_rel = lid_z + 0.5 * float(lying.get("thick") or 0.0) + 0.010
        toward = -hole_xy / max(1e-6, float(np.linalg.norm(hole_xy)))
        here0 = api.grasp_pose(side)[:3, 3]
        spots = [(float(hole_xy[0]) + 0.06 * float(toward[0]), float(hole_xy[1]) + 0.06 * float(toward[1]), lid_z + 0.14),
                 (float(hole_xy[0]) + 0.06 * float(toward[0]), float(hole_xy[1]) + 0.06 * float(toward[1]), lid_z + 0.20),
                 (float(hole_xy[0]) + 0.11 * float(toward[0]), float(hole_xy[1]) + 0.11 * float(toward[1]), lid_z + 0.14),
                 (float(here0[0]), float(here0[1]), max(float(here0[2]), lid_z + 0.10))]

        def upright_rot(psi):
            a = np.array([math.cos(psi), math.sin(psi), 0.0])
            xg = np.array([0.0, 0.0, 1.0])
            return np.column_stack([xg, np.cross(a, xg), a])

        d_al = d_in
        d_sd = 0.0

        def upright(psi, xy, z):
            return pose((float(xy[0]) - d_al * math.cos(psi) - d_sd * math.sin(psi), float(xy[1]) - d_al * math.sin(psi) + d_sd * math.cos(psi), float(z)),
                        upright_rot(psi))
        turn_lim = Limits(linear_velocity=0.3, angular_velocity=1.5)
        pitch_lim = Limits(linear_velocity=0.1, angular_velocity=1.0)
        slide_lim = Limits(linear_velocity=0.15, angular_velocity=0.5)
        down_lim = Limits(linear_velocity=0.06, angular_velocity=0.3)
        errs = []
        m = None
        waist0 = np.array(api.robot_state()["waist"], dtype=float)
        torsos = [None]
        if waist0.size == 5:
            for dyw in (0.2, -0.2, 0.35, -0.35):
                wt = waist0.copy()
                wt[4] += dyw
                torsos.append(wt)
        plan_list = [(wt, dist, psi) for wt in torsos for dist, psi in cands]
        for wt, dist, psi in plan_list:
            if m is not None:
                break
            for si, spot in enumerate(spots):
                try:
                    trial = api.motion()
                    if wt is not None:
                        trial.waist(wt)
                    try:
                        trial.tcp(side, top_down(spot, psi - math.pi / 2.0), linear=True, limits=turn_lim)
                    except MotionError:
                        trial = api.motion()
                        if wt is not None:
                            trial.waist(wt)
                        trial.tcp(side, top_down(spot, psi - math.pi / 2.0), limits=Limits(joint_velocity=1.6, joint_acceleration=5.0))
                    trial.tcp(side, pose(spot, upright_rot(psi)), linear=True, limits=pitch_lim)
                    trial.tcp(side, upright(psi, hole_xy, z_rel + 0.06), linear=True, limits=slide_lim)
                    trial.tcp(side, upright(psi, hole_xy, z_rel), linear=True, limits=down_lim)
                    trial.hold(6)
                    trial.gripper(side, "open", rows=15)
                    trial.hold(6)
                    back = upright(psi, hole_xy, z_rel + 0.03)
                    back[0, 3] -= 0.05 * math.cos(psi)
                    back[1, 3] -= 0.05 * math.sin(psi)
                    trial.tcp(side, back, linear=True)
                    m = trial
                    lv["psi_deg"] = round(math.degrees(psi), 1)
                    lv["spot"] = si
                    lv["waist"] = None if wt is None else round(float(wt[4] - waist0[4]), 2)
                    lv["rows"] = int(trial.rows)
                    break
                except MotionError as error:
                    errs.append(f"{math.degrees(psi):.0f}/{si}{'' if wt is None else 'w'}: " + str(error)[:40])
        lv["plan_errors"] = errs[-6:]
        lv["hole"] = [round(float(hole_xy[0]), 4), round(float(hole_xy[1]), 4)]
        lv["H_deg"] = round(math.degrees(H), 1)
        lv["d_in"] = round(d_in, 4)
        lv["z_rel"] = round(z_rel, 4)
        report["lying_insert"] = lv
        if m is None:
            try:
                api.place_held(side, (float(hole_xy[0]), float(hole_xy[1]), tcp_z), above=0.06, retreat=0.08,
                               label="insert_block", descend_rows=60)
                api.stage_done()
            except MotionError as error2:
                report["place_error"] = str(error2)[:300]
        else:
            sent = False
            try:
                p1 = api.motion()
                if lv.get("waist") is not None:
                    wt1 = waist0.copy()
                    wt1[4] += float(lv["waist"])
                    p1.waist(wt1)
                psi1 = math.radians(float(lv["psi_deg"]))
                spot1 = spots[int(lv["spot"])]
                try:
                    p1.tcp(side, top_down(spot1, psi1 - math.pi / 2.0), linear=True, limits=turn_lim)
                except MotionError:
                    p1 = api.motion()
                    if lv.get("waist") is not None:
                        p1.waist(wt1)
                    p1.tcp(side, top_down(spot1, psi1 - math.pi / 2.0), limits=Limits(joint_velocity=1.6, joint_acceleration=5.0))
                p1.tcp(side, pose(spot1, upright_rot(psi1)), linear=True, limits=pitch_lim)
                p1.hold(8)
                p1.send(depth_next=True, label="upright_star")
                G_ = api.grasp_pose(side)
                a_ = G_[:3, 2]
                c_ = G_[:3, 1]
                pts = None
                for q in queries:
                    for d in api.perceive(q, max_detections=4):
                        if d.points.shape[0] >= 30 and float(np.linalg.norm(d.mid[:3] - G_[:3, 3])) < 0.08:
                            pts = d.points if pts is None or d.points.shape[0] > pts.shape[0] else pts
                    if pts is not None:
                        break
                if pts is not None:
                    rel = pts - G_[:3, 3]
                    rel = rel[np.linalg.norm(rel, axis=1) < 0.08]
                    far = float(np.percentile(rel @ a_, 97))
                    d_al_m = far - 0.809 * float(lying["R"])
                    d_sd_m = float(np.median(rel @ c_))
                    bottom = float(np.percentile(pts[:, 2], 90)) - float(lying.get("thick") or 0.0)
                    dz_m = float(G_[2, 3]) - bottom
                    lv["in_hand"] = {"d_al": round(d_al_m, 4), "d_sd": round(d_sd_m, 4), "dz": round(dz_m, 4), "n": int(rel.shape[0])}
                    if 0.0 <= d_al_m < 0.04 and abs(d_sd_m) < 0.012 and -0.02 < dz_m < 0.06:
                        d_al, d_sd = d_al_m, d_sd_m
                        z_rel = lid_z + 0.010 + dz_m
                p2 = api.motion()
                p2.tcp(side, upright(psi1, hole_xy, z_rel + 0.06), linear=True, limits=slide_lim)
                p2.tcp(side, upright(psi1, hole_xy, z_rel), linear=True, limits=down_lim)
                p2.hold(6)
                p2.gripper(side, "open", rows=15)
                p2.hold(6)
                back2 = upright(psi1, hole_xy, z_rel + 0.03)
                back2[0, 3] -= 0.05 * math.cos(psi1)
                back2[1, 3] -= 0.05 * math.sin(psi1)
                p2.tcp(side, back2, linear=True)
                p2.send(depth_next=True, label="insert_lying")
                sent = True
            except MotionError as error:
                lv["split_error"] = str(error)[:100]
            report["lying_insert"] = lv
            if not sent:
                m.send(depth_next=True, label="insert_lying")
            api.stage_done()
    else:
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
