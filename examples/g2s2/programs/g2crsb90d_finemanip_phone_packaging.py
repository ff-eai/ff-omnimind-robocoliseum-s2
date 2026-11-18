# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

side = "right"
table = api.table_height("head")
report["table_z"] = None if table is None else round(float(table), 4)
phone = api.find_object(["black smartphone", "black mobile phone", "phone"], size=(0.04, 0.20), table_z=table, z_tolerance=0.03)
report["phone_seen"] = None if phone is None else phone.summary()


def base_box():
    best = None
    for d in api.plausible(api.perceive("open white box", max_detections=5), size=(0.05, 0.20), table_z=table, z_tolerance=0.03):
        h = float(d.top_z) - float(table)
        if not (0.02 < h < 0.08) or d.points.shape[0] < 200:
            continue
        if best is None or d.score > best[1].score:
            rim_z = float(np.percentile(d.points[:, 2], 95))
            best = (float(d.score), d, rim_z, True)
    return best


grasp = {"tip_clearance": 0.003, "lift": 0.08, "above": 0.06}
pick = api.pick_object(side, ["black smartphone", "black mobile phone", "phone"],
                       find={"size": (0.04, 0.20), "table_z": table, "z_tolerance": 0.03},
                       grasp=grasp,
                       limits=Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30), clear_view=-1.0, refine_wrist=0.10)
report["pick"] = {k: pick.get(k) for k in ("ok", "reason", "attempts", "wrist_floor")}
if pick["ok"]:
    api.stage_done()
    if not api.has_depth():
        api.hold(1, depth_next=True, label="observe")
    bb = base_box()
    report["base_box"] = None if bb is None else {"det": bb[1].summary(), "rim_z": round(bb[2], 4), "hollow": bb[3]}
    if bb is not None:
        d = bb[1]
        rim_z = bb[2]
        P = d.points
        rim = P[P[:, 2] > rim_z - 0.008]
        yaw_box = float(d.principal_yaw()[0])
        ax = np.array([math.cos(yaw_box), math.sin(yaw_box)])
        nx = np.array([-ax[1], ax[0]])
        rel = rim[:, :2] - np.median(rim[:, :2], axis=0)
        a_ = rel @ ax
        n_ = rel @ nx
        c = np.median(rim[:, :2], axis=0) + ax * 0.5 * float(np.percentile(a_, 98) + np.percentile(a_, 2)) + nx * 0.5 * float(np.percentile(n_, 98) + np.percentile(n_, 2))
        report["box_centre"] = [round(float(v), 4) for v in c] + [round(math.degrees(yaw_box), 1)]
        G = api.grasp_pose(side)
        cy0 = yaw_box + math.pi / 2.0
        cur = math.atan2(float(G[1, 1]), float(G[0, 1]))
        cands = sorted((cy0, cy0 + math.pi), key=lambda y: abs((y - cur + math.pi) % (2 * math.pi) - math.pi))
        released = False
        for cyaw in cands:
            try:
                z_rel = rim_z + 0.006 + 0.005
                m = api.motion()
                m.tcp(side, top_down((float(c[0]), float(c[1]), z_rel + 0.06), cyaw), limits=Limits(joint_velocity=2.0, joint_acceleration=8.0))
                m.send(depth_next=True, label="over_box")
                cam = f"hand_{side}"
                shift = np.zeros(2)
                if api.has_depth(cam):
                    Pw = api.points(cam, stride=2)
                    Pw = Pw[np.all(np.isfinite(Pw), axis=1)]
                    ins = Pw[(np.linalg.norm(Pw[:, :2] - c, axis=1) < 0.09) & (Pw[:, 2] < rim_z - 0.015) & (Pw[:, 2] > float(table) - 0.005)]
                    rimw = Pw[(np.linalg.norm(Pw[:, :2] - c, axis=1) < 0.11) & (np.abs(Pw[:, 2] - rim_z) < 0.006)]
                    report["wrist_box"] = [int(ins.shape[0]), int(rimw.shape[0])]
                    if rimw.shape[0] > 150:
                        rel = rimw[:, :2] - c
                        aa, nn = rel @ ax, rel @ nx
                        span_a = float(np.percentile(aa, 98) - np.percentile(aa, 2))
                        span_n = float(np.percentile(nn, 98) - np.percentile(nn, 2))
                        report["wrist_rim_span"] = [round(span_a, 3), round(span_n, 3)]
                        if 0.13 < span_a < 0.17 and 0.06 < span_n < 0.09:
                            da = 0.5 * float(np.percentile(aa, 98) + np.percentile(aa, 2))
                            dn = 0.5 * float(np.percentile(nn, 98) + np.percentile(nn, 2))
                            shift = ax * da + nx * dn
                            if float(np.linalg.norm(shift)) > 0.010:
                                shift = shift * (0.010 / float(np.linalg.norm(shift)))
                            report["wrist_shift"] = [round(float(v), 4) for v in shift]
                tgt = c + shift
                m = api.motion()
                m.tcp(side, top_down((float(tgt[0]), float(tgt[1]), z_rel), cyaw), linear=True, limits=Limits(linear_velocity=0.05))
                m.hold(8)
                m.gripper(side, "open", rows=10)
                m.hold(12)
                m.tcp(side, top_down((float(tgt[0]), float(tgt[1]), z_rel + 0.08), cyaw), linear=True)
                m.send(depth_next=True, label="drop_in_box")
                report["dropped_at"] = [round(float(v), 4) for v in tgt] + [round(z_rel, 4), round(math.degrees(cyaw), 1)]
                released = True
                api.stage_done()
                break
            except MotionError as error:
                report.setdefault("place_errors", []).append(str(error)[:100])
report["budget"] = api.budget()["phases"]
api.end_episode()
