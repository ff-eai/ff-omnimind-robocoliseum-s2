# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

seg0 = api.segment_text(0)
side = api.instructed_arm(seg0) or "left"
table = api.table_height("head")
report["side"] = side
base = api.find_object(["round metal plate", "grey disc", "metal base plate"], size=(0.10, 0.40))
report["base"] = None if base is None else base.summary()
on_base = (lambda c: float(np.linalg.norm(c[:2] - base.mid[:2])) < 0.5 * float(max(base.extent[0], base.extent[1]))) if base is not None else None
pick = api.pick_object(side, ["loose gear", "metal gear", "small metal ring", "washer"],
                       find={"size": (0.015, 0.08), "table_z": table, "z_tolerance": 0.03, "clip_z": (table - 0.01, table + 0.03),
                             "region": (lambda c: not on_base(c)) if on_base else None},
                       grasp={"depth_into": 0.05, "lift": 0.06, "above": 0.06, "floor_clearance": 0.014},
                       limits=Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30), clear_view=0.15)
report["pick"] = pick
if pick["ok"]:
    api.stage_done()
    GEAR_BELOW = 0.014
    shafts = []
    for q in ("metal shaft", "vertical metal pin", "cylindrical metal post"):
        for d in api.perceive(q, max_detections=4):
            if d.points.shape[0] < 30:
                continue
            h = float(d.top_z) - table
            ex = float(max(d.extent[0], d.extent[1]))
            if not (0.06 < h < 0.26) or not (0.012 < ex < 0.07):
                continue
            if base is not None and float(np.linalg.norm(d.mid[:2] - base.mid[:2])) > 0.5 * float(max(base.extent[0], base.extent[1])):
                continue
            if all(float(np.linalg.norm(d.mid[:2] - o.mid[:2])) > 0.03 for o in shafts):
                shafts.append(d)
        if len(shafts) >= 2:
            break
    report["shafts"] = [[round(float(v), 4) for v in (o.mid[0], o.mid[1], o.top_z)] for o in shafts]
    target = None
    if shafts:
        ref = base.mid[:2] if base is not None else np.mean([o.mid[:2] for o in shafts], axis=0)
        target = min(shafts, key=lambda o: float(np.linalg.norm(o.mid[:2] - ref)))

    def shaft_top(det):
        P = det.points
        top_z = float(np.percentile(P[:, 2], 97))
        rim = P[P[:, 2] > top_z - 0.008]
        c = np.median(rim[:, :2], axis=0) if rim.shape[0] > 10 else det.mid[:2]
        return np.array([float(c[0]), float(c[1]), top_z])

    if target is not None:
        st = shaft_top(target)
        plate_z = table + 0.012 if base is None else min(float(base.top_z), table + 0.03)
        report["shaft_top"] = [round(float(v), 4) for v in st]
        try:
            g = api.grasp_pose(side)
            hover = g.copy()
            hover[0, 3], hover[1, 3], hover[2, 3] = float(st[0]), float(st[1]), float(st[2]) + GEAR_BELOW + 0.035
            m = api.motion()
            m.tcp(side, hover, free_axis=(0.0, 0.0, 1.0))
            m.send(depth_next=True, label="over_shaft")
            cam = f"hand_{side}"
            shift = np.zeros(2)
            if api.has_depth(cam):
                G = api.grasp_pose(side)
                P = api.points(cam, stride=1)
                P = P[np.all(np.isfinite(P), axis=1)]
                near = np.linalg.norm(P[:, :2] - st[:2], axis=1) < 0.03
                ring = P[near & (P[:, 2] > st[2] - 0.012) & (P[:, 2] < st[2] + 0.008)]
                gz = float(G[2, 3]) - GEAR_BELOW
                gear_pts = P[(np.linalg.norm(P[:, :2] - G[:2, 3], axis=1) < 0.045) & (P[:, 2] > gz - 0.003) & (P[:, 2] < gz + 0.024)]
                look = {"ring_points": int(ring.shape[0]), "gear_points": int(gear_pts.shape[0])}
                sc = gc = None
                if ring.shape[0] > 30:
                    x, y = ring[:, 0], ring[:, 1]
                    A = np.column_stack([x, y, np.ones_like(x)])
                    sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
                    cx_, cy_ = 0.5 * float(sol[0]), 0.5 * float(sol[1])
                    r_ = math.sqrt(max(0.0, float(sol[2]) + cx_ * cx_ + cy_ * cy_))
                    if 0.008 < r_ < 0.022 and math.hypot(cx_ - st[0], cy_ - st[1]) < 0.025:
                        sc = np.array([cx_, cy_])
                        look["ring_r"] = round(r_, 4)
                    else:
                        sc = np.median(ring[:, :2], axis=0)
                if gear_pts.shape[0] > 30:
                    lo_, hi_ = np.percentile(gear_pts[:, :2], 3, axis=0), np.percentile(gear_pts[:, :2], 97, axis=0)
                    gc = 0.5 * (lo_ + hi_)
                look["shaft"] = None if sc is None else [round(float(v), 4) for v in sc]
                look["gear"] = None if gc is None else [round(float(v), 4) for v in gc]
                if sc is not None and gc is not None:
                    off = sc - gc
                    look["offset"] = [round(float(v), 4) for v in off]
                    if float(np.linalg.norm(off)) < 0.02:
                        shift = off
                report["wrist_look"] = look
            G = api.grasp_pose(side)
            above = G.copy()
            above[0, 3] += float(shift[0])
            above[1, 3] += float(shift[1])
            enter = above.copy()
            enter[2, 3] = float(st[2]) + GEAR_BELOW - 0.012
            seat = above.copy()
            seat[2, 3] = max(plate_z + GEAR_BELOW - 0.002, float(enter[2, 3]) - 0.20)
            start = above.copy()
            start[2, 3] = float(st[2]) + GEAR_BELOW + 0.004
            m = api.motion()
            m.tcp(side, start, linear=True, limits=Limits(linear_velocity=0.05))
            m.hold(5)
            m.send(label="over_shaft_top")
            z_ref = float(api.grasp_pose(side)[2, 3])
            on_shaft = None
            tries = []
            for dxy in ((0.0, 0.0), (0.0015, 0.0), (0.0, 0.0015), (-0.0015, 0.0), (0.0, -0.0015), (0.0015, 0.0015),
                        (-0.0015, 0.0015), (-0.0015, -0.0015), (0.0015, -0.0015), (0.003, 0.0), (0.0, 0.003), (-0.003, 0.0), (0.0, -0.003)):
                if api.rows_left() < 130:
                    break
                s_ = start.copy()
                s_[0, 3] += dxy[0]
                s_[1, 3] += dxy[1]
                e_ = enter.copy()
                e_[0, 3] = s_[0, 3]
                e_[1, 3] = s_[1, 3]
                m = api.motion()
                if tries:
                    m.tcp(side, s_, linear=True, limits=Limits(linear_velocity=0.05))
                m.tcp(side, e_, linear=True, limits=Limits(linear_velocity=0.03))
                m.hold(5)
                m.send(label="onto_shaft")
                drop = z_ref - float(api.grasp_pose(side)[2, 3])
                tries.append([round(1000 * dxy[0], 1), round(1000 * dxy[1], 1), round(1000 * drop, 1)])
                if drop > 0.011:
                    on_shaft = dxy
                    break
            report["shaft_search"] = tries
            report["on_shaft"] = None if on_shaft is None else [round(1000 * v, 1) for v in on_shaft]
            if on_shaft is not None:
                above[0, 3] += on_shaft[0]
                above[1, 3] += on_shaft[1]
                seat[0, 3] = above[0, 3]
                seat[1, 3] = above[1, 3]
            api.stage_done()
            m = api.motion()
            m.tcp(side, seat, linear=True, limits=Limits(linear_velocity=0.06))
            m.gripper(side, "open", rows=20)
            up = seat.copy()
            up[2, 3] += 0.10
            m.tcp(side, up, linear=True)
            m.send(depth_next=True, label="seat_release")
            report["seat_z"] = round(float(seat[2, 3]), 4)
            api.stage_done()
        except MotionError as error:
            report["shaft_error"] = str(error)[:200]
report["budget"] = api.budget()["phases"]
api.end_episode()
