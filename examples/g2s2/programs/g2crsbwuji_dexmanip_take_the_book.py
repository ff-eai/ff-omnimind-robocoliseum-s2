# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

table = api.table_height("head")
books = []
for q in ("book spine", "book"):
    for d in api.perceive(q, max_detections=8):
        if d.points.shape[0] < 50:
            continue
        if float(d.extent[2]) < 0.10 or float(d.extent[1]) > 0.08:
            continue
        if all(float(abs(d.mid[1] - b.mid[1])) > 0.02 for b in books):
            books.append(d)
report["books"] = [b.summary()["mid"] for b in books]
side = api.instructed_arm(api.segment_text(0)) or ("right" if "right arm" in str(api.task_info().get("prompt", "")).lower() else "left")
report["side"] = side
sgn = 1.0 if side == "left" else -1.0
cands = sorted([b for b in books if 0.03 < sgn * float(b.mid[1]) < 0.40], key=lambda b: abs(sgn * float(b.mid[1]) - 0.14))
done = False
waist0 = np.array(api.robot_state()["waist"], dtype=float)
waists = [None]
for j1, j2, lean in ((-0.6, 0.9, 0.30), (-0.6, 0.9, 0.45), (-0.4, 0.6, 0.35), (-0.838, 1.249, 0.40), (-0.3, 0.45, 0.25),
                     (-0.6, 0.9, 0.60), (-0.2, 0.3, 0.40)):
    waists.append([j1, j2, lean - j1 - j2, float(waist0[3]), float(waist0[4])])
for b in cands[:3]:
    spine_x = float(np.percentile(b.points[:, 0], 10))
    top = float(b.top_z)
    bottom = float(b.bottom_z)
    y = float(b.mid[1])
    report.setdefault("tries", []).append({"book_y": round(y, 3), "spine_x": round(spine_x, 3), "top": round(top, 3)})
    rot0 = np.column_stack([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    variants = [(fu, roll) for fu in (0.0, 0.015, 0.035) for roll in (0.0, 0.35, -0.35, 0.7, -0.7, 1.0, -1.0)]
    for w, finger_up, roll in [(w, fu, r) for w in waists for fu, r in variants]:
        rot = rot0 @ rotz(sgn * roll)[:3, :3]
        origin = np.array([spine_x - 0.005, y, top - finger_up])
        g = pose(origin, rot)
        h = top - bottom
        try:
            m = api.motion()
            if w is not None:
                m.waist(w)
            m.hand(side, "open", rows=10)
            m.tcp(side, offset(g, dx=-0.10))
            m.tcp(side, g, linear=True, limits=Limits(linear_velocity=0.10))
            curl = wuji_power(0.9)
            m.hand(side, curl, rows=30)
            for a_deg in (12, 24, 36, 48, 55):
                a = math.radians(a_deg)
                m.tcp(side, offset(g, dx=-h * math.sin(a), dz=-h * (1.0 - math.cos(a))), limits=Limits(joint_velocity=0.6))
            m.send(depth_next=True, label="tilt_book")
            report["tilt_variant"] = [finger_up, roll, None if w is None else [round(v, 3) for v in w]]
            done = True
            break
        except MotionError as error:
            errs = report.setdefault("errors", [])
            if len(errs) < 6:
                errs.append(str(error)[:160])
    if done:
        api.stage_done()
        break
report["budget"] = api.budget()["phases"]
api.end_episode()
