# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

start_arm = {s_: np.array(q, dtype=float) for s_, q in api.robot_state()["arm"].items()}
door = api.find_object(["white door", "door"], size=(0.3, 1.5), away_from_grippers=0.0)
report["door"] = None if door is None else door.summary()
handle = api.find_object(["door handle", "door lever", "metal lever handle"], size=(0.03, 0.25), away_from_grippers=0.0)
report["handle"] = None if handle is None else handle.summary()
if door is not None:
    px = float(np.median(door.points[:, 0]))
    py = float(door.mid[1])
    dx = max(0.0, px - 0.80)
    dy = py
    report["move"] = [round(dx, 3), round(dy, 3)]
    try:
        m = api.motion()
        m.base(dx, dy, 0.0, speed=0.3)
        m.hold(60)
        m.send(depth_next=True, label="to_door")
        api.stage_done()
        after = api.find_object(["white door", "door"], size=(0.3, 1.5), away_from_grippers=0.0)
        report["door_after"] = None if after is None else after.summary()
    except MotionError as error:
        report["move_error"] = str(error)[:300]
    if handle is not None and report.get("move") is not None:
        P = handle.points - np.array([float(report["move"][0]), float(report["move"][1]), 0.0])
        again = api.find_object(["door handle", "door lever", "metal lever handle"], size=(0.03, 0.25), away_from_grippers=0.0)
        if again is not None and float(np.linalg.norm(again.mid[:2] - np.median(P[:, :2], axis=0))) < 0.15 and again.points.shape[0] > 30:
            P = again.points
            report["handle_again"] = [round(float(v), 3) for v in again.mid]
        y_free = float(np.percentile(P[:, 1], 97))
        x_front = float(np.percentile(P[:, 0], 5))
        near_end = P[P[:, 1] > y_free - 0.05]
        z_top = float(np.percentile(near_end[:, 2], 90)) if near_end.shape[0] > 10 else float(np.percentile(P[:, 2], 90))
        report["lever"] = {"y_free": round(y_free, 3), "y_min": round(float(np.percentile(P[:, 1], 3)), 3), "x_front": round(x_front, 3),
                           "z_top": round(z_top, 3), "points": int(P.shape[0])}
        slow = Limits(linear_velocity=0.10)
        quick = Limits(linear_velocity=0.18)
        for yaw_, tilt_, inset, push in ((math.pi / 2, 0.97, 0.035, 0.12), (math.pi / 2, 0.67, 0.035, 0.12), (0.0, 0.97, 0.035, 0.12),
                                         (math.pi / 2, 0.97, 0.035, 0.08), (math.pi / 2, 1.27, 0.035, 0.08), (math.pi / 2, 0.97, 0.06, 0.08)):
            px, py = x_front + 0.012, y_free - inset
            try:
                m = api.motion()
                m.gripper("right", "close", rows=15)
                pose = lambda x, z: top_down((x, py, z), yaw_, tilt=tilt_, tilt_toward=math.pi)
                m.tcp("right", pose(px - 0.03, z_top + 0.07))
                m.tcp("right", pose(px, z_top + 0.03), linear=True, limits=quick)
                m.tcp("right", pose(px, z_top - 0.06), linear=True, limits=slow)
                m.tcp("right", pose(px + push, z_top - 0.06), linear=True, limits=slow)
                m.hold(6)
                m.tcp("right", pose(px + push - 0.04, z_top + 0.05), linear=True, limits=quick)
                m.send(depth_next=True, label="unlatch_push")
                report["unlatch"] = [round(yaw_, 2), tilt_, inset, push, m.rows]
                break
            except MotionError as error:
                report.setdefault("unlatch_errors", []).append(str(error)[:70])
    try:
        m = api.motion()
        targets = {}
        for side, sy in (("left", 0.12), ("right", -0.12)):
            t = top_down((0.45, sy, 1.05), 0.0)
            targets[side] = t
        m.tcp(targets, free_axis=(0.0, 0.0, 1.0))
        m.send(depth_next=True, label="arms_narrow")
        report["arms_narrow"] = True
    except MotionError as error:
        report["arms_error"] = str(error)[:160]
    after2 = api.find_object(["white door", "door"], size=(0.3, 1.5), away_from_grippers=0.0)
    gap = 0.80 if after2 is None else float(np.median(after2.points[:, 0]))
    report["door_ahead"] = round(gap, 3)
    plane = 0.80
    if report.get("move") is not None:
        plane = max(0.5, float(np.median(door.points[:, 0])) - float(report["move"][0]))
    report["door_plane"] = round(plane, 3)
    try:
        m = api.motion()
        m.base(float(plane + 0.75), 0.0, 0.0, speed=0.30)
        m.hold(30)
        m.send(depth_next=True, label="through_door")
        api.stage_done()
        m = api.motion()
        for sd in ("left", "right"):
            m.joints(sd, start_arm[sd], limits=Limits(joint_velocity=1.5, joint_acceleration=5.0))
        m.base(0.0, 0.0, -2.0, yaw_speed=0.8)
        m.hold(30)
        m.send(depth_next=True, label="turn_round")
        o = np.array(api.robot_state()["odom"], dtype=float)
        hd = math.degrees(float(o[2])) % 360.0
        report["heading_after_turn"] = round(hd, 1)
        if not (110.0 <= hd <= 250.0):
            extra = math.radians(((200.0 - hd + 180.0) % 360.0) - 180.0)
            m = api.motion()
            m.base(0.0, 0.0, float(extra), yaw_speed=0.8)
            m.hold(30)
            m.send(depth_next=True, label="turn_more")
            report["heading_after_more"] = round(math.degrees(float(api.robot_state()["odom"][2])) % 360.0, 1)
        report["through"] = True
        api.stage_done()
    except MotionError as error:
        report["through_error"] = str(error)[:200]
report["budget"] = api.budget()["phases"]
api.end_episode()
