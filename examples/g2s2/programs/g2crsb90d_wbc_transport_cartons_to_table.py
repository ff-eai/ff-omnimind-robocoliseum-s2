# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

QUERIES = ["small cardboard box", "box", "white box", "brown cardboard box"]
fast = Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30)
waist_lim = Limits(joint_velocity=1.0, joint_acceleration=3.0)
w0 = [float(v) for v in api.robot_state()["waist"]]
start_arm = {s: np.array(q, dtype=float) for s, q in api.robot_state()["arm"].items()}
WAIST_LOW = [-0.9, 2.2, -0.3, w0[3], w0[4]]
WAIST_DROP = [-0.7, 1.5, 0.0, w0[3], w0[4]]


def odom():
    return np.array(api.robot_state()["odom"], dtype=float)


def parcels_now(shelf_z):
    found = []
    for q in QUERIES:
        for d in api.plausible(api.perceive(q, max_detections=6), size=(0.03, 0.11), table_z=shelf_z, z_tolerance=0.06,
                               away_from_grippers=0.08):
            if d.points.shape[0] < 30 or not (0.03 < float(max(d.extent[0], d.extent[1])) < 0.11):
                continue
            if abs(float(d.bottom_z) - shelf_z) > 0.05:
                continue
            if float(np.min(d.extent)) < 0.032:
                continue
            hands = [api.grasp_pose(sd)[:3, 3] for sd in ("left", "right")] + [api.tcp_pose(sd)[:3, 3] for sd in ("left", "right")]
            if min(float(np.linalg.norm(d.mid - h)) for h in hands) < 0.10:
                continue
            if all(float(np.linalg.norm(d.mid[:2] - f.mid[:2])) > 0.04 for f in found):
                found.append(d)
    return found


def carry_pose():
    m = api.motion()
    targets = {}
    for side, sy in (("left", 0.22), ("right", -0.22)):
        t = top_down((0.38, sy, 1.10), 0.0)
        targets[side] = t
    try:
        m.tcp(targets, free_axis=(0.0, 0.0, 1.0), limits=fast)
    except MotionError:
        m = api.motion()
        for side in ("left", "right"):
            m.joints(side, start_arm[side], limits=fast)
    m.send(depth_next=True, label="carry")


def trip(k, shelf_z):
    got = 0
    try:
        m = api.motion()
        m.waist([-0.05, 0.05, 0.0, w0[3], w0[4]], limits=waist_lim)
        m.hold(10)
        m.send(depth_next=True, label=f"stand_tall{k}")
    except MotionError as error:
        report[f"trip{k}_stand_error"] = str(error)[:120]
    for side, lat in (("right", -0.15), ("left", 0.15)):
        seen = parcels_now(shelf_z)
        report.setdefault(f"trip{k}_seen", []).append([[round(float(v), 3) for v in d.mid] for d in seen])
        if not seen:
            continue
        ref = min(seen, key=lambda d: float(d.mid[1])) if side == "right" else max(seen, key=lambda d: float(d.mid[1]))
        o = odom()
        step = np.array([float(ref.mid[0]) - 0.62, float(ref.mid[1]) - lat])
        step[0] = min(float(step[0]), SHELF_STOP_X - float(o[0]))
        try:
            if float(np.linalg.norm(step)) > 0.03:
                m = api.motion()
                m.base(float(step[0]), float(step[1]), 0.0, speed=0.25)
                m.hold(90)
                m.send(depth_next=True, label=f"to_parcel{k}_{side}")
            else:
                step = np.zeros(2)
        except MotionError as error:
            report[f"trip{k}_{side}_step_error"] = str(error)[:120]
            continue
        exp = np.array([float(ref.mid[0]) - float(step[0]), float(ref.mid[1]) - float(step[1])])
        report.setdefault(f"trip{k}_exp", []).append([round(float(v), 3) for v in exp])
        raw = []
        for q in QUERIES:
            for d in api.perceive(q, max_detections=6):
                raw.append([q[:10], round(float(d.score), 3), int(d.points.shape[0])] + [round(float(v), 3) for v in d.mid] +
                           [round(float(v), 3) for v in d.extent] + [round(float(d.bottom_z), 3)])
        report.setdefault(f"trip{k}_{side}_raw", raw[:16])
        shelf_now = float(api.table_height("head"))
        report[f"trip{k}_{side}_shelf_now"] = round(shelf_now, 4)
        sz = shelf_now if abs(shelf_now - shelf_z) < 0.05 else shelf_z
        pk = api.pick_object(side, QUERIES,
                             find={"size": (0.03, 0.11), "table_z": sz, "z_tolerance": 0.06, "side": side, "away_from_grippers": 0.06,
                                   "region": (lambda c, e=exp: float(np.linalg.norm(np.asarray(c[:2], dtype=float) - e)) < 0.10)},
                             grasp={"depth_into": 0.03, "lift": 0.08, "above": 0.08, "settle_after_close": 10, "corner_yaw": True},
                             refine_wrist=0.10, limits=fast, label=f"pick{k}_{side}")
        report[f"trip{k}_{side}"] = {kk: pk.get(kk) for kk in ("ok", "reason", "attempts")}
        if pk.get("ok"):
            got += 1
            try:
                m = api.motion()
                up = m.grasp_pose(side).copy()
                up[2, 3] += 0.10
                up[1, 3] += -0.10 if side == "right" else 0.10
                m.tcp(side, up, limits=fast)
                m.send(depth_next=True, label=f"aside{k}_{side}")
            except MotionError:
                pass
    carry_pose()
    return got


shelf = api.table_height("head")
report["shelf_z"] = round(shelf, 4)
Ph = api.points("head", stride=3)
Ph = Ph[np.all(np.isfinite(Ph), axis=1)]
Fr = Ph[(np.abs(Ph[:, 1]) < 0.5) & (Ph[:, 0] > 0.2) & (Ph[:, 2] > float(shelf) - 0.30) & (Ph[:, 2] < float(shelf) + 0.02)]
SHELF_STOP_X = (float(np.percentile(Fr[:, 0], 2)) - 0.30) if Fr.shape[0] > 100 else 0.0
report["shelf_stop_x"] = round(SHELF_STOP_X, 3)
got = trip(0, shelf)
report["picked_first_trip"] = got
report["budget"] = api.budget()["phases"]
api.end_episode()
