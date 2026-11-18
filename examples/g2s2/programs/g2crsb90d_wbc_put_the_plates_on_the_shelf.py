# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

table = api.table_height("head")
report["table_z"] = round(table, 4)


def plates_now():
    found = []
    for q in ("plate", "white plate", "saucer"):
        for d in api.plausible(api.perceive(q, max_detections=5), size=(0.08, 0.30), table_z=table, z_tolerance=0.05):
            if all(float(np.linalg.norm(d.mid[:2] - f.mid[:2])) > 0.06 for f in found):
                found.append(d)
        if len(found) >= 2:
            break
    return found


def wrist_plate(side, plate, toward):
    c = plate.mid
    radius = 0.5 * float(max(plate.extent[0], plate.extent[1]))
    p = np.array([float(c[0] + toward[0] * (radius - 0.03)), float(c[1] + toward[1] * (radius - 0.03))])
    sent = False
    for h in (0.16, 0.13, 0.10):
        for tilt in (0.0, 0.35, 0.6):
            try:
                m = api.motion()
                m.tcp(side, top_down((float(p[0]), float(p[1]), table + h), 0.0, tilt=tilt, tilt_toward=math.atan2(float(toward[1]), float(toward[0]))),
                      gripper="open", free_axis=(0.0, 0.0, 1.0))
                m.send(depth_next=True, label=f"hover_plate_{side}")
                report[f"hover_{side}"] = [h, tilt]
                sent = True
                break
            except MotionError as error:
                report.setdefault(f"hover_errors_{side}", []).append(str(error)[:60])
        if sent:
            break
    cam = f"hand_{side}"
    if not sent or not api.has_depth(cam):
        return None
    tw = api.table_height(cam)
    tw = float(tw) if (math.isfinite(tw) and abs(float(tw) - table) < 0.03) else table
    best = None
    for q in ("plate", "white plate", "saucer"):
        for d in api.perceive(q, cam, max_detections=4):
            if d.points.shape[0] < 50 or float(np.linalg.norm(d.mid[:2] - c[:2])) > 0.12 or abs(float(d.bottom_z) - table) > 0.05:
                continue
            if best is None or d.score > best.score:
                best = d
        if best is not None:
            break
    if best is None:
        return None
    P = best.points[best.points[:, 2] > tw + 0.003]
    report[f"wrist_plate_{side}"] = {"head_mid": [round(float(v), 3) for v in c[:2]], "wrist_mid": [round(float(v), 3) for v in best.mid[:2]],
                                     "points": int(best.points.shape[0]), "above_table": int(P.shape[0]),
                                     "table_head": round(table, 4), "table_wrist": round(tw, 4)}
    return P, tw


def rim_pick(side, plate):
    c = plate.mid
    radius = min(max(0.5 * float(max(plate.extent[0], plate.extent[1])), 0.05), 0.14)
    sh0 = api.shoulder(side)
    t0 = np.array([float(sh0[0] - c[0]), float(sh0[1] - c[1])])
    t0 = t0 / max(1e-6, float(np.linalg.norm(t0)))
    seen = wrist_plate(side, plate, t0)
    wrist_pts, tz = (seen[0], seen[1]) if seen is not None else (None, table)
    sh = api.shoulder(side)
    toward = np.array([float(sh[0] - c[0]), float(sh[1] - c[1])])
    toward = toward / max(1e-6, float(np.linalg.norm(toward)))
    errors = []
    for ang in (0.0, 0.5, -0.5, 1.0, -1.0):
        d = np.array([math.cos(ang) * toward[0] - math.sin(ang) * toward[1], math.sin(ang) * toward[0] + math.cos(ang) * toward[1]])
        edge = radius
        if wrist_pts is not None:
            near = wrist_pts[np.linalg.norm(wrist_pts[:, :2] - np.asarray(c[:2], dtype=float), axis=1) < radius + 0.05]
            proj = (near[:, :2] - np.asarray(c[:2], dtype=float)) @ d
            tang = (near[:, :2] - np.asarray(c[:2], dtype=float)) @ np.array([-d[1], d[0]])
            if proj.shape[0] > 30:
                e = float(np.percentile(proj, 98))
                r_w = 0.5 * float(np.percentile(tang, 98) - np.percentile(tang, 2))
                e_far = float(np.percentile(proj, 2)) + 2.0 * r_w
                if 0.05 < r_w < 0.12 and e_far > e:
                    e = min(e_far, e + 0.02)
                if abs(e - radius) < 0.06:
                    edge = e
                report.setdefault(f"rim_edge_{side}", []).append([round(ang, 2), round(e, 3), round(radius, 3), int(proj.shape[0]), round(tz, 4)])
        radial = math.atan2(float(d[1]), float(d[0]))
        done = False
        for beta in (0.44, 0.30, 0.60, 0.0):
            centre = np.array([float(c[0] + d[0] * (edge - 0.004)), float(c[1] + d[1] * (edge - 0.004)), tz + 0.010])
            G = top_down((float(centre[0]), float(centre[1]), float(centre[2])), radial, tilt=beta, tilt_toward=radial)
            tip_z = float(centre[2]) + 0.0096 * float(G[2, 2])
            if tip_z < tz + 0.002:
                G[2, 3] += (tz + 0.002 - tip_z)
            pre = G.copy()
            pre[:3, 3] = G[:3, 3] - G[:3, 2] * 0.08
            up = G.copy()
            up[2, 3] += 0.10
            try:
                m = api.motion()
                m.tcp(side, pre, gripper="open", limits=Limits(joint_velocity=2.0, joint_acceleration=8.0))
                m.tcp(side, G, linear=True, limits=Limits(linear_velocity=0.08))
                m.hold(6)
                m.gripper(side, "close", rows=25)
                m.hold(18)
                m.tcp(side, up, linear=True, limits=Limits(linear_velocity=0.15))
                m.send(depth_next=True, label=f"pick_plate_{side}")
                report.setdefault(f"pinch_{side}", []).append([round(ang, 2), beta, round(float(G[2, 3]), 4)])
                return {"ok": bool(api.holding(side)), "angle": ang, "tilt": beta}
            except MotionError as error:
                errors.append(f"tilt {beta}: {str(error)[:100]}")
    return {"ok": False, "errors": errors[-2:]}


plates = plates_now()
report["plates"] = [p.summary() for p in plates]
if plates:
    left_plate = max(plates, key=lambda p: float(p.mid[1]))
    res_l = rim_pick("left", left_plate)
    report["left"] = res_l
    if res_l.get("ok"):
        api.stage_done()
        if not api.has_depth():
            api.hold(1, depth_next=True, label="observe")
        others = [p for p in plates_now() if float(np.linalg.norm(p.mid[:2] - left_plate.mid[:2])) > 0.08]
        if others:
            right_plate = min(others, key=lambda p: float(p.mid[1]))
            res_r = rim_pick("right", right_plate)
            report["right"] = res_r
            if res_r.get("ok"):
                api.stage_done()


def drive_start_frame(goal, label):
    odom = np.array(api.robot_state()["odom"], dtype=float)
    d = np.array(goal, dtype=float) - odom[:2]
    c, s_ = math.cos(-odom[2]), math.sin(-odom[2])
    dx, dy = c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]
    m = api.motion()
    m.base(float(dx), float(dy), float(-odom[2]), speed=0.25, yaw_speed=0.4)
    m.hold(30)
    m.send(depth_next=True, label=label)
    return [round(float(v), 3) for v in api.robot_state()["odom"]]


if report.get("left", {}).get("ok") and report.get("right", {}).get("ok"):
    legs = []
    try:
        legs.append(drive_start_frame((0.0, -1.37), "leg_right"))
        legs.append(drive_start_frame((1.91, -1.37), "leg_forward"))
        legs.append(drive_start_frame((1.91, -0.40), "leg_left"))
        api.stage_done()
    except MotionError as error:
        report["drive_error"] = str(error)[:200]
    report["legs_odom"] = legs
    if len(legs) == 3:
        released = False
        for reach in (0.66, 0.58, 0.72, 0.52):
            try:
                m = api.motion()
                targets = {}
                for side, sy in (("left", 0.16), ("right", -0.16)):
                    t = api.grasp_pose(side).copy()
                    t[0, 3], t[1, 3], t[2, 3] = reach, sy, 0.95
                    targets[side] = t
                m.tcp(targets)
                m.gripper("left", "open", rows=20).gripper("right", "open", rows=20)
                m.tcp({sd: offset(t, dz=0.10) for sd, t in targets.items()})
                m.send(depth_next=True, label="release_over_rack")
                report["released_reach"] = reach
                released = True
                api.stage_done()
                break
            except MotionError as error:
                report.setdefault("release_errors", []).append(str(error)[:120])
report["budget"] = api.budget()["phases"]
api.end_episode()
