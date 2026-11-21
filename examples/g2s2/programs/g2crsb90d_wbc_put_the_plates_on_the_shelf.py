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


def rim_pick(side, plate, angles=(0.0, 0.5, -0.5, 1.0, -1.0)):
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
    for ang in angles:
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
                report.setdefault(f"pinch_{side}", []).append([round(ang, 2), beta, round(float(G[2, 3]), 4), round(float(G[0, 3]), 4),
                                                               round(float(G[1, 3]), 4), round(float(edge), 4), [round(float(v), 4) for v in c[:2]]])
                return {"sent": True, "angle": ang, "tilt": beta}
            except MotionError as error:
                errors.append(f"tilt {beta}: {str(error)[:100]}")
    return {"sent": False, "errors": errors[-2:]}


def pick_check(side, plate, other):
    opening = float(api.robot_state()["opening"][side])
    if not api.has_depth():
        api.hold(1, depth_next=True, label="observe")
    seen = plates_now()
    at_spot = [d for d in seen if float(np.linalg.norm(d.mid[:2] - plate.mid[:2])) < 0.10]
    other_only = other is not None and len(seen) == 1 and float(np.linalg.norm(seen[0].mid[:2] - other.mid[:2])) < 0.08
    verdict = "held" if (opening > 0.010 or (other_only and not at_spot)) else ("missed" if (opening < 0.006 and at_spot) else "unknown")
    report.setdefault(f"check_{side}", []).append({"opening": round(opening, 4), "seen": len(seen), "at_spot": len(at_spot),
                                                    "other_only": bool(other_only), "verdict": verdict})
    return verdict, (at_spot[0] if at_spot else None)


def pick_with_retry(side, plate, other, retries=2):
    res = rim_pick(side, plate)
    if not res.get("sent"):
        return res, "unsent"
    verdict, again = pick_check(side, plate, other)
    for angles in ((0.5, -0.5, 1.0, -1.0, 0.0), (-0.5, 0.5, -1.0, 1.0, 0.0))[:retries]:
        if not (verdict == "missed" and again is not None and api.rows_left() > 520):
            break
        moved = float(np.linalg.norm(again.mid[:2] - plate.mid[:2]))
        target = again if moved > 0.03 else plate
        report.setdefault(f"retry_{side}", []).append([round(float(v), 4) for v in target.mid[:2]] + [round(moved, 3)])
        res = rim_pick(side, target, angles=angles)
        if not res.get("sent"):
            break
        verdict, again = pick_check(side, target, other)
    return res, verdict


plates = plates_now()
report["plates"] = [p.summary() for p in plates]
if len(plates) >= 2:
    two = sorted(plates[:2], key=lambda p: -float(p.top_z))
    tall, flat = two[0], two[1]
    report["heights_mm"] = [round(1000.0 * (float(p.top_z) - table), 1) for p in two]
    if float(tall.top_z) - float(flat.top_z) < 0.005:
        tall, flat = max(two, key=lambda p: float(p.mid[1])), min(two, key=lambda p: float(p.mid[1]))
        report["assign"] = "by_side"
    else:
        report["assign"] = "crossed" if float(tall.mid[1]) < float(flat.mid[1]) else "natural"
    res_l, ver_l = pick_with_retry("left", tall, flat)
    report["left"] = dict(res_l, verdict=ver_l)
    api.stage_done()
    if report["assign"] == "crossed" and ver_l in ("held", "unknown"):
        for xy in ((0.50, 0.42), (0.55, 0.34), (0.45, 0.30)):
            try:
                t = api.grasp_pose("left").copy()
                up = t.copy()
                up[2, 3] = table + 0.22
                out = t.copy()
                out[0, 3], out[1, 3], out[2, 3] = xy[0], xy[1], table + 0.25
                m = api.motion()
                m.tcp("left", up, linear=True, limits=Limits(linear_velocity=0.15))
                m.tcp("left", out, limits=Limits(joint_velocity=1.0, joint_acceleration=3.0))
                m.send(depth_next=True, label="left_plate_aside")
                report["aside"] = list(xy)
                break
            except MotionError as error:
                report.setdefault("aside_errors", []).append(str(error)[:80])
    flat_now = flat
    if report["assign"] == "crossed":
        shift = min(0.30, max(0.0, float(flat.mid[1]) - 0.10))
        if shift > 0.03:
            try:
                m = api.motion()
                m.base(0.0, shift, 0.0, speed=0.20)
                m.hold(25)
                m.send(depth_next=True, label="strafe_left")
                report["strafe"] = round(shift, 3)
                expect = np.array([float(flat.mid[0]), float(flat.mid[1]) - shift])
                seen = [d for d in plates_now() if float(np.linalg.norm(d.mid[:2] - expect)) < 0.12]
                if seen:
                    flat_now = min(seen, key=lambda d: float(np.linalg.norm(d.mid[:2] - expect)))
                    report["flat_after_strafe"] = [round(float(v), 4) for v in flat_now.mid[:2]]
                else:
                    flat.points[:, 1] -= shift
                    flat_now = flat
                    report["flat_after_strafe"] = ["shifted"] + [round(float(v), 4) for v in flat_now.mid[:2]]
            except MotionError as error:
                report["strafe_error"] = str(error)[:120]
    res_r, ver_r = pick_with_retry("right", flat_now, None)
    report["right"] = dict(res_r, verdict=ver_r)
    api.stage_done()
    report["left_after"] = round(float(api.robot_state()["opening"]["left"]), 4)
elif plates:
    res_l, ver_l = pick_with_retry("left", plates[0], None)
    report["left"] = dict(res_l, verdict=ver_l)


report["budget"] = api.budget()["phases"]
api.end_episode()
