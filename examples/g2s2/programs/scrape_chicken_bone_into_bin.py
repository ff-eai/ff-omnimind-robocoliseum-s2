# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
active = info["active_arms"]
start_arm = {s_: np.array(q, dtype=float) for s_, q in api.robot_state()["arm"].items()}
table = api.table_height("head")
report["embodiment"] = info["embodiment"]
BONE_Q = ["chicken bone", "bone", "white bone"]
APPLE_Q = ["apple core", "apple", "eaten apple"]
whole = api.find_object(["green scraper", "green squeegee", "scraper"], size=(0.06, 0.30), table_z=table, z_tolerance=0.06)
handle = api.find_object(["green squeegee handle", "scraper handle"], size=(0.03, 0.20), table_z=table, z_tolerance=0.06)
bone0 = api.find_object(BONE_Q, size=(0.02, 0.12), table_z=table, z_tolerance=0.06)
apple0 = api.find_object(APPLE_Q, size=(0.03, 0.12), table_z=table, z_tolerance=0.08)
report["whole"] = None if whole is None else whole.summary()
report["handle"] = None if handle is None else handle.summary()
report["bone0"] = None if bone0 is None else bone0.summary()
report["apple0"] = None if apple0 is None else apple0.summary()


def table_near_edge():
    P = api.points("head", stride=3)
    P = P[np.all(np.isfinite(P), axis=1)]
    T = P[(np.abs(P[:, 2] - float(table)) < 0.015) & (np.abs(P[:, 1]) < 0.35) & (P[:, 0] > 0.05)]
    return float(np.percentile(T[:, 0], 2)) if T.shape[0] >= 200 else None


edge_x = table_near_edge()
report["table_edge_x"] = None if edge_x is None else round(edge_x, 3)
LOOKS = ((0.26, 0.0, 0.0), (0.20, 0.5, math.pi), (0.14, 0.0, 0.0), (0.12, 0.5, None), (0.20, 0.5, 0.0))
canlook = (edge_x - 0.06, (0.20, -0.20)) if edge_x is not None else None
if len(active) == 2:
    LOOKS = LOOKS[:1] + ((0.32, 0.0, 0.0),) + LOOKS[1:]
two_arm = len(active) == 2


def own_first(arm):
    return (0.20, -0.20) if arm == "left" else (-0.20, 0.20)


SIDES2 = None
looked = []
found = []


def look_pose(py, dz, tilt, toward):
    tw = toward if toward is not None else (-math.pi / 2.0 if py > 0 else math.pi / 2.0)
    return top_down((float(canlook[0]), float(py), table + dz), 0.0, tilt=tilt, tilt_toward=tw)


def white_patch(side, near):
    cam = f"hand_{side}"
    if not api.has_depth(cam):
        return None
    view = api.camera(cam)
    if view.depth_m is None:
        return None
    rgb = view.rgb.astype(np.float64) / 255.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    white = (mx > 0.55) & ((mx - mn) / np.maximum(mx, 1e-6) < 0.18)
    if white.shape != view.depth_m.shape:
        hh, ww = view.depth_m.shape
        white = white[(np.arange(hh) * white.shape[0] // hh)[:, None], (np.arange(ww) * white.shape[1] // ww)[None, :]]
    P = view.points_from_mask(white, stride=2)
    P = P[np.all(np.isfinite(P), axis=1)]
    P = P[(P[:, 2] < table - 0.03) & (P[:, 2] > table - 0.22) & (np.linalg.norm(P[:, :2] - np.asarray(near), axis=1) < 0.16)]
    report.setdefault("white_points", []).append(int(P.shape[0]))
    if P.shape[0] < 120:
        return None
    ext = np.percentile(P[:, :2], 95, axis=0) - np.percentile(P[:, :2], 5, axis=0)
    report.setdefault("can", []).append({"xy": [round(float(v), 4) for v in np.median(P[:, :2], axis=0)], "extent": [round(float(v), 3) for v in ext]})
    if float(ext[0]) < 0.06 or float(ext[1]) < 0.06:
        return None
    return np.median(P[:, :2], axis=0), int(P.shape[0])


def note_look(side, py, dz, tilt):
    looked.append(float(py))
    report.setdefault("look", []).append([round(float(canlook[0]), 3), round(float(py), 3), dz, tilt, side])
    got = white_patch(side, (float(canlook[0]), float(py)))
    if got is not None:
        found.append((got[1], got[0], float(py)))


def stem_grasp(det, frac):
    P = det.points
    base = float(np.percentile(P[:, 2], 10))
    S = P[P[:, 2] > base + 0.015]
    if S.shape[0] < 40:
        return None
    sc = np.median(S[:, :2], axis=0)
    vals_s, vecs_s = np.linalg.eigh(np.cov((S[:, :2] - sc).T))
    ax = vecs_s[:, int(np.argmax(vals_s))]
    pr = (S[:, :2] - sc) @ ax
    lo_s, hi_s = float(np.percentile(pr, 3)), float(np.percentile(pr, 97))
    slope = float(np.polyfit(pr, S[:, 2], 1)[0])
    perp = np.array([-float(ax[1]), float(ax[0])])
    X = P[np.abs((P[:, :2] - sc) @ perp) > 0.03]
    if X.shape[0] >= 15:
        jhi = float(np.median((X[:, :2] - sc) @ ax)) > 0.5 * (lo_s + hi_s)
    else:
        jhi = slope > 0.0
    sj, st = (hi_s, lo_s) if jhi else (lo_s, hi_s)
    sgr = sj + frac * (st - sj)
    g = sc + ax * sgr
    nr = S[np.abs(pr - sgr) < 0.012]
    tp = float(np.percentile(nr[:, 2], 80)) if nr.shape[0] >= 8 else float(np.percentile(S[:, 2], 50))
    return {"xy": g, "top": tp, "yaw": math.atan2(float(ax[1]), float(ax[0])), "len": hi_s - lo_s, "n": int(S.shape[0]),
            "n_cross": int(X.shape[0]), "slope": slope, "base": base}


done = False
if handle is not None:
    H = handle.points
    hc = np.median(H, axis=0)
    xy = H[:, :2] - hc[:2]
    vals, vecs = np.linalg.eigh(np.cov(xy.T))
    axis = vecs[:, int(np.argmax(vals))]
    proj = xy @ axis
    lo, hi = float(np.percentile(proj, 3)), float(np.percentile(proj, 97))
    junction_hi = True
    if whole is not None:
        wc = whole.mid[:2] - hc[:2]
        junction_hi = float(wc @ axis) > 0.5 * (lo + hi)
    s_j, s_t = (hi, lo) if junction_hi else (lo, hi)
    s_g = s_j + 0.4 * (s_t - s_j)
    g_xy = hc[:2] + axis * s_g
    near = H[np.abs(proj - s_g) < 0.015]
    top = float(np.percentile(near[:, 2], 90)) if near.shape[0] > 10 else handle.top_z
    stem_yaw = math.atan2(float(axis[1]), float(axis[0]))
    depth_in = 0.05
    sg = stem_grasp(handle, 0.55)
    if sg is not None and 0.05 <= sg["len"] <= 0.25:
        report["stem"] = {"top": round(sg["top"], 4), "len": round(sg["len"], 3), "n": sg["n"], "n_cross": sg["n_cross"],
                          "slope": round(sg["slope"], 3), "base": round(sg["base"], 4)}
        if info["embodiment"] in ("g2_omnipicker", "g1_omnipicker"):
            g_xy, top, stem_yaw, depth_in = sg["xy"], sg["top"], sg["yaw"], 0.010
        else:
            g_xy, top, stem_yaw = sg["xy"], sg["top"], sg["yaw"]
    report["grasp_xy"] = [round(float(v), 4) for v in g_xy]
    report["stem_yaw_deg"] = round(math.degrees(stem_yaw), 1)
    first = api.choose_arm(g_xy)
    floor = table
    hover_t = top_down((float(g_xy[0]), float(g_xy[1]), top + 0.12), stem_yaw + math.pi / 2.0)
    moves = [({first: hover_t}, "hover_stem", None)]
    if two_arm and canlook is not None:
        free0 = [sd for sd in active if sd != first][0]
        SIDES2 = own_first(free0)
        pairs = [({first: hover_t, free0: look_pose(SIDES2[0], dz, tilt, tw)}, "hover_stem_look", (free0, dz, tilt)) for dz, tilt, tw in LOOKS]
        alone = [({free0: look_pose(SIDES2[0], dz, tilt, tw)}, "look_can", (free0, dz, tilt)) for dz, tilt, tw in LOOKS]
        moves = pairs + moves + alone
    for goals, lab, lk in moves:
        try:
            hv = api.motion()
            if lk is None:
                hv.tcp(first, goals[first], gripper="open", free_axis=(0.0, 0.0, 1.0))
            else:
                if first in goals:
                    hv.gripper(first, "open", rows=6)
                hv.tcp(goals, free_axis=(0.0, 0.0, 1.0), limits=Limits(joint_velocity=2.4, joint_acceleration=9.0))
            hv.send(depth_next=True, label=lab)
        except MotionError as error:
            report.setdefault("hover_errors", []).append(f"{lab}: {str(error)[:100]}")
            continue
        if first in goals and api.has_depth(f"hand_{first}"):
            wf = api.table_height(f"hand_{first}")
            if math.isfinite(wf) and abs(wf - float(top)) < 0.04:
                floor = wf
                report["wrist_floor"] = round(wf, 4)
        if lk is not None:
            note_look(lk[0], SIDES2[0], lk[1], lk[2])
        break
    omni = info["embodiment"] == "g1_omnipicker"
    fc, settle = (0.015, 35) if omni else (0.003, 15)
    if depth_in < 0.05:
        settle = 35
    for side in [first] + [sd for sd in active if sd != first]:
        try:
            plan = api.top_grasp(side, center=(float(g_xy[0]), float(g_xy[1]), top), top_z=top, bottom_z=floor,
                                 yaw=stem_yaw, depth_into=depth_in, floor_clearance=fc, lift=0.10, above=0.07,
                                 settle_after_close=settle)
        except MotionError as error:
            report.setdefault("plan_errors", []).append(f"{side}: {str(error)[:160]}")
            continue
        plan.send(depth_next=True, label="pick_scraper")
        report["side"] = side
        report["holding"] = api.holding(side)
        done = True
        break
    if done:
        api.stage_done()


def find_can(side, items):
    if canlook is None:
        return None
    px = canlook[0]
    order = SIDES2 if (two_arm and SIDES2 is not None) else own_first(side)
    for py in order:
        if two_arm and (any(abs(py - lk) < 1e-6 for lk in looked) or any(f[0] >= 1200 for f in found)):
            continue
        if api.rows_left() < 560:
            break
        for dz, tilt, toward in LOOKS:
            try:
                m = api.motion()
                if not two_arm:
                    m.gripper(side, "open", rows=10)
                m.tcp(side, look_pose(py, dz, tilt, toward), free_axis=(0.0, 0.0, 1.0),
                      limits=Limits(joint_velocity=2.4, joint_acceleration=9.0))
                m.send(depth_next=True, label="look_can")
            except MotionError as error:
                report.setdefault("look_errors", []).append(str(error)[:60])
                continue
            note_look(side, py, dz, tilt)
            break
    if found:
        n, c, py = max(found, key=lambda t: t[0])
        report["can_side"] = [round(float(py), 3), n]
        return c
    report["can_not_found"] = True
    return None


def drop_in(side, can_xy, label, look_side=None):
    lim = Limits(joint_velocity=3.2, joint_acceleration=12.0) if two_arm else Limits(joint_velocity=2.4, joint_acceleration=9.0)
    drops = ((0.09, 0.0, 0.0), (0.13, 0.0, 0.0), (0.06, 0.0, 0.0), (0.09, 0.5, -math.pi / 2.0),
             (0.09, 0.5, math.pi / 2.0), (0.09, 0.5, math.pi))
    if two_arm:
        drops = ((0.09, 0.0, 0.0), (0.13, 0.0, 0.0), (0.06, 0.0, 0.0), (0.09, 0.3, 0.0), (0.13, 0.3, 0.0), (0.06, 0.3, 0.0),
                 (0.30, 0.0, 0.0), (0.36, 0.0, 0.0)) + drops[3:]
    for dz, tilt, toward in drops:
        try:
            m = api.motion()
            if look_side is not None and look_side != side and report.get("parked") is None:
                m.joints(look_side, start_arm[look_side], limits=lim)
            m.tcp(side, top_down((float(can_xy[0]), float(can_xy[1]), table + dz), 0.0, tilt=tilt, tilt_toward=toward), free_axis=(0.0, 0.0, 1.0),
                  limits=lim)
            m.gripper(side, "open", rows=10)
            m.hold(8)
            m.send(depth_next=True, label=label)
            if look_side is not None and look_side != side:
                report["parked"] = look_side
            return [dz, tilt]
        except MotionError as error:
            report.setdefault(f"{label}_errors", []).append(str(error)[:80])
    return None


if done:
    holder = report["side"]
    free = [sd for sd in active if sd != holder][0] if two_arm else holder
    report["free_arm"] = free
    items = [d for d in (bone0, apple0, whole) if d is not None]
    can_xy = find_can(free, items)
    if can_xy is not None:
        ANY_Q = BONE_Q + APPLE_Q + ["food scraps", "garbage"]
        picked_at = []
        for k_pass in range(3):
            if api.rows_left() < 260:
                report.setdefault("skipped", []).append(k_pass)
                break
            find = {"size": (0.02, 0.12), "table_z": table, "z_tolerance": 0.08, "exclude": [list(p) for p in picked_at], "exclude_radius": 0.05}
            det = api.find_object(ANY_Q, **find)
            if det is None:
                report[f"pick_{k_pass}"] = {"ok": False, "reason": "not found"}
                break
            if two_arm:
                arms = [free, holder]
                find = dict(find, side=None)
            else:
                arms = [api.choose_arm(det.mid)]
            for arm in arms:
                pk = api.pick_object(arm, ANY_Q, retries=1 if api.rows_left() > 650 else 0, find=find,
                                     grasp={"depth_into": 0.025, "floor_clearance": 0.006, "lift": 0.08},
                                     refine_wrist=0.10, limits=Limits(joint_velocity=2.4, joint_acceleration=9.0))
                if pk.get("attempts") or not str(pk.get("reason", "")).startswith("plan"):
                    break
            report[f"pick_{k_pass}"] = {k: pk.get(k) for k in ("ok", "reason", "attempts")}
            report[f"pick_{k_pass}"]["at"] = [round(float(v), 3) for v in det.mid[:2]]
            report[f"pick_{k_pass}"]["arm"] = arm
            at = det.mid[:2]
            if two_arm and isinstance(pk.get("detection"), dict) and pk["detection"].get("mid") is not None:
                at = pk["detection"]["mid"][:2]
            picked_at.append((float(at[0]), float(at[1])))
            if pk.get("attempts"):
                report[f"drop_{k_pass}"] = drop_in(arm, can_xy, f"drop_{k_pass}", look_side=free if two_arm else report["side"])
report["budget"] = api.budget()["phases"]
api.end_episode()
