# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

QUERIES = ["crumpled white paper ball", "crushed red soda can", "white paper trash", "red can"]
FLOOR = 0.0
w0 = [float(v) for v in api.robot_state()["waist"]]
WAIST_FLOOR = [-0.9, 2.2, -0.3, w0[3], w0[4]]
waist_lim = Limits(joint_velocity=1.0, joint_acceleration=3.0)
fast = Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30)
start_arm = {s: np.array(q, dtype=float) for s, q in api.robot_state()["arm"].items()}


def trash_now(exclude_held=False):
    found = []
    for q in QUERIES:
        for d in api.plausible(api.perceive(q, min_score=0.02, max_detections=4), size=(0.03, 0.16), table_z=FLOOR,
                               z_tolerance=0.08, away_from_grippers=0.12 if exclude_held else 0.0):
            far = float(np.linalg.norm(d.mid[:2])) > 1.2
            ok_h = (-0.03 < d.top_z - FLOOR < 0.14) if far else (0.02 < d.top_z - FLOOR < 0.13 and d.bottom_z - FLOOR > -0.06)
            if ok_h and all(float(np.linalg.norm(d.mid[:2] - f.mid[:2])) > 0.10 for f in found):
                found.append(d)
    return found


def scan(exclude_held, label):
    seen = trash_now(exclude_held)
    turns = 0
    while not seen and turns < 3:
        try:
            m = api.motion()
            m.base(0.0, 0.0, math.pi / 2.0, yaw_speed=1.5)
            m.hold(20)
            m.send(depth_next=True, label=f"{label}_turn_{turns + 1}")
        except MotionError as error:
            report[f"{label}_turn_error"] = str(error)[:200]
            break
        turns += 1
        seen = trash_now(exclude_held)
    report[f"{label}_turns"] = turns
    return seen


def drive_beside(c, lateral, label, ahead=0.46):
    phi = math.atan2(float(c[1]), float(c[0]))
    R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    p = np.asarray(c[:2], dtype=float) - R @ np.array([ahead, lateral])
    m = api.motion()
    m.base(float(p[0]), float(p[1]), phi, speed=0.3, yaw_speed=0.8)
    m.hold(30)
    m.send(depth_next=True, label=label)
    report[label] = [round(float(p[0]), 3), round(float(p[1]), 3), round(math.degrees(phi), 1)]


def floor_pick(side, label, expect=None):
    m = api.motion()
    m.waist(WAIST_FLOOR, limits=waist_lim)
    m.send(depth_next=True, label=f"crouch_{label}")
    region = None
    if expect is not None:
        region = (lambda c, e=np.asarray(expect, dtype=float): float(np.linalg.norm(np.asarray(c[:2], dtype=float) - e)) < 0.15)
    pk = api.pick_object(side, QUERIES,
                         find={"size": (0.03, 0.16), "table_z": FLOOR, "z_tolerance": 0.10, "side": side, "away_from_grippers": 0.12,
                               "region": region},
                         grasp={"tip_clearance": 0.012, "contact_width": "footprint", "lift": 0.12, "above": 0.08, "settle_after_close": 12},
                         refine_wrist=0.10, limits=fast, label=f"pick_{label}")
    report[f"pick_{label}"] = {k: pk.get(k) for k in ("ok", "reason", "attempts", "wrist_floor")}
    report[f"pick_{label}_det"] = (pk.get("detection") or {}).get("mid")
    held = bool(pk.get("ok"))
    try:
        m = api.motion()
        m.joints(side, start_arm[side], limits=fast)
        m.waist(w0, limits=waist_lim)
        m.send(depth_next=True, label=f"stand_{label}")
    except MotionError as error:
        report[f"stand_error_{label}"] = str(error)[:160]
    return held


def find_can():
    best = None
    for q in ("trash can", "pedal trash can", "grey trash bin"):
        for d in api.plausible(api.perceive(q, max_detections=4), size=(0.12, 0.6), table_z=FLOOR, z_tolerance=0.15, away_from_grippers=0.15):
            if float(np.linalg.norm(d.mid[:2])) < 0.40 or not (0.15 < float(d.top_z) - FLOOR < 0.55):
                continue
            if best is None or d.score > best.score:
                best = d
        if best is not None:
            return best
    return best


TYPES = {"crumpled white paper ball": "paper", "white paper trash": "paper", "crushed red soda can": "soda", "red can": "soda"}


def views_add(store, label):
    for d in trash_now(False):
        q = str(d.label)
        store.append({"type": TYPES.get(q, q), "score": round(float(d.score), 3), "p": api.base_to_start(d.mid), "view": label})
    cn = find_can()
    return None if cn is None else {"score": round(float(cn.score), 3), "p": api.base_to_start(cn.mid), "view": label}


store = []
cans = []
c0 = views_add(store, "v0")
if c0 is not None:
    cans.append(c0)
for k in range(3):
    try:
        m = api.motion()
        m.base(0.0, 0.0, math.pi / 2.0, yaw_speed=1.5)
        m.hold(20)
        m.send(depth_next=True, label=f"look_{k + 1}")
    except MotionError as error:
        report["look_error"] = str(error)[:200]
        break
    ck = views_add(store, f"v{k + 1}")
    if ck is not None:
        cans.append(ck)
clusters = []
for s in sorted(store, key=lambda e: -e["score"]):
    hit = None
    for c in clusters:
        if float(np.linalg.norm(c["p"][:2] - s["p"][:2])) < 0.18:
            hit = c
            break
    if hit is None:
        clusters.append({"p": s["p"].copy(), "score": s["score"], "type": s["type"], "n": 1})
    else:
        hit["n"] += 1
report["pieces_seen"] = [{"p": [round(float(v), 3) for v in c["p"][:2]], "score": c["score"], "type": c["type"], "n": c["n"]} for c in clusters]
picked = []
for c in sorted(clusters, key=lambda c: -float(c["score"])):
    if all(float(np.linalg.norm(c["p"][:2] - o["p"][:2])) > 0.3 for o in picked):
        picked.append(c)
    if len(picked) == 2:
        break
can_store = max(cans, key=lambda c: c["score"]) if cans else None
report["can_seen"] = None if can_store is None else {"p": [round(float(v), 3) for v in can_store["p"][:2]], "score": can_store["score"]}
first_ok = False
second = None
if picked:
    here = lambda c: float(np.linalg.norm(api.start_to_base(c["p"])[:2]))
    picked.sort(key=here)
    first = picked[0]
    second = picked[1] if len(picked) > 1 else None
    report["first_target"] = {"p": [round(float(v), 3) for v in first["p"][:2]], "score": first["score"], "type": first["type"]}
    try:
        drive_beside(api.start_to_base(first["p"]), -0.18, "to_first")
        api.stage_done()
        first_ok = floor_pick("right", "first")
        if first_ok:
            api.stage_done()
    except MotionError as error:
        report["first_error"] = str(error)[:200]
if first_ok:
    c2 = None
    if second is not None:
        c2 = api.start_to_base(second["p"])[:2]
        report["second_target"] = {"p": [round(float(v), 3) for v in second["p"][:2]], "score": second["score"], "type": second["type"], "now": [round(float(v), 3) for v in c2]}
    else:
        others = scan(True, "scan2")
        report["seen2"] = [d.summary() for d in others]
        if others:
            c2 = np.asarray(min(others, key=lambda d: float(np.linalg.norm(d.mid[:2]))).mid[:2], dtype=float)
    if c2 is not None:
        try:
            dist = float(np.linalg.norm(c2))
            if dist > 1.1:
                phi = math.atan2(float(c2[1]), float(c2[0]))
                go = dist - 0.85
                m = api.motion()
                m.base(go * math.cos(phi), go * math.sin(phi), phi, speed=0.3, yaw_speed=0.8)
                m.hold(25)
                m.send(depth_next=True, label="close_in")
                pred = np.array([0.85, 0.0])
                near = [d for d in trash_now(True) if float(np.linalg.norm(np.asarray(d.mid[:2], dtype=float) - pred)) < 0.40]
                report["close_in_seen"] = [[round(float(v), 3) for v in d.mid] for d in near]
                if near:
                    c2 = np.asarray(min(near, key=lambda d: float(np.linalg.norm(np.asarray(d.mid[:2], dtype=float) - pred))).mid[:2], dtype=float)
                else:
                    c2 = pred
            drive_beside(c2, 0.18, "to_second")
            expect = np.array([0.46, 0.18])
            chk = [d for d in trash_now(True) if float(np.linalg.norm(np.asarray(d.mid[:2], dtype=float) - expect)) < 0.30]
            report["beside_seen"] = [[round(float(v), 3) for v in d.mid] for d in chk]
            if chk:
                best = min(chk, key=lambda d: float(np.linalg.norm(np.asarray(d.mid[:2], dtype=float) - expect)))
                off = np.asarray(best.mid[:2], dtype=float) - expect
                if float(np.linalg.norm(off)) > 0.08:
                    m = api.motion()
                    m.base(float(off[0]), float(off[1]), 0.0, speed=0.15)
                    m.hold(20)
                    m.send(depth_next=True, label="beside_fix")
                    report["beside_fix"] = [round(float(v), 3) for v in off]
            api.stage_done()
            second_ok = floor_pick("left", "second", expect=expect)
            report["second_ok"] = second_ok
            if second_ok:
                api.stage_done()
        except MotionError as error:
            report["second_error"] = str(error)[:200]
    can = None
    if can_store is not None:
        cb = api.start_to_base(can_store["p"])
        can = {"mid": cb}
    else:
        found = find_can()
        k = 0
        while found is None and k < 3:
            try:
                m = api.motion()
                m.base(0.0, 0.0, math.pi / 2.0, yaw_speed=1.5)
                m.hold(20)
                m.send(depth_next=True, label=f"can_turn_{k + 1}")
            except MotionError:
                break
            k += 1
            found = find_can()
        if found is not None:
            can = {"mid": found.mid}
    report["can"] = None if can is None else [round(float(v), 3) for v in can["mid"][:2]]
    if can is not None:
        try:
            c = can["mid"]
            phi = math.atan2(float(c[1]), float(c[0]))
            dist = math.hypot(float(c[0]), float(c[1]))
            go = max(0.0, dist - 0.60)
            m = api.motion()
            m.base(go * math.cos(phi), go * math.sin(phi), phi, speed=0.3, yaw_speed=0.8)
            m.hold(30)
            m.send(depth_next=True, label="to_can")
            api.stage_done()
        except MotionError as error:
            report["can_error"] = str(error)[:200]
report["budget"] = api.budget()["phases"]
api.end_episode()
