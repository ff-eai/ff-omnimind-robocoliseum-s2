# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
report["segments"] = info["segments"]
side = "left"
KIN = api.episode.kin
HQ_OPEN = np.array([0.4302, -0.1386, 1.5622, 0.5377, 1.0963, 0.3699, -0.3505, 0.5439, 0.7127, 0.2147, 1.2075, 1.1638, 0.8932, 0.3699, 1.1831, 1.5633, 0.7254, 0.3699, 1.5128, 1.5734])
HQ_CLOSE = np.array([0.5134, -0.1386, 1.5622, 0.0285, 1.2553, 0.3699, -0.4776, 1.0738, 0.7706, 0.1644, 1.1604, 0.5958, 0.8932, 0.3699, 1.1831, 1.5633, 0.7254, 0.3699, 1.5128, 1.5734])
Q_ABOVE = np.array([0.3063, -0.9575, 0.0980, -0.9650, 0.9975, -0.9112, 0.9792])
Q_NEAR = np.array([0.4262, -1.1216, 0.0838, -0.6229, 1.2744, -0.8959, 1.2921])
Q_PRESS = np.array([0.4482, -1.1789, 0.0974, -0.5023, 1.3456, -0.8881, 1.3980])
Q_TWIST = np.array([
    [0.4226, -1.0954, 0.2206, -0.6933, 1.0389, -0.7740, 1.1931],
    [0.2729, -1.0973, 0.6825, -0.8404, 0.6136, -0.5470, 1.2055],
    [0.2091, -1.0988, 0.8477, -0.9632, 0.4187, -0.4022, 1.1955],
    [0.1717, -1.0989, 0.9338, -1.0683, 0.2849, -0.2792, 1.1856],
    [0.1480, -1.0993, 0.9844, -1.1620, 0.1756, -0.1643, 1.1794],
    [0.1360, -1.0993, 1.0093, -1.2445, 0.0828, -0.0559, 1.1770],
    [0.1331, -1.0995, 1.0160, -1.3176, -0.0007, 0.0486, 1.1785],
    [0.1387, -1.0996, 1.0072, -1.3819, -0.0776, 0.1506, 1.1838],
    [0.1525, -1.0995, 0.9844, -1.4377, -0.1495, 0.2506, 1.1923],
    [0.1747, -1.0992, 0.9478, -1.4849, -0.2163, 0.3492, 1.2039],
    [0.2052, -1.0988, 0.8978, -1.5237, -0.2778, 0.4468, 1.2174],
    [0.2453, -1.0979, 0.8328, -1.5538, -0.3313, 0.5445, 1.2313],
    [0.2956, -1.0976, 0.7531, -1.5752, -0.3754, 0.6425, 1.2450],
    [0.3599, -1.0967, 0.6522, -1.5878, -0.3993, 0.7425, 1.2510],
    [0.4464, -1.0952, 0.5189, -1.5913, -0.3821, 0.8485, 1.2344],
    [0.5758, -1.0933, 0.3226, -1.5860, -0.2581, 0.9626, 1.1448],
    [0.5104, -1.1684, 0.4324, -1.5717, -0.6204, 0.9823, 1.4267],
    [0.5259, -1.2036, 0.4094, -1.5531, -0.7211, 0.9823, 1.5059]])
TOOL_G = np.array([-0.005, 0.0, 0.015])
H0 = -math.pi / 2
ZOFF = 0.01905
DEPTH, EXTRA = 0.003, 0.003
MAX_RMS = 0.006
RELEASE_SHAPE = None


def grasp_rot(h):
    return np.column_stack([[-math.cos(h), -math.sin(h), 0.0], [-math.sin(h), math.cos(h), 0.0], [0.0, 0.0, -1.0]])


def grasp_at(p, h):
    T = np.eye(4)
    T[:3, :3] = grasp_rot(h)
    T[:3, 3] = np.asarray(p, dtype=float) - T[:3, :3] @ TOOL_G
    return T


def ik(T, seed):
    r = KIN.solve(side, T, seed, api.episode.positions(), frame="grasp", restarts=0, max_iterations=600,
                  position_tolerance_m=0.0025, rotation_tolerance_rad=math.radians(8.0), orientation_weight=0.05)
    return np.asarray(r.joints, dtype=float) if r.ok else None


def locate():
    for text in ("rubik's cube", "small colorful cube"):
        f = api.locate_rgb(text, cameras=("head", "hand_right"), max_rms_m=MAX_RMS)
        if f is not None:
            return f
    for text in ("rubik's cube", "small colorful cube"):
        f = api.locate_rgb(text)
        if f is not None:
            return f
    return None


def short(f):
    return None if f is None else {"point": [round(float(v), 4) for v in f["point"]], "rms_m": round(f["rms_m"], 4), "cameras": f["cameras"]}


Q_HOME = np.array(api.robot_state()["arm"][side], dtype=float)
found = locate()
report["cube"] = short(found)
if found is not None:
    c = np.asarray(found["point"], dtype=float)
    P0 = np.array([c[0], c[1], c[2] + ZOFF])
    try:
        if found["rms_m"] > MAX_RMS or "hand_left" in found["cameras"]:
            raise MotionError(f"localisation not precise enough for the turn (rms {found['rms_m']:.4f}, {found['cameras']})")
        q_above = ik(grasp_at(P0 + [0, 0, 0.08], H0), Q_ABOVE)
        q_near = ik(grasp_at(P0 + [0, 0, 0.012], H0), Q_NEAR if q_above is None else q_above)
        q_press = ik(grasp_at(P0 - [0, 0, DEPTH], H0), Q_PRESS if q_near is None else q_near)
        if q_above is None or q_near is None or q_press is None:
            raise MotionError("no approach IK")
        qs, q = [], q_press
        n = len(Q_TWIST)
        for i in range(1, n + 1):
            T = grasp_at(P0 - [0, 0, DEPTH + EXTRA * i / n], H0 + (math.pi / 2) * i / n)
            qi = ik(T, Q_TWIST[i - 1] + (q - (Q_TWIST[i - 2] if i > 1 else Q_PRESS)))
            if qi is None:
                qi = ik(T, q)
            if qi is None:
                raise MotionError(f"no twist IK at step {i}")
            if float(np.max(np.abs(qi - q))) > 0.8:
                raise MotionError(f"twist joint jump {float(np.max(np.abs(qi - q))):.2f} at step {i}")
            qs.append(qi)
            q = qi
        h_end = H0 + math.pi / 2
        q_up2 = ik(grasp_at(P0 + [0, 0, 0.02], h_end), q)
        q_clear = ik(grasp_at(P0 + [0, 0.10, 0.14], h_end), q_up2 if q_up2 is not None else q)
        m = api.motion()
        m.hand(side, HQ_OPEN, rows=8)
        m.joints(side, q_above, rows=15)
        m.joints(side, q_near, rows=4)
        m.hold(2)
        m.joints(side, q_press, rows=2)
        m.hand(side, HQ_CLOSE, rows=2)
        m.hold(1)
        for qi in qs:
            m.joints(side, qi, rows=1)
        m.hand(side, HQ_OPEN, rows=3)
        if q_up2 is not None:
            m.joints(side, q_up2, rows=4, gripper=RELEASE_SHAPE)
        if q_clear is not None:
            m.joints(side, q_clear, rows=10)
        m.joints(side, Q_HOME, rows=15)
        report["turn_rows"] = m.rows
        report["turn_q_jump"] = round(float(max(np.max(np.abs(b - a)) for a, b in zip([q_press] + qs[:-1], qs))), 3)
        m.send(label="turn", settle=6)
        report["turn"] = "sent"
    except MotionError as error:
        report["turn"] = str(error)[:160]
    found2 = locate()
    report["cube_after_turn"] = short(found2)
    if found2 is not None:
        found = found2
    c = found["point"]
    report["side"] = side
    grasp = pose((float(c[0]), float(c[1]), float(c[2]) + 0.005), np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]))
    sent = False
    errors = []
    for above, linear in ((0.10, True), (0.07, True), (0.13, True), (0.10, False), (0.06, False)):
        try:
            m = api.motion()
            m.tcp(side, offset(grasp, dz=above), gripper="open", free_axis=(0, 0, 1))
            m.tcp(side, grasp, linear=linear, free_axis=(0, 0, 1))
            m.hand(side, wuji_power(0.75), rows=20)
            m.tcp(side, offset(grasp, dz=0.10), linear=linear, free_axis=(0, 0, 1))
            if m.rows > 250:
                raise MotionError(f"{m.rows} rows > one Rubik reply")
            report["rows"] = m.rows
            report["variant"] = [above, linear]
            m.send(settle=20, label="pick_cube")
            api.stage_done()
            sent = True
            break
        except MotionError as error:
            errors.append(str(error)[:120])
    if not sent:
        report["pick_errors"] = errors
report["budget"] = api.budget()["phases"]
api.end_episode()
