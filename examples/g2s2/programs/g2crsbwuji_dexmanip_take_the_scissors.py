# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

SC_Q = ["scissors", "pair of scissors", "scissors handle"]
table = api.table_height("head")
sc = api.find_object(SC_Q, size=(0.05, 0.25), table_z=table, z_tolerance=0.04)
report["scissors"] = None if sc is None else sc.summary()
if sc is not None:
    side = api.instructed_arm(api.segment_text(0)) or ("right" if "right arm picks" in str(api.task_info().get("prompt", "")).lower() else "left")
    report["side"] = side
    yaw_long = sc.principal_yaw()[0]
    variants = ((0.030, wuji_power(0.9), yaw_long + math.pi / 2.0), (0.030, wuji_power(0.9), None),
                (0.022, wuji_pinch(1.0), yaw_long + math.pi / 2.0))
    for i, (gz_off, closure, yaw) in enumerate(variants):
        try:
            m = api.motion()
            if i > 0:
                m.hand(side, "open", rows=15)
            api.hand_top_grasp(side, sc.mid, grasp_z=sc.top_z - gz_off, closure=closure, yaw=yaw, lift=0.10, motion=m)
            away = m.grasp_pose(side).copy()
            away[1, 3] += -0.08 if float(sc.mid[1]) > 0.0 else 0.08
            m.tcp(side, away)
            m.send(depth_next=True, label=f"pick_scissors{i}")
        except MotionError as error:
            report.setdefault("plan_errors", []).append(str(error)[:200])
            continue
        chk = api.lifted_check(SC_Q, sc, size=(0.03, 0.25))
        chk["hand_lag"] = api.hand_lag(side)
        report.setdefault("checks", []).append(chk)
        if chk.get("lifted"):
            api.stage_done()
            break
report["budget"] = api.budget()["phases"]
api.end_episode()
