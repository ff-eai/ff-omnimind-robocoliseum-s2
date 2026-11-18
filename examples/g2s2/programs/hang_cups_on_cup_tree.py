# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
active = info["active_arms"]
table = api.table_height("head")
report["embodiment"] = info["embodiment"]
tree = api.find_object(["cup tree", "mug tree stand", "cup holder rack"], size=(0.05, 0.40))
report["tree"] = None if tree is None else tree.summary()
cup = api.find_object(["mug", "coffee mug", "ceramic cup", "cup"], size=(0.04, 0.18), table_z=table, z_tolerance=0.05,
                      exclude=[tree.mid] if tree is not None else [], exclude_radius=0.06)
report["cup"] = None if cup is None else cup.summary()
done = False
if cup is not None:
    c = cup.mid
    radius = 0.5 * float(min(cup.extent[0], cup.extent[1]))
    radius = min(max(radius, 0.02), 0.06)
    sides = [api.choose_arm(c)] + [s for s in active if s != api.choose_arm(c)]
    for side in sides:
        sh = api.shoulder(side)
        to_robot = np.array([float(sh[0] - c[0]), float(sh[1] - c[1])])
        to_robot = to_robot / max(1e-6, float(np.linalg.norm(to_robot)))
        for ang in (0.0, 0.6, -0.6, 1.2, -1.2):
            d = np.array([math.cos(ang) * to_robot[0] - math.sin(ang) * to_robot[1], math.sin(ang) * to_robot[0] + math.cos(ang) * to_robot[1]])
            rim = np.array([float(c[0] + d[0] * (radius - 0.004)), float(c[1] + d[1] * (radius - 0.004)), cup.top_z])
            radial = math.atan2(float(d[1]), float(d[0]))
            try:
                plan = api.top_grasp(side, center=rim, top_z=cup.top_z, bottom_z=table, yaw=radial + math.pi / 2.0,
                                     depth_into=0.025, lift=0.10, above=0.08)
            except MotionError as error:
                report.setdefault("plan_errors", []).append(f"{side} {ang}: {str(error)[:160]}")
                continue
            plan.send(depth_next=True, label="pick_cup_rim")
            report["side"], report["rim_angle"] = side, ang
            report["holding"] = api.holding(side)
            done = True
            break
        if done:
            break
    if done and report.get("holding"):
        api.stage_done()
report["budget"] = api.budget()["phases"]
api.end_episode()
