# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
active = info["active_arms"]
report["embodiment"] = info["embodiment"]
table = api.table_height("head")
kb = api.find_object(["white keyboard", "numeric keypad"], size=(0.05, 0.40))
report["keyboard"] = None if kb is None else kb.summary()
excl = [kb.mid] if kb is not None else []
queries = ["small gray tool", "gray plastic holder", "gray keycap puller"]
puller = api.find_object(queries, size=(0.02, 0.12), table_z=table, z_tolerance=0.05, exclude=excl, exclude_radius=0.07)
report["puller"] = None if puller is None else puller.summary()
if puller is not None:
    first = api.choose_arm(puller.mid)
    for side in [first] + [s for s in active if s != first]:
        pick = api.pick_object(side, queries,
                               find={"size": (0.02, 0.12), "table_z": table, "z_tolerance": 0.05, "exclude": excl,
                                     "exclude_radius": 0.07},
                               grasp={"depth_into": 0.02, "lift": 0.12, "above": 0.08})
        report["pick_" + side] = pick
        if pick["attempts"]:
            if pick["ok"]:
                api.stage_done()
            break
report["budget"] = api.budget()["phases"]
api.end_episode()
