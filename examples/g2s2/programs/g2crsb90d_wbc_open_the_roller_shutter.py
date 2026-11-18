# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

shutter = api.find_object(["orange rolling door", "roller shutter", "orange shutter door"], size=(0.8, 6.0), away_from_grippers=0.0)
report["shutter"] = None if shutter is None else shutter.summary()
if shutter is not None:
    yaw, length, width = shutter.principal_yaw()
    normal = yaw - math.pi / 2.0
    if math.cos(normal) < 0.0:
        normal += math.pi
    while normal > math.pi:
        normal -= 2 * math.pi
    plane_d = float(np.percentile(shutter.points[:, 0] * math.cos(normal) + shutter.points[:, 1] * math.sin(normal), 30))
    c = shutter.mid
    along = np.array([-math.sin(normal), math.cos(normal)])
    centre_lat = float(c[0] * along[0] + c[1] * along[1])
    go = max(0.0, plane_d - 0.85)
    tx = go * math.cos(normal) + centre_lat * along[0]
    ty = go * math.sin(normal) + centre_lat * along[1]
    report["move"] = [round(tx, 3), round(ty, 3), round(math.degrees(normal), 1)]
    try:
        m = api.motion()
        m.base(tx, ty, normal, speed=0.35)
        m.hold(60)
        m.send(depth_next=True, label="to_shutter")
        api.stage_done()
    except MotionError as error:
        report["move_error"] = str(error)[:300]
report["budget"] = api.budget()["phases"]
api.end_episode()
