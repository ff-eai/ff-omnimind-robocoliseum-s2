# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

info = api.task_info()
active = info["active_arms"]
report["embodiment"] = info["embodiment"]
EMB = report["embodiment"]
table = api.table_height("head")
report["table_z"] = round(table, 4)
cup = api.find_object(["mug", "coffee mug", "ceramic cup", "cup"], size=(0.04, 0.18), table_z=table, z_tolerance=0.05)
report["cup"] = None if cup is None else cup.summary()
tree = api.find_object(["cup tree", "mug tree stand", "cup holder rack"], size=(0.05, 0.40), table_z=table, z_tolerance=0.08,
                       exclude=[cup.mid] if cup is not None else [], exclude_radius=0.07)
report["tree"] = None if tree is None else tree.summary()

FAST = Limits(joint_velocity=2.4, joint_acceleration=9.0, linear_velocity=0.30)
CAMS = {"g2_omnipicker": (240.0, 140.0), "g1_omnipicker": (240.0, 140.0), "agilex": (320.0, 150.0), "arx": (240.0, 110.0)}
V_FREE, LOOK_V = CAMS.get(EMB, (240.0, 140.0))
GRASP_IN = 0.008
DEPTH_RIM = {"arx": 0.012}.get(EMB, 0.025)
TWO_ARM = len(active) == 2


def look_pose(side, cam, target, drop, yaw):
    G = api.grasp_pose(side)
    V = api.camera(cam)
    T_go = np.linalg.inv(G) @ V.T_base_optical
    d = V.model.pixel_rays(np.array([float(V.model.cx)]), np.array([LOOK_V]))[0]
    d = d / float(np.linalg.norm(d))
    Tg = top_down((0.0, 0.0, 0.0), yaw)
    R = Tg[:3, :3]
    r = R @ (T_go[:3, :3] @ d)
    if float(r[2]) > -0.5:
        return None
    t = drop / (-float(r[2]))
    Tg[:3, 3] = np.asarray(target, dtype=float) - t * r - R @ T_go[:3, 3]
    return Tg


def wrist_yaw(side):
    G = api.grasp_pose(side)
    return math.atan2(float(G[1, 1]), float(G[0, 1]))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rnd(v, n=4):
    if isinstance(v, np.ndarray):
        return [round(float(x), n) for x in v]
    if isinstance(v, float):
        return round(v, n)
    return v


def free_points(view):
    H, W = view.depth_m.shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:int(V_FREE), :] = True
    return view.points_from_mask(mask, stride=1)


def waist_variants():
    w0 = np.array(api.robot_state()["waist"], dtype=float)
    out = [None]
    if w0.size == 5:
        for d2, d3 in ((0.2, -0.2), (0.0, 0.15), (0.4, -0.4), (0.0, 0.3)):
            w = w0.copy()
            w[1] += d2
            w[2] += d3
            if 0.0 <= w[1] <= 2.6 and -1.85 <= w[2] <= 1.5:
                out.append(w)
    elif w0.size == 2:
        for dp, dl in ((-0.15, 0.0), (0.0, -0.08), (-0.3, 0.0), (0.0, 0.08), (-0.2, 0.08), (-0.3, -0.08), (0.15, 0.0)):
            w = w0.copy()
            w[0] += dp
            w[1] += dl
            if 0.0 <= w[0] <= 1.55 and 0.0 <= w[1] <= 0.54:
                out.append(w)
    return out


def tree_from_head(det):
    view = api.camera("head")
    if view.depth_m is None:
        return None
    P = view.points_from_mask(None, stride=1)
    mid = np.asarray(det.mid[:2], dtype=float)
    P = P[np.hypot(P[:, 0] - mid[0], P[:, 1] - mid[1]) < 0.15]
    Ht = max(float(np.percentile(P[(P[:, 2] > table + 0.01) & (P[:, 2] < table + 0.8)][:, 2], 99)) - float(table), 0.05) \
        if int(((P[:, 2] > table + 0.01) & (P[:, 2] < table + 0.8)).sum()) > 50 else 0.05
    band = P[(P[:, 2] > table + 0.10 * Ht) & (P[:, 2] < table + 0.35 * Ht)]
    if band.shape[0] < 20:
        return None
    c = np.median(band[:, :2], axis=0)
    k = band
    for it_ in range(3):
        d = np.hypot(band[:, 0] - c[0], band[:, 1] - c[1])
        k = band[d < 0.02]
        if k.shape[0] < 10:
            return None
        c = np.median(k[:, :2], axis=0)
    cam = view.T_base_optical[:2, 3]
    vdir = c - cam
    vdir = vdir / max(1e-6, float(np.linalg.norm(vdir)))
    across = np.array([-vdir[1], vdir[0]])
    sp = (k[:, :2] - c) @ across
    r_post = min(max(0.5 * float(np.percentile(sp, 95) - np.percentile(sp, 5)), 0.004), 0.03)
    c = c + 0.64 * r_post * vdir
    r = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    r_max = max(0.5 * float(max(det.extent[0], det.extent[1])) + 0.02, 0.6 * Ht)
    pp = P[(r > r_post + 0.02) & (r < r_max) & (P[:, 2] > table + 0.35 * Ht)]
    out = {"post": c, "r_post": r_post, "n_post": int(k.shape[0]), "pegs": []}
    if pp.shape[0] < 15:
        return out
    az = np.arctan2(pp[:, 1] - c[1], pp[:, 0] - c[0])
    ia = np.floor((az + math.pi) / math.radians(20.0)).astype(int) % 18
    iz = np.floor((pp[:, 2] - table) / 0.03).astype(int)
    cells = {}
    for j in range(pp.shape[0]):
        cells.setdefault((int(ia[j]), int(iz[j])), []).append(j)
    cells = {key: v for key, v in cells.items() if len(v) >= 5}
    seen = set()
    for key in sorted(cells, key=lambda q: -len(cells[q])):
        if key in seen:
            continue
        comp, stack = [], [key]
        seen.add(key)
        while stack:
            q = stack.pop()
            comp.extend(cells[q])
            for da in (-1, 0, 1):
                for dzi in (-1, 0, 1):
                    nb = ((q[0] + da) % 18, q[1] + dzi)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        Q = pp[np.array(comp)]
        if Q.shape[0] < 15:
            continue
        rq = np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1])
        tip = Q[rq >= np.percentile(rq, 75)]
        u = np.median(tip[:, :2], axis=0) - c
        rk = float(np.linalg.norm(u))
        if rk < 1e-3:
            continue
        u = u / rk
        lat = (tip[:, :2] - c) @ np.array([-u[1], u[0]])
        r_peg = min(max(0.5 * float(np.percentile(lat, 95) - np.percentile(lat, 5)), 0.003), 0.015)
        slope = float(np.polyfit(rq, Q[:, 2], 1)[0]) if float(np.ptp(rq)) > 0.01 else 0.0
        K = np.array([float(c[0] + rk * u[0]), float(c[1] + rk * u[1]), float(np.median(tip[:, 2])) - r_peg])
        out["pegs"].append({"K": K, "u": u, "rk": rk, "slope": min(max(slope, -0.2), 1.0), "r_peg": r_peg, "n": int(Q.shape[0])})
    return out


def circle_fit(pts, r0=None):
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(pts.shape[0])])
    b = pts[:, 0] ** 2 + pts[:, 1] ** 2
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = np.array([sol[0] / 2.0, sol[1] / 2.0])
    r = math.sqrt(max(1e-6, float(sol[2]) + float(c @ c)))
    if r0 is not None and abs(r - r0) > 0.02:
        r = r0
    inl = pts
    for it_ in range(4):
        d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
        inl = pts[np.abs(d - r) < 0.004]
        if inl.shape[0] < 20:
            break
        for k_ in range(4):
            v = inl[:, :2] - c
            dd = np.maximum(1e-6, np.hypot(v[:, 0], v[:, 1]))
            J = np.column_stack([-v[:, 0] / dd, -v[:, 1] / dd, -np.ones(dd.shape[0])])
            step = np.linalg.lstsq(J, dd - r, rcond=None)[0]
            c = c - step[:2]
            r = r - float(step[2])
    d = np.hypot(inl[:, 0] - c[0], inl[:, 1] - c[1])
    return c, r, float(np.sqrt(np.mean((d - r) ** 2))) if inl.shape[0] else 1.0, int(inl.shape[0])


def cup_from_wrist(view, prior_xy):
    if view.depth_m is None:
        return {"depth": None}
    P = free_points(view)
    p0 = np.asarray(prior_xy[:2], dtype=float)
    P = P[P[:, 2] < table + 0.13]
    dist0 = np.hypot(P[:, 0] - p0[0], P[:, 1] - p0[1])
    near = P[dist0 < 0.10]
    if near.shape[0] < 200:
        return {"near": int(near.shape[0])}
    dn = dist0[dist0 < 0.10]
    top = float(np.percentile(near[dn < 0.07, 2], 98)) if int((dn < 0.07).sum()) > 50 else float(np.percentile(near[:, 2], 98))
    ring = near[(near[:, 2] > top - 0.003) & (near[:, 2] < top + 0.003)]
    if ring.shape[0] < 40:
        return {"ring": int(ring.shape[0])}
    c, r, rms, n_in = circle_fit(ring)
    d = np.hypot(near[:, 0] - c[0], near[:, 1] - c[1])
    ann = near[(d > r + 0.045) & (d < r + 0.075) & (near[:, 2] < top - 0.03)]
    tab = float(np.median(ann[:, 2])) if ann.shape[0] >= 100 else float(table)
    height = top - tab
    out = {"centre": c, "r": r, "rms": rms, "n_ring": n_in, "top": top, "table_w": tab, "n_table": int(ann.shape[0]), "height": height}
    if not (0.05 < height < 0.115 and 0.028 < r < 0.058 and rms < 0.004):
        out["bad"] = "not an upright mug rim"
        return out
    body = near[(near[:, 2] < top - 0.004) & (near[:, 2] > tab + 0.012)]
    db = np.hypot(body[:, 0] - c[0], body[:, 1] - c[1])
    hp = body[(db > r + 0.008) & (db < r + 0.06)]
    dbh = db[(db > r + 0.008) & (db < r + 0.06)]
    out["n_handle"] = int(hp.shape[0])
    if hp.shape[0] >= 15:
        a = np.arctan2(hp[:, 1] - c[1], hp[:, 0] - c[0])
        a0 = math.atan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a))))
        keep = np.abs(np.arctan2(np.sin(a - a0), np.cos(a - a0))) < math.radians(50.0)
        if int(keep.sum()) >= 10:
            a0 = math.atan2(float(np.mean(np.sin(a[keep]))), float(np.mean(np.cos(a[keep]))))
        out["handle"] = a0
        hk = hp[keep] if int(keep.sum()) >= 10 else hp
        dk = dbh[keep] if int(keep.sum()) >= 10 else dbh
        smax_m = float(np.percentile(dk, 95))
        z_lo_h, z_hi_h = float(np.percentile(hk[:, 2], 5)), float(np.percentile(hk[:, 2], 95))
        out["mug"] = {"s": 0.5 * (r + smax_m), "zh": top - 0.5 * (z_lo_h + z_hi_h), "smax": smax_m, "h_span": z_hi_h - z_lo_h}
        out["handle_spread_deg"] = round(math.degrees(float(np.std(np.arctan2(np.sin(a[keep] - a0), np.cos(a[keep] - a0))))), 1) if int(keep.sum()) else None
    return out


def refine_knob(view, K):
    if view.depth_m is None:
        return None, {"depth": None}
    P = free_points(view)
    sel = P[(np.hypot(P[:, 0] - K[0], P[:, 1] - K[1]) < 0.025) & (np.abs(P[:, 2] - K[2]) < 0.03)]
    if sel.shape[0] < 30:
        return None, {"n": int(sel.shape[0])}
    zt = float(np.percentile(sel[:, 2], 99.5))
    cap = sel[sel[:, 2] > zt - 0.0025]
    info = {"n": int(sel.shape[0]), "n_cap": int(cap.shape[0])}
    if cap.shape[0] < 8:
        return None, info
    cc_ = np.mean(cap[:, :2], axis=0)
    a_cap = float(np.percentile(np.hypot(cap[:, 0] - cc_[0], cap[:, 1] - cc_[1]), 90))
    R_k = min(max((a_cap ** 2 + 0.0025 ** 2) / (2.0 * 0.0025), 0.003), 0.015)
    info["r_knob_mm"] = round(1000.0 * R_k, 1)
    Kn = np.array([float(cc_[0]), float(cc_[1]), zt - R_k])
    info["shift_mm"] = rnd((Kn - K) * 1000.0, 1)
    if float(np.hypot(Kn[0] - K[0], Kn[1] - K[1])) > 0.02 or abs(float(Kn[2] - K[2])) > 0.02:
        return None, info
    return Kn, info


def rim_pick():
    done = False
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


def hang_targets(K, u, sgn, hole_g, slope=0.0):
    h = sgn * np.array([u[1], -u[0]])
    yc = math.atan2(float(h[1]), float(h[0])) - math.atan2(float(hole_g[0]), float(hole_g[1]))
    off = top_down((0.0, 0.0, 0.0), yc)[:3, :3] @ np.asarray(hole_g, dtype=float)

    def G(H):
        return top_down((float(H[0] - off[0]), float(H[1] - off[1]), float(H[2] - off[2])), yc)

    P1 = np.array([K[0] + 0.03 * u[0], K[1] + 0.03 * u[1], K[2] + 0.002])
    P2 = np.array([K[0] - 0.015 * u[0], K[1] - 0.015 * u[1], K[2] - 0.015 * slope])
    P0 = P1 + np.array([0.0, 0.0, 0.05])
    return h, [G(P0), G(P1), G(P2)], (P0, P1, P2)


def cup_in_hand(view, side, r, ax_pred):
    if view.depth_m is None:
        return None, {"depth": None}
    G = api.grasp_pose(side)
    gp = G[:3, 3]
    zr = float(gp[2]) + DEPTH_RIM
    P = view.points_from_mask(None, stride=1)
    sel = P[(np.abs(P[:, 2] - zr) < 0.004) & (np.hypot(P[:, 0] - ax_pred[0], P[:, 1] - ax_pred[1]) < r + 0.025)]
    info = {"n": int(sel.shape[0]), "rim_z": round(zr, 4)}
    if sel.shape[0] < 60:
        return None, info
    if sel.shape[0] > 4000:
        sel = sel[np.linspace(0, sel.shape[0] - 1, 4000).astype(int)]
    Gi = np.linalg.inv(G)
    loc = (Gi[:3, :3] @ sel.T).T + Gi[:3, 3]
    info["pts_mm"] = [[int(round(1000 * v)) for v in q] for q in loc[np.linspace(0, loc.shape[0] - 1, min(120, loc.shape[0])).astype(int)]]
    best = (0, None)
    for dx in np.arange(-0.016, 0.0161, 0.001):
        for dy in np.arange(-0.016, 0.0161, 0.001):
            cx, cy = ax_pred[0] + dx, ax_pred[1] + dy
            nin = int(np.sum(np.abs(np.hypot(sel[:, 0] - cx, sel[:, 1] - cy) - r) < 0.002))
            if nin > best[0]:
                best = (nin, np.array([cx, cy]))
    c = best[1]
    inl = sel[np.abs(np.hypot(sel[:, 0] - c[0], sel[:, 1] - c[1]) - r) < 0.003]
    for k_ in range(5):
        v = inl[:, :2] - c
        dd = np.maximum(1e-6, np.hypot(v[:, 0], v[:, 1]))
        J = -v / dd[:, None]
        c = c - np.linalg.lstsq(J, dd - r, rcond=None)[0]
    frac = best[0] / float(sel.shape[0])
    d = float(np.hypot(c[0] - gp[0], c[1] - gp[1]))
    info.update({"inliers": best[0], "frac": round(frac, 2), "axis_from_grasp": round(d, 4), "shift_mm": rnd((c - ax_pred) * 1000.0, 1)})
    if best[0] < 60 or frac < 0.25 or float(np.hypot(c[0] - ax_pred[0], c[1] - ax_pred[1])) > 0.014:
        return None, info
    return c, info


def pick_and_hang():
    side = api.choose_arm(cup.mid) if TWO_ARM else active[0]
    report["side"] = side
    cam = "hand_" + side
    looked = None
    y0 = wrist_yaw(side)
    for drop in (0.13, 0.10, 0.17):
        for dy in (0.0, 0.6, -0.6, 1.2, -1.2, math.pi):
            T = look_pose(side, cam, (float(cup.mid[0]), float(cup.mid[1]), float(cup.top_z)), drop, y0 + dy)
            if T is None:
                continue
            try:
                m = api.motion()
                m.tcp(side, T, gripper="open", limits=FAST)
                m.send(depth_next=True, label="look_cup")
                looked = (drop, y0 + dy)
                break
            except MotionError as error:
                if len(report.get("look_errors", [])) < 4:
                    report.setdefault("look_errors", []).append(f"{drop} {round(dy, 2)}: {str(error)[:60]}")
        if looked is not None:
            break
    if looked is None:
        report["fallback"] = "rim pick (no cup look)"
        rim_pick()
        return
    fc = cup_from_wrist(api.camera(cam), cup.mid)
    report["cup_wrist"] = {k: ({kk: rnd(vv) for kk, vv in v.items()} if isinstance(v, dict) else rnd(v)) for k, v in fc.items()}
    if "handle" not in fc or "bad" in fc:
        report["fallback"] = "rim pick (" + str(fc.get("bad", "no handle from the wrist view")) + ")"
        rim_pick()
        return
    c = fc["centre"]
    r_rim = fc["r"]
    mm = fc["mug"]
    hdir = np.array([math.cos(fc["handle"]), math.sin(fc["handle"])])
    top = fc["top"]
    zg = top - DEPTH_RIM
    yaw_g = fc["handle"]
    rim = c - (r_rim - GRASP_IN) * hdir

    def hole_pred(fl):
        return np.array([0.0, (r_rim - GRASP_IN + mm["s"]) * (1.0 if fl == 0.0 else -1.0), mm["zh"] - DEPTH_RIM])

    def pick_poses(fl):
        yc = wrap(yaw_g + fl)
        return [top_down((float(rim[0]), float(rim[1]), zg + dz), yc) for dz in (0.04, 0.0, 0.08)]

    cands = []
    tr = None
    if tree is not None:
        tr = tree_from_head(tree)
        report["tree_est"] = None if tr is None else {"post": rnd(tr["post"]), "r_post": rnd(tr["r_post"]), "n_post": tr["n_post"],
                                                    "pegs": [{"K": rnd(pg["K"]), "az": round(math.degrees(math.atan2(pg["u"][1], pg["u"][0])), 1),
                                                              "slope": round(pg["slope"], 3), "r_peg": rnd(pg["r_peg"]), "n": pg["n"]}
                                                             for pg in tr["pegs"]]}
    if tr is not None and tr.get("pegs"):
        post = tr["post"]
        pegs = tr["pegs"]
        knobs = [(pg["K"], pg["u"], round(float(pg["K"][2]), 3), pg["slope"]) for pg in pegs if pg["u"][0] < 0.3]
        zmid = float(np.median([float(pg["K"][2]) for pg in pegs]))
        knobs.sort(key=lambda t: (not (t[1][0] < -0.3), abs(float(t[0][2]) - zmid), float(t[0][2])))
        waists = waist_variants()
        flips = sorted((0.0, math.pi), key=lambda f: abs(wrap(yaw_g + f - wrist_yaw(side))))
        n_quick = 0
        for K, u, zk, slope in knobs:
            for sgn in (1.0, -1.0):
                h, Gs0, Ps = hang_targets(K, u, sgn, hole_pred(0.0), slope)
                why = blocked(post, pegs, K, h, Ps, mm, r_rim, fc["height"], tr["r_post"])
                if why is not None:
                    report.setdefault("plan_blocked", []).append(f"{zk} {round(math.degrees(math.atan2(u[1], u[0])))} {sgn}: {why}")
                    continue
                for w in waists:
                    for fl in flips:
                        if api.reply_elapsed() > 50.0 or len(cands) >= 4 or (cands and api.reply_elapsed() > 15.0):
                            break
                        h, Gs, Ps = hang_targets(K, u, sgn, hole_pred(fl), slope)
                        n_quick += 1
                        try:
                            m = api.motion()
                            m.tcp(side, Gs[0], limits=FAST, waist=w)
                            m.tcp(side, Gs[1], linear=True, limits=Limits(linear_velocity=0.15))
                            m.tcp(side, Gs[2], linear=True, limits=Limits(linear_velocity=0.08))
                            cands.append({"K": K, "u": u, "zk": zk, "slope": slope, "sgn": sgn, "w": w, "flip": fl})
                        except MotionError as error:
                            if len(report.get("plan_errors", [])) < 6:
                                report.setdefault("plan_errors", []).append(f"{zk} {round(math.degrees(math.atan2(u[1], u[0])))} {sgn} {round(fl, 1)}: {str(error)[:50]}")
        report["plan_quick"] = [n_quick, len(cands), round(api.reply_elapsed(), 1)]
    knob_shift = np.zeros(3)
    if cands:
        Kp = cands[0]["K"]
        done = False
        for drop in (0.16, 0.12):
            for dy in (0.0, 0.6, -0.6, 1.2, -1.2, math.pi):
                T = look_pose(side, cam, (float(Kp[0]), float(Kp[1]), float(Kp[2])), drop, wrist_yaw(side) + dy)
                if T is None:
                    continue
                try:
                    m = api.motion()
                    m.tcp(side, T, gripper="open", limits=FAST, waist=cands[0]["w"])
                    used = int(api.budget()["phases"][0]["rows"])
                    if used + m.rows + 250 > 440:
                        report["knob_look_skipped"] = [used, m.rows]
                        done = True
                        break
                    m.send(depth_next=True, label="look_knob")
                    done = True
                    break
                except MotionError as error:
                    if len(report.get("knob_look_errors", [])) < 4:
                        report.setdefault("knob_look_errors", []).append(f"{drop} {round(dy, 2)}: {str(error)[:50]}")
            if done and "knob_look_skipped" not in report:
                Kr, kinfo = refine_knob(api.camera(cam), Kp)
                report["knob_wrist"] = kinfo
                if Kr is not None:
                    knob_shift = Kr - Kp
                break
    plan = None
    n_full = 0
    for cd in cands:
        if api.reply_elapsed() > 50.0:
            break
        same = float(np.linalg.norm(cd["K"] - cands[0]["K"])) < 1e-6
        K = cd["K"] + (knob_shift if same else np.array([knob_shift[0], knob_shift[1], 0.0]))
        h, Gs, Ps = hang_targets(K, cd["u"], cd["sgn"], hole_pred(cd["flip"]), cd["slope"])
        pk = pick_poses(cd["flip"])
        for wa in [cd["w"]] + [w for w in waist_variants() if not (w is None and cd["w"] is None)][:4]:
            n_full += 1
            try:
                m = api.motion()
                m.tcp(side, pk[0], limits=FAST, waist=wa)
                m.tcp(side, pk[1], linear=True, limits=Limits(linear_velocity=0.20))
                m.tcp(side, pk[2], linear=True, limits=Limits(linear_velocity=0.30))
                m.tcp(side, Gs[0], limits=FAST, waist=cd["w"])
                m.tcp(side, Gs[1], linear=True, limits=Limits(linear_velocity=0.15))
                m.tcp(side, Gs[2], linear=True, limits=Limits(linear_velocity=0.08))
                plan = dict(cd)
                plan["K"] = K
                plan["wa"] = wa
                break
            except MotionError as error:
                if len(report.get("chain_errors", [])) < 4:
                    report.setdefault("chain_errors", []).append(str(error)[:60])
        if plan is not None:
            break
    report["plan_full"] = [n_full, round(api.reply_elapsed(), 1)]
    report["plan"] = None if plan is None else {"knob": rnd(plan["K"]), "u_deg": round(math.degrees(math.atan2(plan["u"][1], plan["u"][0])), 1),
                                                "level_z": plan["zk"], "sgn": plan["sgn"], "flip": round(plan["flip"], 2),
                                                "waist": None if plan["w"] is None else rnd(plan["w"], 3)}
    w_plan = None if plan is None else plan["wa"]
    picked = None
    fls = [plan["flip"], wrap(plan["flip"] + math.pi)] if plan is not None else sorted((0.0, math.pi), key=lambda f: abs(wrap(yaw_g + f - wrist_yaw(side))))
    for fl in fls:
        pk = pick_poses(fl)
        try:
            m = api.motion()
            m.tcp(side, pk[0], gripper="open", limits=FAST, waist=w_plan)
            m.tcp(side, pk[1], linear=True, limits=Limits(linear_velocity=0.20))
            m.hold(8)
            m.gripper(side, "close")
            m.hold(12)
            m.tcp(side, pk[2], linear=True, limits=Limits(linear_velocity=0.30))
            m.send(depth_next=True, label="pick_cup")
            picked = wrap(yaw_g + fl)
            break
        except MotionError as error:
            report.setdefault("pick_errors", []).append(f"{round(math.degrees(wrap(yaw_g + fl)))}: {str(error)[:70]}")
    if picked is None:
        report["fallback"] = "rim pick (no oriented rim grasp)"
        rim_pick()
        return
    held = bool(api.holding(side))
    report["holding"] = held
    st_ = api.robot_state()
    if "opening" in st_:
        report["opening_after_pick"] = round(float(st_["opening"][side]), 4)
    report["grasp"] = {"rim": rnd(rim), "z": round(zg, 4), "yaw_deg": round(math.degrees(picked), 1)}
    if not held:
        return
    api.stage_done()
    if plan is None:
        return
    flip = 0.0 if abs(wrap(picked - yaw_g)) < 0.1 else math.pi
    K = plan["K"]
    hole_g = hole_pred(flip)
    Gp = api.grasp_pose(side)
    hp_w = Gp[:3, :3] @ hole_g + Gp[:3, 3]
    hdir_w = np.array([hp_w[0] - Gp[0, 3], hp_w[1] - Gp[1, 3]])
    hdir_w = hdir_w / max(1e-6, float(np.linalg.norm(hdir_w)))
    ax_pred = np.array([Gp[0, 3], Gp[1, 3]]) + (r_rim - GRASP_IN) * hdir_w
    ax, hinfo = cup_in_hand(api.camera(cam), side, r_rim, ax_pred)
    report["in_hand"] = hinfo
    if ax is not None:
        G = api.grasp_pose(side)
        gp = G[:3, 3]
        hm = np.array([ax[0] - gp[0], ax[1] - gp[1]])
        hm = hm / max(1e-6, float(np.linalg.norm(hm)))
        Hw = np.array([ax[0] + mm["s"] * hm[0], ax[1] + mm["s"] * hm[1], float(gp[2]) + DEPTH_RIM - mm["zh"]])
        hole_g = (np.linalg.inv(G) @ np.array([Hw[0], Hw[1], Hw[2], 1.0]))[:3]
        report["hole_g"] = rnd(hole_g)
        report["hole_g_pred"] = rnd(hole_pred(flip))
    slp = float(plan.get("slope", 0.0))
    tries = [(K, plan["u"], plan["sgn"], plan["w"], True, slp), (K, plan["u"], plan["sgn"], plan["w"], False, slp)]
    for sg in (plan["sgn"], -plan["sgn"]):
        for w in waist_variants():
            tries.append((K, plan["u"], sg, w, True, slp))
    for cd in cands:
        if float(np.linalg.norm(cd["K"][:2] - (plan["K"][:2] - knob_shift[:2]))) > 1e-6:
            Kc = cd["K"] + np.array([knob_shift[0], knob_shift[1], 0.0])
            for w in (cd["w"], None):
                tries.append((Kc, cd["u"], cd["sgn"], w, True, float(cd.get("slope", 0.0))))
    hung = None
    for Kt, ut, sgn, w, lin, slt in tries:
        if api.reply_elapsed() > 70.0:
            report["hang_timeout"] = True
            break
        h, Gs, Ps = hang_targets(Kt, ut, sgn, hole_g, slt)
        try:
            m = api.motion()
            m.tcp(side, Gs[0], limits=FAST, waist=w, waist_limits=Limits(joint_velocity=1.0))
            if lin:
                m.tcp(side, Gs[1], linear=True, limits=Limits(linear_velocity=0.15))
                m.tcp(side, Gs[2], linear=True, limits=Limits(linear_velocity=0.08))
            else:
                m.tcp(side, Gs[1], limits=Limits(joint_velocity=0.5))
                m.tcp(side, Gs[2], limits=Limits(joint_velocity=0.3))
            m.hold(10)
            m.gripper(side, "open")
            m.hold(15)
            up = Gs[2].copy()
            up[2, 3] += 0.05
            m.tcp(side, up, linear=True, limits=Limits(linear_velocity=0.15))
            m.send(depth_next=True, label="hang_cup")
            hung = {"knob": rnd(Kt), "sgn": sgn, "waist": None if w is None else rnd(w, 3), "linear": lin}
            break
        except MotionError as error:
            if len(report.get("hang_errors", [])) < 6:
                report.setdefault("hang_errors", []).append(str(error)[:60])
    report["hang"] = hung


def blocked(post, pegs, K, h, holes, mm, r_rim, height, r_post):
    for H in holes:
        cc = np.array([H[0] - mm["s"] * h[0], H[1] - mm["s"] * h[1]])
        zc = H[2] + mm["zh"] - 0.5 * height
        if float(np.hypot(cc[0] - post[0], cc[1] - post[1])) < r_rim + r_post + 0.004:
            return "post"
        for pg in pegs:
            K2, rk, sl2 = pg["K"], pg["rk"], pg["slope"]
            if float(np.linalg.norm(K2 - K)) < 1e-6:
                continue
            for f in (0.2, 0.4, 0.6, 0.8, 1.0):
                p = np.array([post[0] + f * (K2[0] - post[0]), post[1] + f * (K2[1] - post[1]), K2[2] - (1.0 - f) * rk * sl2])
                if float(np.hypot(p[0] - cc[0], p[1] - cc[1])) < r_rim + 0.006 and abs(float(p[2]) - zc) < 0.5 * height + 0.006:
                    return "peg"
    return None


start_q = {s: np.array(q, dtype=float) for s, q in api.robot_state()["arm"].items()}
w_start = np.array(api.robot_state()["waist"], dtype=float)
if cup is None:
    pass
elif EMB not in ("g2_omnipicker", "g1_omnipicker"):
    rim_pick()
else:
    try:
        pick_and_hang()
    except EpisodeEnded:
        raise
    except Exception as error:
        report["pick_and_hang_exception"] = str(error)[:200]
        if not report.get("holding"):
            try:
                rim_pick()
            except EpisodeEnded:
                raise
            except Exception as error2:
                report["rim_pick_exception"] = str(error2)[:200]
report["budget"] = api.budget()["phases"]
api.end_episode()
