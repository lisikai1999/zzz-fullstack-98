"""Four-bar linkage kinematics solver using closed-loop vector equations."""

import numpy as np
from dataclasses import dataclass


@dataclass
class LinkageParams:
    a: float  # crank (input link)
    b: float  # coupler
    c: float  # follower (output link)
    d: float  # ground (frame)
    coupler_x: float = 0.0  # coupler point offset along coupler
    coupler_y: float = 0.0  # coupler point offset perpendicular to coupler


@dataclass
class LinkageState:
    theta2: float  # input angle (rad)
    theta3: float  # coupler angle (rad)
    theta4: float  # follower angle (rad)
    A: tuple  # ground pivot A (fixed)
    B: tuple  # crank-coupler joint
    C: tuple  # coupler-follower joint
    D: tuple  # ground pivot D (fixed)
    P: tuple  # coupler point
    mu: float  # transmission angle (rad)
    valid: bool


def grashof_condition(a: float, b: float, c: float, d: float) -> dict:
    """Determine mechanism type via Grashof's criterion.

    s + l <= p + q  => Grashof mechanism
    s = shortest, l = longest, p, q = remaining
    """
    lengths = sorted([a, b, c, d])
    s, p, q, l = lengths[0], lengths[1], lengths[2], lengths[3]
    grashof_sum = s + l
    other_sum = p + q
    is_grashof = grashof_sum <= other_sum

    bars = [a, b, c, d]
    shortest_idx = bars.index(min(bars))

    if not is_grashof:
        mech_type = "double_rocker"
        description = "双摇杆机构：不满足Grashof条件(s+l > p+q)，无杆可整转"
    else:
        # Grashof mechanism: type depends on which bar is shortest
        # Bar indices: 0=a(crank/input), 1=b(coupler), 2=c(follower), 3=d(ground)
        change_point = grashof_sum == other_sum
        suffix = "（变换点）" if change_point else ""

        if shortest_idx == 0:
            # Shortest is input link => it can fully rotate => crank-rocker
            mech_type = "crank_rocker"
            description = f"曲柄摇杆机构：最短杆为曲柄（主动件），曲柄可整转{suffix}"
        elif shortest_idx == 3:
            # Shortest is ground => both links adjacent to ground can rotate
            mech_type = "double_crank"
            description = f"双曲柄机构：最短杆为机架，两侧杆均可整转{suffix}"
        elif shortest_idx == 1:
            # Shortest is coupler => double rocker
            mech_type = "double_rocker_grashof"
            description = f"双摇杆机构：最短杆为连杆，满足Grashof但连杆最短{suffix}"
        else:
            # Shortest is follower => crank-rocker (input is crank if adjacent to ground)
            mech_type = "crank_rocker"
            description = f"曲柄摇杆机构：最短杆为从动件侧，曲柄可整转{suffix}"

    return {
        "is_grashof": bool(is_grashof),
        "type": mech_type,
        "description": description,
        "s_plus_l": float(grashof_sum),
        "p_plus_q": float(other_sum),
        "shortest_bar": int(shortest_idx),
        "lengths_sorted": [float(x) for x in lengths],
    }


def solve_position(params: LinkageParams, theta2: float, branch: int = 1) -> LinkageState:
    """Solve four-bar position for given input angle using closed-loop vector equation.

    Closed-loop: a*e^(j*theta2) + b*e^(j*theta3) = d + c*e^(j*theta4)
    Solve for theta3, theta4 given theta2.
    """
    a, b, c, d = params.a, params.b, params.c, params.d

    # Joint B position
    Bx = a * np.cos(theta2)
    By = a * np.sin(theta2)

    # Distance from B to D
    BD_x = d - Bx
    BD_y = -By
    BD = np.sqrt(BD_x**2 + BD_y**2)

    # Check assembly condition
    if BD > b + c or BD < abs(b - c) or BD < 1e-10:
        return LinkageState(
            theta2=theta2, theta3=0, theta4=0,
            A=(0, 0), B=(Bx, By), C=(0, 0), D=(d, 0),
            P=(0, 0), mu=0, valid=False
        )

    # Angle of BD from horizontal
    alpha = np.arctan2(BD_y, BD_x)

    # Triangle BCD: law of cosines for angle at B
    cos_beta = (b**2 + BD**2 - c**2) / (2 * b * BD)
    cos_beta = np.clip(cos_beta, -1, 1)
    beta = np.arccos(cos_beta)

    # Two assembly modes
    if branch == 1:
        theta3 = alpha + beta
    else:
        theta3 = alpha - beta

    # Joint C from B + coupler
    Cx = Bx + b * np.cos(theta3)
    Cy = By + b * np.sin(theta3)

    # Follower angle
    theta4 = np.arctan2(Cy, Cx - d)

    # Transmission angle: angle between coupler and follower at joint C
    mu = abs(theta3 - theta4)
    mu = mu % np.pi
    if mu > np.pi / 2:
        mu = np.pi - mu

    # Coupler point
    coupler_angle = theta3
    Px = Bx + params.coupler_x * np.cos(coupler_angle) - params.coupler_y * np.sin(coupler_angle)
    Py = By + params.coupler_x * np.sin(coupler_angle) + params.coupler_y * np.cos(coupler_angle)

    return LinkageState(
        theta2=theta2,
        theta3=theta3,
        theta4=theta4,
        A=(0.0, 0.0),
        B=(float(Bx), float(By)),
        C=(float(Cx), float(Cy)),
        D=(float(d), 0.0),
        P=(float(Px), float(Py)),
        mu=float(mu),
        valid=True
    )


def compute_full_rotation(params: LinkageParams, steps: int = 360, branch: int = 1) -> dict:
    """Compute mechanism state for full crank rotation."""
    angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    states = []
    trajectory = []
    transmission_angles = []
    min_mu = np.pi
    min_mu_angle = 0

    for theta2 in angles:
        state = solve_position(params, theta2, branch)
        if state.valid:
            states.append({
                "theta2": float(np.degrees(state.theta2)),
                "theta3": float(np.degrees(state.theta3)),
                "theta4": float(np.degrees(state.theta4)),
                "B": list(state.B),
                "C": list(state.C),
                "P": list(state.P),
                "mu": float(np.degrees(state.mu)),
            })
            trajectory.append(list(state.P))
            transmission_angles.append(float(np.degrees(state.mu)))
            if state.mu < min_mu:
                min_mu = state.mu
                min_mu_angle = theta2
        else:
            states.append(None)

    valid_count = sum(1 for s in states if s is not None)

    return {
        "states": states,
        "trajectory": trajectory,
        "transmission_angles": transmission_angles,
        "min_transmission_angle": float(np.degrees(min_mu)),
        "min_transmission_angle_at": float(np.degrees(min_mu_angle)),
        "full_rotation_possible": valid_count == steps,
        "valid_count": valid_count,
        "total_steps": steps,
    }


def find_dead_points(params: LinkageParams, steps: int = 3600) -> list:
    """Find dead points for all mechanism types.

    For crank-rocker: dead points occur when crank and coupler are collinear
    (A, B, C collinear). At these positions the crank cannot be driven by
    the follower. Two cases:
      - Extended: theta2 where AB and BC point same direction (a+b)
      - Folded: theta2 where AB and BC point opposite directions (|a-b|)

    For non-crank mechanisms: assembly limits and near-zero transmission angle.
    """
    a, b, c, d = params.a, params.b, params.c, params.d
    dead_points = []
    angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)

    # === Crank-coupler collinearity (crank-rocker dead points) ===
    # When A,B,C are collinear, crank+coupler form a straight line.
    # This happens at specific theta2 values found by checking the
    # cross product of AB and BC vectors for sign changes.
    prev_state = solve_position(params, angles[-1])
    if prev_state.valid:
        prev_cross_abc = (prev_state.B[0] * (prev_state.C[1] - prev_state.B[1]) -
                          prev_state.B[1] * (prev_state.C[0] - prev_state.B[0]))
    else:
        prev_cross_abc = None

    for theta2 in angles:
        state = solve_position(params, theta2)
        if not state.valid:
            prev_cross_abc = None
            prev_state = state
            continue

        # Cross product of vectors AB and BC (A is origin)
        # AB = B, BC = C - B
        ab = state.B
        bc = (state.C[0] - state.B[0], state.C[1] - state.B[1])
        cross_abc = ab[0] * bc[1] - ab[1] * bc[0]

        if prev_cross_abc is not None and cross_abc * prev_cross_abc < 0:
            # Sign change => A,B,C just crossed collinearity
            # Determine if extended (same direction) or folded
            dot = ab[0] * bc[0] + ab[1] * bc[1]
            if dot > 0:
                config = "extended"
                desc = "曲柄与连杆伸直共线（死点）"
                detail = f"AB+BC同向，从动件无法驱动曲柄"
            else:
                config = "folded"
                desc = "曲柄与连杆折叠共线（死点）"
                detail = f"AB+BC反向，从动件无法驱动曲柄"

            dead_points.append({
                "theta2_deg": float(np.degrees(theta2)),
                "type": "crank_coupler_collinear",
                "config": config,
                "description": desc,
                "detail": detail,
                "mu_deg": float(np.degrees(state.mu)),
                "B": list(state.B),
                "C": list(state.C),
            })

        prev_cross_abc = cross_abc
        prev_state = state

    # === Transmission angle minima (near-dead for any mechanism) ===
    mu_values = []
    valid_angles = []
    for theta2 in angles:
        state = solve_position(params, theta2)
        if state.valid:
            mu_values.append(state.mu)
            valid_angles.append(theta2)

    if len(mu_values) > 2:
        mu_arr = np.array(mu_values)
        angle_arr = np.array(valid_angles)
        threshold = np.radians(10)

        min_indices = []
        for i in range(1, len(mu_arr) - 1):
            if mu_arr[i] < mu_arr[i-1] and mu_arr[i] < mu_arr[i+1] and mu_arr[i] < threshold:
                min_indices.append(i)

        global_min_idx = int(np.argmin(mu_arr))
        if mu_arr[global_min_idx] < threshold and global_min_idx not in min_indices:
            min_indices.append(global_min_idx)

        for idx in min_indices:
            t2 = angle_arr[idx]
            # Avoid duplicating crank-coupler dead points
            already = any(abs(dp["theta2_deg"] - np.degrees(t2)) < 2.0 for dp in dead_points)
            if already:
                continue
            state = solve_position(params, t2)
            if not state.valid:
                continue
            mu_deg = float(np.degrees(mu_arr[idx]))
            dead_points.append({
                "theta2_deg": float(np.degrees(t2)),
                "type": "near_dead_point",
                "description": f"传动角极小({mu_deg:.1f}°)，接近死点",
                "detail": "传动角过小，力传递效率极低",
                "mu_deg": mu_deg,
                "B": list(state.B),
                "C": list(state.C),
            })

    # === Assembly boundary (non-crank mechanisms) ===
    prev_valid = solve_position(params, angles[-1]).valid
    for theta2 in angles:
        state = solve_position(params, theta2)
        if prev_valid and not state.valid:
            boundary_state = solve_position(params, theta2 - 2*np.pi/steps)
            if boundary_state.valid:
                already = any(abs(dp["theta2_deg"] - np.degrees(theta2)) < 2.0 for dp in dead_points)
                if not already:
                    dead_points.append({
                        "theta2_deg": float(np.degrees(theta2)),
                        "type": "assembly_limit",
                        "description": "装配极限位置（无法继续运动）",
                        "detail": "杆长无法满足封闭条件",
                        "mu_deg": 0.0,
                        "B": list(boundary_state.B),
                        "C": list(boundary_state.C),
                    })
        prev_valid = state.valid

    return dead_points


def find_extreme_positions(params: LinkageParams, steps: int = 3600) -> dict:
    """Find extreme positions (极位) of the follower and compute the swing angle."""
    angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)

    theta4_values = []
    valid_angles = []

    for theta2 in angles:
        state = solve_position(params, theta2)
        if state.valid:
            theta4_values.append(state.theta4)
            valid_angles.append(theta2)

    if len(theta4_values) < 2:
        return {"extreme_positions": [], "swing_angle": 0}

    theta4_arr = np.array(theta4_values)
    max_idx = np.argmax(theta4_arr)
    min_idx = np.argmin(theta4_arr)

    theta4_max = theta4_arr[max_idx]
    theta4_min = theta4_arr[min_idx]
    swing_angle = theta4_max - theta4_min

    # Extreme position angles (crank angles at extreme follower positions)
    theta2_at_max = valid_angles[max_idx]
    theta2_at_min = valid_angles[min_idx]

    # Compute acute angle between crank positions at extremes (极位夹角)
    angle_diff = abs(theta2_at_max - theta2_at_min)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff

    state_max = solve_position(params, theta2_at_max)
    state_min = solve_position(params, theta2_at_min)

    return {
        "extreme_positions": [
            {
                "type": "max",
                "theta2_deg": float(np.degrees(theta2_at_max)),
                "theta4_deg": float(np.degrees(theta4_max)),
                "B": list(state_max.B),
                "C": list(state_max.C),
            },
            {
                "type": "min",
                "theta2_deg": float(np.degrees(theta2_at_min)),
                "theta4_deg": float(np.degrees(theta4_min)),
                "B": list(state_min.B),
                "C": list(state_min.C),
            },
        ],
        "swing_angle_deg": float(np.degrees(swing_angle)),
        "extreme_angle_deg": float(np.degrees(angle_diff)),
        "travel_ratio": float((np.pi + angle_diff) / (np.pi - angle_diff)) if angle_diff < np.pi else float('inf'),
    }


def synthesize_coupler_curve(target_points: list, num_bars: int = 4, iterations: int = 500) -> dict:
    """Optimize bar lengths to approximate a target coupler curve.

    Two-phase approach for robust convergence:
      Phase 1: Latin hypercube sampling to find good starting regions
      Phase 2: Nelder-Mead refinement from best candidates

    Uses Procrustes-aligned Chamfer distance (rotation+scale+translation invariant).
    """
    from scipy.optimize import minimize

    target = np.array(target_points)
    n_target = len(target)

    # Pre-compute normalized target
    t_center = target.mean(axis=0)
    t_centered = target - t_center
    t_scale = np.sqrt(np.sum(t_centered ** 2) / n_target)
    if t_scale < 1e-8:
        t_scale = 1.0
    t_norm = t_centered / t_scale

    def procrustes_align(gen_pts):
        """Align generated points to target using Procrustes (translation+scale+rotation)."""
        g = np.array(gen_pts)
        n = len(g)
        g_center = g.mean(axis=0)
        g_centered = g - g_center
        g_scale = np.sqrt(np.sum(g_centered ** 2) / n)
        if g_scale < 1e-8:
            return g_centered
        g_norm = g_centered / g_scale

        # Find optimal rotation via SVD
        H = g_norm.T @ t_norm[:n]
        U, S, Vt = np.linalg.svd(H)
        R = (Vt.T @ U.T)
        # Apply rotation
        aligned = (g_norm @ R.T)
        return aligned

    def chamfer_distance(gen_aligned, target_norm_sub):
        """Bidirectional Chamfer distance between aligned point sets."""
        total = 0.0
        # gen -> target
        for p in gen_aligned:
            total += np.min(np.sum((target_norm_sub - p) ** 2, axis=1))
        # target -> gen
        for p in target_norm_sub:
            total += np.min(np.sum((gen_aligned - p) ** 2, axis=1))
        return total / (2 * len(gen_aligned))

    def objective(x):
        a, b, c, d_len, px, py, angle_offset = x
        if a <= 0.1 or b <= 0.1 or c <= 0.1 or d_len <= 0.1:
            return 1e10

        params = LinkageParams(a=a, b=b, c=c, d=d_len, coupler_x=px, coupler_y=py)

        # Generate coupler points at n_target evenly spaced angles
        angles = np.linspace(0, 2 * np.pi, n_target, endpoint=False) + angle_offset
        generated = []
        for th in angles:
            state = solve_position(params, th)
            if state.valid:
                generated.append([state.P[0], state.P[1]])
            else:
                return 1e8

        if len(generated) < n_target * 0.8:
            return 1e8

        # Procrustes alignment then Chamfer distance
        try:
            aligned = procrustes_align(generated)
            return chamfer_distance(aligned, t_norm[:len(generated)])
        except Exception:
            return 1e8

    # Phase 1: Latin hypercube sampling for diverse starting points
    rng = np.random.default_rng(42)
    n_lhs = 12

    # Stratified random samples across the parameter space
    starts = []
    for i in range(n_lhs):
        a = rng.uniform(0.3, 3.5)
        b = rng.uniform(1.0, 6.0)
        c = rng.uniform(0.8, 5.0)
        d_len = rng.uniform(1.0, 7.0)
        px = rng.uniform(-1, b * 0.8)
        py = rng.uniform(-1, b * 0.5)
        angle_off = i * (2 * np.pi / n_lhs)
        starts.append([a, b, c, d_len, px, py, angle_off])

    # Add hand-tuned starts for common curve types
    starts.extend([
        [1.0, 3.0, 2.5, 4.0, 1.5, 0.8, 0.0],
        [1.5, 2.5, 2.0, 3.5, 1.0, 1.0, np.pi / 4],
        [2.0, 4.0, 3.0, 5.0, 2.0, 1.5, np.pi / 2],
        [0.8, 2.0, 1.8, 3.0, 1.2, 0.5, np.pi],
        [1.2, 4.5, 3.5, 5.5, 2.5, 1.0, 3 * np.pi / 4],
        [0.5, 2.0, 1.5, 2.5, 1.0, 0.3, np.pi / 6],
        [2.5, 5.0, 4.0, 6.0, 3.0, 2.0, 5 * np.pi / 4],
        [1.0, 1.5, 1.2, 2.0, 0.8, 0.4, np.pi / 3],
    ])

    # Phase 1: Quick evaluation of all starts (low iteration limit)
    scored = []
    for x0 in starts:
        try:
            result = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': iterations // 4, 'xatol': 1e-3, 'fatol': 1e-4}
            )
            scored.append((result.fun, result.x))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0])

    # Phase 2: Refine top candidates with more iterations
    best_fun = 1e20
    best_x = None
    top_k = min(3, len(scored))

    for _, x0 in scored[:top_k]:
        try:
            result = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': iterations, 'xatol': 1e-5, 'fatol': 1e-6}
            )
            if result.fun < best_fun:
                best_fun = result.fun
                best_x = result.x
        except Exception:
            continue

    if best_x is None:
        return {"error": 1e10, "success": False, "params": {}, "trajectory": [], "grashof": {}}

    a, b, c, d_len, px, py, angle_offset = best_x
    a, b, c, d_len = abs(a), abs(b), abs(c), abs(d_len)

    params = LinkageParams(a=a, b=b, c=c, d=d_len, coupler_x=px, coupler_y=py)
    rotation = compute_full_rotation(params, steps=360)

    return {
        "params": {
            "a": float(a), "b": float(b), "c": float(c), "d": float(d_len),
            "coupler_x": float(px), "coupler_y": float(py),
        },
        "angle_offset_deg": float(np.degrees(angle_offset % (2 * np.pi))),
        "error": float(best_fun),
        "success": bool(best_fun < 0.1),
        "trajectory": rotation["trajectory"],
        "grashof": grashof_condition(a, b, c, d_len),
    }
