import os
import numpy as np
import jax.numpy as jnp
import dynamiqs as dq


# Paper-aligned binomial code states used in this project.
# PRX 2022 binomial example target: (sqrt(3)|3> + |9>) / 2.
_BINOMIAL_CODE_SPECS = {
    "s1_plus": ((0, 1.0 / np.sqrt(2.0)), (4, 1.0 / np.sqrt(2.0))),
    "s2_plus": ((0, 0.5), (6, np.sqrt(3.0) / 2.0)),
    # Distance-3 binomial codeword used in PRX 2022 and PRL 2024:
    # (sqrt(3)|3> + |9>) / 2.
    "d3_z": ((3, np.sqrt(3.0) / 2.0), (9, 0.5)),
}

_BINOMIAL_ALIASES = {
    "binomial": "d3_z",
    "paper": "d3_z",
    "prx": "d3_z",
    "distance3": "d3_z",
    "d3": "d3_z",
    "d3_plus": "d3_z",
    "d3_minus": "d3_z",
    "s2_z": "d3_z",
    "z_s2": "d3_z",
    "s2_minus": "d3_z",
    "s2plus": "s2_plus",
    "plus_s2": "s2_plus",
    "+z_s2": "s2_plus",
    "s1plus": "s1_plus",
    "plus_s1": "s1_plus",
    "+z_s1": "s1_plus",
}


def _resolve_binomial_code(code):
    if code is None:
        return "d3_z"
    key = str(code).strip().lower()
    if key in _BINOMIAL_CODE_SPECS:
        return key
    if key in _BINOMIAL_ALIASES:
        return _BINOMIAL_ALIASES[key]
    raise ValueError(
        f"Unsupported binomial_code={code}. Supported: {sorted(_BINOMIAL_CODE_SPECS.keys())}"
    )


def _binomial_state(binomial_code, n_boson, rel_phase=None):
    code = _resolve_binomial_code(binomial_code)
    components = list(_BINOMIAL_CODE_SPECS[code])
    max_n = max(n for n, _ in components)
    if max_n >= int(n_boson):
        raise ValueError(
            f"n_boson={n_boson} is too small for {code}; requires at least {max_n + 1}."
        )

    if rel_phase is not None and len(components) >= 2:
        n_last, coeff_last = components[-1]
        components[-1] = (n_last, coeff_last * np.exp(1j * float(rel_phase)))

    psi = None
    for n_fock, coeff in components:
        basis = dq.fock(n_boson, int(n_fock))
        term = coeff * basis
        psi = term if psi is None else psi + term
    return psi.unit()


def _binomial_support_fock_numbers(binomial_code):
    code = _resolve_binomial_code(binomial_code)
    return [int(n) for n, _ in _BINOMIAL_CODE_SPECS[code]]


def binomial_target_fock_statistics(binomial_code, n_boson, rel_phase=None):
    target = _binomial_state(binomial_code, n_boson, rel_phase=rel_phase)
    amp = np.asarray(target).reshape(-1)
    probs = np.abs(amp) ** 2
    probs = probs / float(np.sum(probs))
    n_axis = np.arange(n_boson, dtype=float)
    mean_n = float(np.sum(n_axis * probs))
    tail_start = int(max(0, np.floor(0.9 * n_boson)))
    tail_mass = float(np.sum(probs[tail_start:]))
    edge_prob = float(probs[-1])
    return {
        "mean_n": mean_n,
        "tail_start": tail_start,
        "tail_mass": tail_mass,
        "edge_prob": edge_prob,
    }


def _parity_operator(n_boson):
    return dq.parity(n_boson)


def _sample_points(grid_size, extent):
    axis = np.linspace(-extent, extent, grid_size)
    return [x + 1j * y for x in axis for y in axis]


def _binomial_focus_points(binomial_code):
    support = _binomial_support_fock_numbers(binomial_code)
    radii = sorted(float(np.sqrt(max(n, 0))) for n in support)
    points = [0.0 + 0.0j]
    for r in radii:
        points.extend(
            [
                r + 0.0j,
                -r + 0.0j,
                0.0 + 1j * r,
                0.0 - 1j * r,
                0.5 * r + 0.0j,
                -0.5 * r + 0.0j,
                0.0 + 0.5j * r,
                0.0 - 0.5j * r,
            ]
        )
    if len(radii) >= 2:
        r_mid = 0.5 * (radii[0] + radii[-1])
        points.extend([r_mid + 0.0j, -r_mid + 0.0j, 0.0 + 1j * r_mid, 0.0 - 1j * r_mid])

    unique = []
    seen = set()
    for z in points:
        key = (round(float(np.real(z)), 12), round(float(np.imag(z)), 12))
        if key in seen:
            continue
        seen.add(key)
        unique.append(complex(z))
    return unique


def _random_points(count, extent, rng):
    xs = rng.uniform(-extent, extent, size=count)
    ys = rng.uniform(-extent, extent, size=count)
    return [x + 1j * y for x, y in zip(xs, ys)]


def _wigner_at_point(rho, alpha, parity_op):
    n_boson = rho.shape[-1]
    disp = dq.displace(n_boson, alpha)
    displaced = disp @ rho @ disp.dag()
    return (2.0 / np.pi) * dq.expect(parity_op, displaced).real


def _target_wigner_values(target_rho, sample_points, parity_op):
    return np.array(
        [_wigner_at_point(target_rho, alpha, parity_op) for alpha in sample_points],
        dtype=float,
    )


def _sample_parity(parity_expect, n_shots, rng):
    if n_shots <= 0:
        return float(np.clip(parity_expect, -1.0, 1.0))
    p_plus = 0.5 * (1.0 + np.clip(parity_expect, -1.0, 1.0))
    if n_shots <= 1:
        return 1.0 if rng.random() < p_plus else -1.0
    shots = rng.random(n_shots) < p_plus
    return 2.0 * shots.mean() - 1.0


def _characteristic_at_point(rho, alpha):
    n_boson = rho.shape[-1]
    disp = dq.displace(n_boson, alpha)
    return dq.expect(disp, rho)


def _characteristic_at_point_full(psi_full, alpha, n_boson):
    disp = dq.displace(n_boson, alpha)
    disp_full = dq.tensor(dq.eye(2), disp)
    return dq.expect(disp_full, psi_full)


def _full_disp_ops(n_boson, alphas):
    disp_ops = dq.displace(n_boson, alphas)
    return dq.tensor(dq.eye(2), disp_ops)


def _target_characteristic_values(target_rho, sample_points):
    return np.array(
        [_characteristic_at_point(target_rho, alpha) for alpha in sample_points],
        dtype=complex,
    )


def characteristic_norm(target_values, sample_area):
    norm = float((sample_area / np.pi) * np.sum(np.abs(target_values) ** 2))
    if not np.isfinite(norm) or norm <= 0.0:
        return 1.0
    return norm


def prepare_characteristic_distribution(
    n_boson,
    extent,
    grid_size,
    binomial_code="d3_z",
    mix_uniform=0.0,
    alpha_scale=1.0,
    binomial_phase=None,
    importance_power=1.0,
):
    axis = np.linspace(-extent, extent, grid_size)
    if grid_size > 1:
        delta = float(axis[1] - axis[0])
    else:
        delta = float(2.0 * extent)
    scaled_delta = delta * float(alpha_scale)
    area_element = scaled_delta * scaled_delta
    target = _binomial_state(
        binomial_code=binomial_code,
        n_boson=n_boson,
        rel_phase=binomial_phase,
    )
    target_rho = target @ target.dag()
    points = [alpha_scale * (x + 1j * y) for x in axis for y in axis]
    chi_target = _target_characteristic_values(target_rho, points)
    importance_power = float(importance_power)
    if not np.isfinite(importance_power) or importance_power <= 0.0:
        raise ValueError(
            f"importance_power must be > 0 and finite, got {importance_power}"
        )
    weights = np.abs(chi_target) ** importance_power
    if mix_uniform > 0.0:
        weights = (1.0 - mix_uniform) * weights + mix_uniform * np.ones_like(weights)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        weights = np.ones_like(weights) / weights.size
    else:
        weights = weights / total
    return points, chi_target, weights, area_element


def _broadcast_time_array(value, like, name):
    arr = jnp.asarray(value, dtype=jnp.float32)
    if arr.ndim == 0:
        return jnp.full_like(like, float(arr))
    if arr.ndim == 1:
        if arr.shape[0] != like.shape[-1]:
            raise ValueError(f"{name} must be scalar or length {like.shape[-1]}")
        return jnp.broadcast_to(arr, like.shape)
    if arr.shape != like.shape:
        raise ValueError(f"{name} must be scalar or shape {like.shape}")
    return arr


def _broadcast_motional_detuning(value, like, name):
    arr = jnp.asarray(value, dtype=jnp.float32)
    if arr.ndim == 0:
        return jnp.full_like(like, float(arr))
    if arr.ndim == 1:
        match_steps = arr.shape[0] == like.shape[-1]
        match_batch = arr.shape[0] == like.shape[0]
        if match_steps and match_batch:
            raise ValueError(
                f"{name} length is ambiguous when batch={like.shape[0]} equals n_steps={like.shape[-1]}. "
                "Use an explicit 2D array with shape (batch, n_steps)."
            )
        if match_steps:
            return jnp.broadcast_to(arr[None, :], like.shape)
        if match_batch:
            return jnp.broadcast_to(arr[:, None], like.shape)
        raise ValueError(
            f"{name} must be scalar, length {like.shape[0]}, "
            f"length {like.shape[-1]}, or shape {like.shape}"
        )
    if arr.shape != like.shape:
        raise ValueError(
            f"{name} must be scalar, length {like.shape[0]}, "
            f"length {like.shape[-1]}, or shape {like.shape}"
        )
    return arr


_OP_CACHE = {}


def _get_ops(n_boson):
    ops = _OP_CACHE.get(n_boson)
    if ops is not None:
        return ops
    eye2 = dq.eye(2)
    eye_b = dq.eye(n_boson)
    a = dq.destroy(n_boson)
    a_dag = a.dag()
    sigma_p = dq.sigmap()
    sigma_m = dq.sigmam()
    a_full = dq.tensor(eye2, a)
    a_dag_full = dq.tensor(eye2, a_dag)
    sigma_p_full = dq.tensor(sigma_p, eye_b)
    sigma_m_full = dq.tensor(sigma_m, eye_b)
    n_full = dq.tensor(eye2, a_dag @ a)
    ops = (a_full, a_dag_full, sigma_p_full, sigma_m_full, n_full)
    _OP_CACHE[n_boson] = ops
    return ops


def simulate_boson_state(
    phi_r,
    phi_b,
    n_boson,
    omega_r,
    omega_b,
    t_step,
    n_times=None,
    return_density=False,
    motional_detuning=0.0,
):
    phi_r = jnp.asarray(phi_r, dtype=jnp.float32)
    phi_b = jnp.asarray(phi_b, dtype=jnp.float32)
    if phi_r.shape != phi_b.shape:
        raise ValueError("phi_r and phi_b must have the same shape.")
    if phi_r.ndim == 1:
        psi_final = _simulate_boson_state_batch(
            phi_r[jnp.newaxis, :],
            phi_b[jnp.newaxis, :],
            n_boson,
            omega_r,
            omega_b,
            t_step,
            n_times=n_times,
            motional_detuning=motional_detuning,
        )
        if return_density:
            rho_boson, rho_qubit = _ptrace_boson_qubit(psi_final, n_boson)
            return psi_final[0], rho_boson[0], rho_qubit[0]
        return psi_final[0]
    psi_final = _simulate_boson_state_batch(
        phi_r,
        phi_b,
        n_boson,
        omega_r,
        omega_b,
        t_step,
        n_times=n_times,
        motional_detuning=motional_detuning,
    )
    if return_density:
        rho_boson, rho_qubit = _ptrace_boson_qubit(psi_final, n_boson)
        return psi_final, rho_boson, rho_qubit
    return psi_final


def _simulate_boson_state_batch(
    phi_r,
    phi_b,
    n_boson,
    omega_r,
    omega_b,
    t_step,
    n_times=None,
    motional_detuning=0.0,
):
    phi_r = jnp.asarray(phi_r, dtype=jnp.float32)
    phi_b = jnp.asarray(phi_b, dtype=jnp.float32)
    if phi_r.shape != phi_b.shape:
        raise ValueError("phi_r and phi_b must have the same shape.")

    _, n_steps = phi_r.shape
    t_duration = float(n_steps * t_step)
    t_edges = jnp.linspace(0.0, t_duration, n_steps + 1)

    omega_r = _broadcast_time_array(omega_r, phi_r, "omega_r")
    omega_b = _broadcast_time_array(omega_b, phi_b, "omega_b")
    detuning = _broadcast_motional_detuning(
        motional_detuning,
        phi_r,
        "motional_detuning",
    )
    coeff_r = 0.5 * omega_r * jnp.exp(1j * phi_r)
    coeff_b = 0.5 * omega_b * jnp.exp(1j * phi_b)

    a, a_dag, sigma_p, sigma_m, n_op = _get_ops(n_boson)
    H_r_up = dq.pwc(t_edges, coeff_r, sigma_p @ a)
    H_r_down = dq.pwc(t_edges, jnp.conj(coeff_r), sigma_m @ a_dag)
    H_b_up = dq.pwc(t_edges, coeff_b, sigma_p @ a_dag)
    H_b_down = dq.pwc(t_edges, jnp.conj(coeff_b), sigma_m @ a)
    H_det = dq.pwc(t_edges, detuning, n_op)
    H = H_r_up + H_r_down + H_b_up + H_b_down + H_det

    psi0 = dq.tensor(dq.basis(2, 0), dq.fock(n_boson, 0))
    tsave = jnp.array([t_duration], dtype=jnp.float32)
    options = dq.Options(save_states=False, progress_meter=False, t0=0.0)
    result = dq.sesolve(H, psi0, tsave=tsave, options=options)
    psi_final = result.final_state
    return psi_final


def _ptrace_boson_qubit(psi_final, n_boson):
    rho_boson = dq.ptrace(psi_final, 1, dims=(2, n_boson))
    rho_qubit = dq.ptrace(psi_final, 0, dims=(2, n_boson))
    return rho_boson, rho_qubit


def wigner_grid(rho, xvec, yvec):
    parity_op = _parity_operator(rho.shape[-1])
    vals = np.zeros((len(yvec), len(xvec)), dtype=float)
    for yi, y in enumerate(yvec):
        for xi, x in enumerate(xvec):
            beta = x + 1j * y
            vals[yi, xi] = float(_wigner_at_point(rho, beta, parity_op))
    return vals


def characteristic_grid(rho, xvec, yvec):
    vals = np.zeros((len(yvec), len(xvec)))
    for yi, y in enumerate(yvec):
        for xi, x in enumerate(xvec):
            beta = x + 1j * y
            vals[yi, xi] = float(_characteristic_at_point(rho, beta).real)
    return vals


def select_wigner_points(
    n_boson,
    extent,
    grid_size,
    top_k,
    binomial_code="d3_z",
    binomial_phase=None,
):
    """
    Select a fixed set of phase-space points where |W_target| is largest.
    """
    axis = np.linspace(-extent, extent, grid_size)
    target = _binomial_state(binomial_code, n_boson, rel_phase=binomial_phase)
    w_target = wigner_grid(target @ target.dag(), axis, axis)
    flat = np.abs(w_target).ravel()
    if top_k >= flat.size:
        top_idx = np.argsort(flat)[::-1]
    else:
        top_idx = np.argpartition(flat, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

    points = []
    for idx in top_idx:
        row = idx // grid_size
        col = idx % grid_size
        points.append(axis[col] + 1j * axis[row])
    return points


def trapped_ion_binomial_sim(
    phi_r,
    phi_b,
    amp_r=None,
    amp_b=None,
    n_boson=20,
    omega=2 * np.pi * 2000.0,
    t_step=1.0e-5,
    n_times=None,
    binomial_code="d3_z",
    binomial_phase=None,
    sample_mode="binomial",
    sample_grid=5,
    sample_extent=2.5,
    n_sample_points=30,
    sample_points=None,
    target_values=None,
    sample_weights=None,
    sample_area=None,
    reward_scale=1.0,
    reward_clip=None,
    n_shots=0,
    seed=None,
    return_details=False,
    reward_mode="characteristic",
    characteristic_objective="overlap_real",
    reward_norm=None,
    return_density=False,
    motional_detuning=0.0,
):
    """
    Simulate trapped-ion state preparation with RSB/BSB controls and return
    a measurement-based reward derived from sampled characteristic values.
    `motional_detuning` adds a quasi-static (or time-varying) term
    delta(t) * a^\dagger a on the motional mode.
    """
    rng = np.random.default_rng(seed)

    if amp_r is None:
        omega_r = omega
    else:
        omega_r = omega * np.asarray(amp_r, dtype=float)
    if amp_b is None:
        omega_b = omega
    else:
        omega_b = omega * np.asarray(amp_b, dtype=float)

    need_rho = return_density or reward_mode != "characteristic"
    if need_rho:
        psi_full, rho_boson, rho_qubit = simulate_boson_state(
            phi_r,
            phi_b,
            n_boson=n_boson,
            omega_r=omega_r,
            omega_b=omega_b,
            t_step=t_step,
            n_times=n_times,
            return_density=True,
            motional_detuning=motional_detuning,
        )
    else:
        psi_full = simulate_boson_state(
            phi_r,
            phi_b,
            n_boson=n_boson,
            omega_r=omega_r,
            omega_b=omega_b,
            t_step=t_step,
            n_times=n_times,
            return_density=False,
            motional_detuning=motional_detuning,
        )
        rho_boson = None

    target = _binomial_state(
        binomial_code=binomial_code,
        n_boson=n_boson,
        rel_phase=binomial_phase,
    )
    target_rho = target @ target.dag()

    if sample_points is None:
        if sample_mode in ("binomial", "focus"):
            sample_points = _binomial_focus_points(binomial_code)
        elif sample_mode == "random":
            sample_points = _random_points(n_sample_points, sample_extent, rng)
        else:
            sample_points = _sample_points(sample_grid, sample_extent)
        target_values = None
        sample_weights = None
    else:
        sample_points = list(sample_points)

    if reward_mode == "characteristic":
        if target_values is None:
            target_values = _target_characteristic_values(target_rho, sample_points)

        if sample_weights is None:
            sample_weights = np.full(len(sample_points), 1.0 / len(sample_points))
        sample_weights = np.asarray(sample_weights, dtype=float)
        if sample_area is None:
            sample_area = 1.0

        meas = []
        for alpha in sample_points:
            chi_expect = _characteristic_at_point_full(psi_full, alpha, n_boson)
            if n_shots <= 0:
                meas.append(chi_expect)
            else:
                chi_real = float(np.clip(chi_expect.real, -1.0, 1.0))
                p_plus = 0.5 * (1.0 + chi_real)
                meas.append(1.0 if rng.random() < p_plus else -1.0)
        meas = np.array(meas, dtype=complex)
        denom = sample_weights
        overlap = np.mean(meas * np.conjugate(target_values) / denom)
        norm = np.mean((np.abs(target_values) ** 2) / denom)
        if reward_norm is not None:
            try:
                fixed_norm = float(np.asarray(reward_norm))
            except (TypeError, ValueError):
                fixed_norm = np.nan
            if np.isfinite(fixed_norm) and fixed_norm > 0.0:
                norm = fixed_norm
        if not np.isfinite(norm) or norm <= 0.0:
            norm = 1.0
        p = np.asarray(sample_weights, dtype=float)
        p = np.maximum(p, 1e-12)
        inv_p = 1.0 / p
        if characteristic_objective == "nmse":
            # Importance-corrected MSE: avoid double-counting when points are
            # already sampled from P(alpha).
            num = np.mean((np.abs(meas - target_values) ** 2) * inv_p)
            den = np.mean((np.abs(target_values) ** 2) * inv_p)
            if not np.isfinite(den) or den <= 0.0:
                den = 1.0
            reward = float(reward_scale * (1.0 - num / den))
        elif characteristic_objective == "nmse_exp":
            num = np.mean((np.abs(meas - target_values) ** 2) * inv_p)
            den = np.mean((np.abs(target_values) ** 2) * inv_p)
            if not np.isfinite(den) or den <= 0.0:
                den = 1.0
            reward = float(reward_scale * np.exp(-num / den))
        elif characteristic_objective == "overlap_abs":
            reward = float(reward_scale * np.abs(overlap) / norm)
        else:
            # Historical behavior: normalized real-part overlap.
            reward = float(reward_scale * overlap.real / norm)
        if reward_clip is not None:
            reward = float(np.clip(reward, -reward_clip, reward_clip))
    else:
        parity_op = _parity_operator(n_boson)
        if target_values is None:
            target_values = _target_wigner_values(target_rho, sample_points, parity_op)
        w_meas = []
        for alpha in sample_points:
            parity_expect = _wigner_at_point(rho_boson, alpha, parity_op) * (np.pi / 2.0)
            parity_sample = _sample_parity(parity_expect, n_shots, rng)
            w_meas.append((2.0 / np.pi) * parity_sample)
        w_meas = np.array(w_meas, dtype=float)
        denom = float(np.mean(target_values ** 2))
        if not np.isfinite(denom) or denom <= 0:
            denom = 1.0
        reward = float(np.mean(target_values * w_meas) / denom)
        if reward_clip is not None:
            reward = float(np.clip(reward, -reward_clip, reward_clip))

    if not return_details:
        return reward

    target_op = dq.tensor(dq.eye(2), target_rho)
    fidelity = float(dq.expect(target_op, psi_full).real)
    return reward, fidelity, rho_boson, target_rho


def trapped_ion_binomial_sim_batch(
    phi_r,
    phi_b,
    amp_r=None,
    amp_b=None,
    n_boson=20,
    omega=2 * np.pi * 2000.0,
    t_step=1.0e-5,
    n_times=None,
    binomial_code="d3_z",
    binomial_phase=None,
    sample_mode="binomial",
    sample_grid=5,
    sample_extent=2.5,
    n_sample_points=30,
    sample_points=None,
    target_values=None,
    sample_weights=None,
    sample_area=None,
    reward_scale=1.0,
    reward_clip=None,
    n_shots=0,
    seed=None,
    return_details=False,
    reward_mode="characteristic",
    characteristic_objective="overlap_real",
    reward_norm=None,
    return_density=False,
    motional_detuning=0.0,
):
    rng = np.random.default_rng(seed)

    phi_r = np.asarray(phi_r, dtype=float)
    phi_b = np.asarray(phi_b, dtype=float)
    if phi_r.shape != phi_b.shape:
        raise ValueError("phi_r and phi_b must have the same shape.")

    force_serial = os.environ.get("QCRL_SERIAL_SIM", "0") == "1"
    if force_serial and phi_r.ndim > 1 and phi_r.shape[0] > 1:
        def _serial_detuning_sample(motion_detune, index, n_batch, n_steps):
            arr = np.asarray(motion_detune, dtype=float)
            if arr.ndim == 0:
                return float(arr)
            if arr.ndim == 1:
                match_batch = arr.shape[0] == n_batch
                match_steps = arr.shape[0] == n_steps
                if match_batch and match_steps:
                    raise ValueError(
                        "motional_detuning length is ambiguous when "
                        f"batch={n_batch} equals n_steps={n_steps}. "
                        "Use an explicit 2D array with shape "
                        f"({n_batch}, {n_steps})."
                    )
                if match_batch:
                    return float(arr[index])
                if match_steps:
                    return arr.copy()
                raise ValueError(
                    "motional_detuning must be scalar, length "
                    f"{n_batch}, length {n_steps}, or shape ({n_batch}, {n_steps})"
                )
            if arr.shape == (n_batch, n_steps):
                return arr[index].copy()
            raise ValueError(
                "motional_detuning must be scalar, length "
                f"{n_batch}, length {n_steps}, or shape ({n_batch}, {n_steps})"
            )

        rewards = []
        fidelities = []
        rho_list = []
        target_rho = None
        n_batch, n_steps = phi_r.shape
        for ii in range(phi_r.shape[0]):
            seed_i = None if seed is None else int(rng.integers(1, 2**31 - 1))
            motional_detuning_i = _serial_detuning_sample(
                motional_detuning,
                ii,
                n_batch,
                n_steps,
            )
            if return_details:
                r_i, f_i, rho_i, target_rho = trapped_ion_binomial_sim(
                    phi_r[ii],
                    phi_b[ii],
                    amp_r=None if amp_r is None else np.asarray(amp_r, dtype=float)[ii],
                    amp_b=None if amp_b is None else np.asarray(amp_b, dtype=float)[ii],
                    n_boson=n_boson,
                    omega=omega,
                    t_step=t_step,
                    n_times=n_times,
                    binomial_code=binomial_code,
                    binomial_phase=binomial_phase,
                    sample_mode=sample_mode,
                    sample_grid=sample_grid,
                    sample_extent=sample_extent,
                    n_sample_points=n_sample_points,
                    sample_points=sample_points,
                    target_values=target_values,
                    sample_weights=sample_weights,
                    sample_area=sample_area,
                    reward_scale=reward_scale,
                    reward_clip=reward_clip,
                    n_shots=n_shots,
                    seed=seed_i,
                    return_details=True,
                    reward_mode=reward_mode,
                    characteristic_objective=characteristic_objective,
                    reward_norm=reward_norm,
                    return_density=return_density,
                    motional_detuning=motional_detuning_i,
                )
                rewards.append(r_i)
                fidelities.append(f_i)
                if return_density:
                    rho_list.append(rho_i)
            else:
                r_i = trapped_ion_binomial_sim(
                    phi_r[ii],
                    phi_b[ii],
                    amp_r=None if amp_r is None else np.asarray(amp_r, dtype=float)[ii],
                    amp_b=None if amp_b is None else np.asarray(amp_b, dtype=float)[ii],
                    n_boson=n_boson,
                    omega=omega,
                    t_step=t_step,
                    n_times=n_times,
                    binomial_code=binomial_code,
                    binomial_phase=binomial_phase,
                    sample_mode=sample_mode,
                    sample_grid=sample_grid,
                    sample_extent=sample_extent,
                    n_sample_points=n_sample_points,
                    sample_points=sample_points,
                    target_values=target_values,
                    sample_weights=sample_weights,
                    sample_area=sample_area,
                    reward_scale=reward_scale,
                    reward_clip=reward_clip,
                    n_shots=n_shots,
                    seed=seed_i,
                    return_details=False,
                    reward_mode=reward_mode,
                    characteristic_objective=characteristic_objective,
                    reward_norm=reward_norm,
                    return_density=return_density,
                    motional_detuning=motional_detuning_i,
                )
                rewards.append(r_i)

        reward = np.array(rewards, dtype=float)
        if not return_details:
            return reward

        fidelity = np.array(fidelities, dtype=float)
        rho_boson = np.stack(rho_list) if return_density and rho_list else None
        return reward, fidelity, rho_boson, target_rho

    if amp_r is None:
        omega_r = omega
    else:
        omega_r = omega * np.asarray(amp_r, dtype=float)
    if amp_b is None:
        omega_b = omega
    else:
        omega_b = omega * np.asarray(amp_b, dtype=float)

    need_rho = return_density or reward_mode != "characteristic"
    if need_rho:
        psi_full, rho_boson, rho_qubit = simulate_boson_state(
            phi_r,
            phi_b,
            n_boson=n_boson,
            omega_r=omega_r,
            omega_b=omega_b,
            t_step=t_step,
            n_times=n_times,
            return_density=True,
            motional_detuning=motional_detuning,
        )
    else:
        psi_full = simulate_boson_state(
            phi_r,
            phi_b,
            n_boson=n_boson,
            omega_r=omega_r,
            omega_b=omega_b,
            t_step=t_step,
            n_times=n_times,
            return_density=False,
            motional_detuning=motional_detuning,
        )
        rho_boson = None

    target = _binomial_state(
        binomial_code=binomial_code,
        n_boson=n_boson,
        rel_phase=binomial_phase,
    )
    target_rho = target @ target.dag()

    if sample_points is None:
        if sample_mode in ("binomial", "focus"):
            sample_points = _binomial_focus_points(binomial_code)
        elif sample_mode == "random":
            sample_points = _random_points(n_sample_points, sample_extent, rng)
        else:
            sample_points = _sample_points(sample_grid, sample_extent)
        target_values = None
        sample_weights = None
    else:
        sample_points = list(sample_points)

    if reward_mode == "characteristic":
        if target_values is None:
            target_values = _target_characteristic_values(target_rho, sample_points)
        if sample_weights is None:
            sample_weights = np.full(len(sample_points), 1.0 / len(sample_points))
        sample_weights = np.asarray(sample_weights, dtype=float)
        if sample_area is None:
            sample_area = 1.0

        alphas = jnp.asarray(np.array(sample_points))
        target_vals = jnp.asarray(target_values)
        weights = jnp.asarray(sample_weights)

        full_ops = _full_disp_ops(n_boson, alphas)
        meas = dq.expect(full_ops, psi_full)
        if meas.shape[0] == len(sample_points):
            meas = jnp.swapaxes(meas, 0, 1)

        if n_shots > 0:
            chi_real = np.clip(np.array(meas.real), -1.0, 1.0)
            p_plus = 0.5 * (1.0 + chi_real)
            meas = np.where(rng.random(size=p_plus.shape) < p_plus, 1.0, -1.0)
            meas = jnp.asarray(meas, dtype=jnp.complex64)

        overlap = jnp.mean(meas * jnp.conjugate(target_vals) / weights, axis=1)
        norm = jnp.mean((jnp.abs(target_vals) ** 2) / weights)
        if reward_norm is not None:
            try:
                fixed_norm = float(np.asarray(reward_norm))
            except (TypeError, ValueError):
                fixed_norm = np.nan
            if np.isfinite(fixed_norm) and fixed_norm > 0.0:
                norm = jnp.asarray(fixed_norm, dtype=norm.dtype)
        norm = jnp.where(jnp.isfinite(norm) & (norm > 0), norm, 1.0)
        p = jnp.maximum(weights, 1e-12)
        inv_p = 1.0 / p
        if characteristic_objective == "nmse":
            num = jnp.mean((jnp.abs(meas - target_vals) ** 2) * inv_p[None, :], axis=1)
            den = jnp.mean((jnp.abs(target_vals) ** 2) * inv_p)
            den = jnp.where(jnp.isfinite(den) & (den > 0), den, 1.0)
            reward = reward_scale * (1.0 - num / den)
        elif characteristic_objective == "nmse_exp":
            num = jnp.mean((jnp.abs(meas - target_vals) ** 2) * inv_p[None, :], axis=1)
            den = jnp.mean((jnp.abs(target_vals) ** 2) * inv_p)
            den = jnp.where(jnp.isfinite(den) & (den > 0), den, 1.0)
            reward = reward_scale * jnp.exp(-num / den)
        elif characteristic_objective == "overlap_abs":
            reward = reward_scale * jnp.abs(overlap) / norm
        else:
            # Historical behavior: normalized real-part overlap.
            reward = reward_scale * overlap.real / norm
        if reward_clip is not None:
            reward = jnp.clip(reward, -reward_clip, reward_clip)
    else:
        parity_op = _parity_operator(n_boson)
        if target_values is None:
            target_values = _target_wigner_values(target_rho, sample_points, parity_op)
        target_values = np.asarray(target_values, dtype=float)
        denom = float(np.mean(target_values ** 2))
        if not np.isfinite(denom) or denom <= 0:
            denom = 1.0

        rewards = []
        for ii in range(phi_r.shape[0]):
            w_meas = []
            rho_i = rho_boson[ii]
            for alpha in sample_points:
                parity_expect = _wigner_at_point(rho_i, alpha, parity_op) * (np.pi / 2.0)
                parity_sample = _sample_parity(parity_expect, n_shots, rng)
                w_meas.append((2.0 / np.pi) * parity_sample)
            w_meas = np.array(w_meas, dtype=float)
            reward_i = np.mean(target_values * w_meas) / denom
            rewards.append(reward_i)
        reward = np.array(rewards, dtype=float)
        if reward_clip is not None:
            reward = np.clip(reward, -reward_clip, reward_clip)

    reward = np.array(reward, dtype=float)

    if not return_details:
        return reward

    target_op = dq.tensor(dq.eye(2), target_rho)
    fidelity = np.array(dq.expect(target_op, psi_full).real, dtype=float)
    return reward, fidelity, rho_boson, target_rho
