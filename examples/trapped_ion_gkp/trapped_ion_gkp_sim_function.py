import os
import numpy as np
import jax.numpy as jnp
import dynamiqs as dq


def _resolve_gkp_params(
    alpha_cat,
    cat_parity,
    gkp_delta=None,
    gkp_logical=None,
):
    """
    Backward-compatible parameter resolver.
    `alpha_cat/cat_parity` were inherited from cat-state code; GKP code should
    use `gkp_delta/gkp_logical`.
    """
    delta = float(alpha_cat if gkp_delta is None else gkp_delta)
    logical = cat_parity if gkp_logical is None else gkp_logical
    return delta, logical


def _gkp_basis_state(
    delta,
    n_boson,
    logical="0",
    squeeze_r=None,
    envelope_kappa=None,
    lattice_trunc=4,
):
    """
    Approximate finite-energy square GKP logical basis state in a truncated
    Fock space using a displaced squeezed-vacuum comb in q:

        |mu> ~ sum_s exp[-(kappa^2/2) q_s^2] D(q_s / sqrt(2)) S(r) |0>
        q_s = (2 s + mu) sqrt(pi), mu in {0, 1}

    For finite-energy symmetric approximate GKP, a common choice is kappa = delta
    and peak squeezing r = -log(delta). We use that mapping by default unless
    overridden.
    """
    logical_str = str(logical).lower()
    if logical_str in ("0", "zero", "even"):
        offset = 0
    elif logical_str in ("1", "one", "odd"):
        offset = 1
    else:
        raise ValueError(f"Unsupported GKP basis logical state: {logical}")

    delta = float(delta)
    if delta <= 0.0:
        raise ValueError(f"GKP delta must be positive, got {delta}")
    if squeeze_r is None:
        # Matches the standard finite-energy scaling where peak width and
        # envelope are controlled by the same delta parameter.
        squeeze_r = float(-np.log(delta))
    if envelope_kappa is None:
        envelope_kappa = delta
    squeeze_r = float(squeeze_r)
    envelope_kappa = float(envelope_kappa)

    base = dq.squeeze(n_boson, squeeze_r) @ dq.fock(n_boson, 0)
    psi = None
    for k in range(-int(lattice_trunc), int(lattice_trunc) + 1):
        q_shift = (2 * k + offset) * np.sqrt(np.pi)
        alpha_shift = q_shift / np.sqrt(2.0)
        env = np.exp(-0.5 * (envelope_kappa ** 2) * (q_shift ** 2))
        peak = dq.displace(n_boson, alpha_shift) @ base
        psi = env * peak if psi is None else psi + env * peak
    return psi.unit()


def _gkp_state(
    delta,
    n_boson,
    logical="0",
    rel_phase=None,
    squeeze_r=None,
    envelope_kappa=None,
    lattice_trunc=4,
):
    logical_str = str(logical).lower()
    if logical_str in ("0", "1", "zero", "one", "even", "odd"):
        return _gkp_basis_state(
            delta=delta,
            n_boson=n_boson,
            logical=logical_str,
            squeeze_r=squeeze_r,
            envelope_kappa=envelope_kappa,
            lattice_trunc=lattice_trunc,
        )

    if rel_phase is None:
        rel_phase = 0.0 if logical_str in ("plus", "+") else np.pi

    psi0 = _gkp_basis_state(
        delta=delta,
        n_boson=n_boson,
        logical="0",
        squeeze_r=squeeze_r,
        envelope_kappa=envelope_kappa,
        lattice_trunc=lattice_trunc,
    )
    psi1 = _gkp_basis_state(
        delta=delta,
        n_boson=n_boson,
        logical="1",
        squeeze_r=squeeze_r,
        envelope_kappa=envelope_kappa,
        lattice_trunc=lattice_trunc,
    )
    return (psi0 + np.exp(1j * rel_phase) * psi1).unit()


def _parity_operator(n_boson):
    return dq.parity(n_boson)


def _sample_points(grid_size, extent):
    axis = np.linspace(-extent, extent, grid_size)
    return [x + 1j * y for x in axis for y in axis]


def _gkp_focus_points():
    # Square-GKP characteristic-function reciprocal lattice spacing.
    # In our alpha convention, a = sqrt(pi / 2).
    a = np.sqrt(np.pi / 2.0)
    points = [0.0 + 0.0j]
    for m in (1, 2, 3):
        points += [m * a + 0.0j, -m * a + 0.0j, 0.0 + 1j * m * a, 0.0 - 1j * m * a]
    for m in (1, 2):
        for n in (1, 2):
            points += [
                (m * a) + 1j * (n * a),
                (m * a) - 1j * (n * a),
                (-m * a) + 1j * (n * a),
                (-m * a) - 1j * (n * a),
            ]
    # Keep unique values while preserving order.
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
    alpha_cat,
    n_boson,
    extent,
    grid_size,
    cat_parity="0",
    gkp_delta=None,
    gkp_logical=None,
    mix_uniform=0.0,
    alpha_scale=1.0,
    cat_phase=None,
    gkp_squeeze_r=None,
    gkp_envelope_kappa=None,
    gkp_lattice_trunc=4,
):
    gkp_delta, gkp_logical = _resolve_gkp_params(
        alpha_cat=alpha_cat,
        cat_parity=cat_parity,
        gkp_delta=gkp_delta,
        gkp_logical=gkp_logical,
    )
    axis = np.linspace(-extent, extent, grid_size)
    if grid_size > 1:
        delta = float(axis[1] - axis[0])
    else:
        delta = float(2.0 * extent)
    scaled_delta = delta * float(alpha_scale)
    area_element = scaled_delta * scaled_delta
    target = _gkp_state(
        delta=gkp_delta,
        n_boson=n_boson,
        logical=gkp_logical,
        rel_phase=cat_phase,
        squeeze_r=gkp_squeeze_r,
        envelope_kappa=gkp_envelope_kappa,
        lattice_trunc=gkp_lattice_trunc,
    )
    target_rho = target @ target.dag()
    points = [alpha_scale * (x + 1j * y) for x in axis for y in axis]
    chi_target = _target_characteristic_values(target_rho, points)
    weights = np.abs(chi_target)
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
    ops = (a_full, a_dag_full, sigma_p_full, sigma_m_full)
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
        )
        if return_density:
            rho_boson, rho_qubit = _ptrace_boson_qubit(psi_final, n_boson)
            return psi_final[0], rho_boson[0], rho_qubit[0]
        return psi_final[0]
    psi_final = _simulate_boson_state_batch(
        phi_r, phi_b, n_boson, omega_r, omega_b, t_step, n_times=n_times
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
    coeff_r = 0.5 * omega_r * jnp.exp(1j * phi_r)
    coeff_b = 0.5 * omega_b * jnp.exp(1j * phi_b)

    a, a_dag, sigma_p, sigma_m = _get_ops(n_boson)
    H_r_up = dq.pwc(t_edges, coeff_r, sigma_p @ a)
    H_r_down = dq.pwc(t_edges, jnp.conj(coeff_r), sigma_m @ a_dag)
    H_b_up = dq.pwc(t_edges, coeff_b, sigma_p @ a_dag)
    H_b_down = dq.pwc(t_edges, jnp.conj(coeff_b), sigma_m @ a)
    H = H_r_up + H_r_down + H_b_up + H_b_down

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
    alpha_cat,
    n_boson,
    extent,
    grid_size,
    top_k,
    cat_parity="0",
    gkp_delta=None,
    gkp_logical=None,
    cat_phase=None,
    gkp_squeeze_r=None,
    gkp_envelope_kappa=None,
    gkp_lattice_trunc=4,
):
    """
    Select fixed phase-space points where |W_target| is largest.
    """
    gkp_delta, gkp_logical = _resolve_gkp_params(
        alpha_cat=alpha_cat,
        cat_parity=cat_parity,
        gkp_delta=gkp_delta,
        gkp_logical=gkp_logical,
    )
    axis = np.linspace(-extent, extent, grid_size)
    target = _gkp_state(
        delta=gkp_delta,
        n_boson=n_boson,
        logical=gkp_logical,
        rel_phase=cat_phase,
        squeeze_r=gkp_squeeze_r,
        envelope_kappa=gkp_envelope_kappa,
        lattice_trunc=gkp_lattice_trunc,
    )
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


def trapped_ion_gkp_sim(
    phi_r,
    phi_b,
    amp_r=None,
    amp_b=None,
    n_boson=20,
    omega=2 * np.pi * 0.002,
    t_step=1.0,
    n_times=None,
    alpha_cat=0.301,
    cat_parity="0",
    gkp_delta=None,
    gkp_logical=None,
    cat_phase=None,
    sample_mode="gkp",
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
    gkp_squeeze_r=None,
    gkp_envelope_kappa=None,
    gkp_lattice_trunc=4,
):
    """
    Simulate trapped-ion state preparation with RSB/BSB controls and return
    a measurement-based reward derived from sampled characteristic values.
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
        )
        rho_boson = None

    target = _gkp_state(
        delta=gkp_delta,
        n_boson=n_boson,
        logical=gkp_logical,
        rel_phase=cat_phase,
        squeeze_r=gkp_squeeze_r,
        envelope_kappa=gkp_envelope_kappa,
        lattice_trunc=gkp_lattice_trunc,
    )
    target_rho = target @ target.dag()

    if sample_points is None:
        if sample_mode in ("gkp", "cat"):
            sample_points = _gkp_focus_points()
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


def trapped_ion_gkp_sim_batch(
    phi_r,
    phi_b,
    amp_r=None,
    amp_b=None,
    n_boson=20,
    omega=2 * np.pi * 0.002,
    t_step=1.0,
    n_times=None,
    alpha_cat=0.301,
    cat_parity="0",
    gkp_delta=None,
    gkp_logical=None,
    cat_phase=None,
    sample_mode="gkp",
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
    gkp_squeeze_r=None,
    gkp_envelope_kappa=None,
    gkp_lattice_trunc=4,
):
    rng = np.random.default_rng(seed)

    phi_r = np.asarray(phi_r, dtype=float)
    phi_b = np.asarray(phi_b, dtype=float)
    if phi_r.shape != phi_b.shape:
        raise ValueError("phi_r and phi_b must have the same shape.")

    force_serial = os.environ.get("QCRL_SERIAL_SIM", "0") == "1"
    if force_serial and phi_r.ndim > 1 and phi_r.shape[0] > 1:
        rewards = []
        fidelities = []
        rho_list = []
        target_rho = None
        for ii in range(phi_r.shape[0]):
            seed_i = None if seed is None else int(rng.integers(1, 2**31 - 1))
            if return_details:
                r_i, f_i, rho_i, target_rho = trapped_ion_gkp_sim(
                    phi_r[ii],
                    phi_b[ii],
                    amp_r=None if amp_r is None else np.asarray(amp_r, dtype=float)[ii],
                    amp_b=None if amp_b is None else np.asarray(amp_b, dtype=float)[ii],
                    n_boson=n_boson,
                    omega=omega,
                    t_step=t_step,
                    n_times=n_times,
                    alpha_cat=gkp_delta,
                    cat_parity=gkp_logical,
                    gkp_delta=gkp_delta,
                    gkp_logical=gkp_logical,
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
                    gkp_squeeze_r=gkp_squeeze_r,
                    gkp_envelope_kappa=gkp_envelope_kappa,
                    gkp_lattice_trunc=gkp_lattice_trunc,
                )
                rewards.append(r_i)
                fidelities.append(f_i)
                if return_density:
                    rho_list.append(rho_i)
            else:
                r_i = trapped_ion_gkp_sim(
                    phi_r[ii],
                    phi_b[ii],
                    amp_r=None if amp_r is None else np.asarray(amp_r, dtype=float)[ii],
                    amp_b=None if amp_b is None else np.asarray(amp_b, dtype=float)[ii],
                    n_boson=n_boson,
                    omega=omega,
                    t_step=t_step,
                    n_times=n_times,
                    alpha_cat=gkp_delta,
                    cat_parity=gkp_logical,
                    gkp_delta=gkp_delta,
                    gkp_logical=gkp_logical,
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
                    gkp_squeeze_r=gkp_squeeze_r,
                    gkp_envelope_kappa=gkp_envelope_kappa,
                    gkp_lattice_trunc=gkp_lattice_trunc,
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
        )
        rho_boson = None

    target = _gkp_state(
        delta=gkp_delta,
        n_boson=n_boson,
        logical=gkp_logical,
        rel_phase=cat_phase,
        squeeze_r=gkp_squeeze_r,
        envelope_kappa=gkp_envelope_kappa,
        lattice_trunc=gkp_lattice_trunc,
    )
    target_rho = target @ target.dag()

    if sample_points is None:
        if sample_mode in ("gkp", "cat"):
            sample_points = _gkp_focus_points()
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


# Backward-compatible aliases for older naming.
def trapped_ion_cat_sim(*args, **kwargs):
    return trapped_ion_gkp_sim(*args, **kwargs)


def trapped_ion_cat_sim_batch(*args, **kwargs):
    return trapped_ion_gkp_sim_batch(*args, **kwargs)
