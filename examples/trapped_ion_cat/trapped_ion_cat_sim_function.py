import numpy as np
import jax.numpy as jnp
import dynamiqs as dq


def _cat_state(alpha, n_boson, parity="even"):
    psi_p = dq.coherent(n_boson, alpha)
    psi_m = dq.coherent(n_boson, -alpha)
    psi = psi_p - psi_m if parity == "odd" else psi_p + psi_m
    return psi.unit()


def _parity_operator(n_boson):
    return dq.parity(n_boson)


def _sample_points(grid_size, extent):
    axis = np.linspace(-extent, extent, grid_size)
    return [x + 1j * y for x in axis for y in axis]


def _cat_focus_points(alpha):
    a = float(alpha)
    return [
        0.0 + 0.0j,
        a + 0.0j,
        -a + 0.0j,
        0.5 * a + 0.0j,
        -0.5 * a + 0.0j,
        1.2 * a + 0.0j,
        -1.2 * a + 0.0j,
        0.0 + 0.5 * a * 1j,
        0.0 - 0.5 * a * 1j,
        0.0 + a * 1j,
        0.0 - a * 1j,
    ]


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
    cat_parity="even",
    mix_uniform=0.0,
):
    axis = np.linspace(-extent, extent, grid_size)
    if grid_size > 1:
        delta = float(axis[1] - axis[0])
    else:
        delta = float(2.0 * extent)
    area_element = delta * delta
    target = _cat_state(alpha_cat, n_boson, parity=cat_parity)
    target_rho = target @ target.dag()
    points = [x + 1j * y for x in axis for y in axis]
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
):
    phi_r = jnp.asarray(phi_r, dtype=jnp.float32)
    phi_b = jnp.asarray(phi_b, dtype=jnp.float32)
    if phi_r.shape != phi_b.shape:
        raise ValueError("phi_r and phi_b must have the same shape.")
    if phi_r.ndim == 1:
        rho_boson, rho_qubit = _simulate_boson_state_batch(
            phi_r[jnp.newaxis, :],
            phi_b[jnp.newaxis, :],
            n_boson,
            omega_r,
            omega_b,
            t_step,
            n_times=n_times,
        )
        return rho_boson[0], rho_qubit[0]
    return _simulate_boson_state_batch(
        phi_r, phi_b, n_boson, omega_r, omega_b, t_step, n_times=n_times
    )


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
    options = dq.Options(save_states=False, progress_meter=False)
    result = dq.sesolve(H, psi0, tsave=tsave, options=options)
    psi_final = result.final_state
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
    cat_parity="even",
):
    """
    Select a fixed set of phase-space points where |W_target| is largest.
    This yields a high-SNR, fixed observable for model-free rewards.
    """
    axis = np.linspace(-extent, extent, grid_size)
    target = _cat_state(alpha_cat, n_boson, parity=cat_parity)
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


def trapped_ion_cat_sim(
    phi_r,
    phi_b,
    amp_r=None,
    amp_b=None,
    n_boson=20,
    omega=2 * np.pi * 0.002,
    t_step=1.0,
    n_times=None,
    alpha_cat=2.0,
    cat_parity="even",
    sample_mode="cat",
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
    reward_norm=None,
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

    rho_boson, rho_qubit = simulate_boson_state(
        phi_r,
        phi_b,
        n_boson=n_boson,
        omega_r=omega_r,
        omega_b=omega_b,
        t_step=t_step,
        n_times=n_times,
    )

    target = _cat_state(alpha_cat, n_boson, parity=cat_parity)
    target_rho = target @ target.dag()

    if sample_points is None:
        if sample_mode == "cat":
            sample_points = _cat_focus_points(alpha_cat)
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
            chi_expect = _characteristic_at_point(rho_boson, alpha)
            if n_shots <= 0:
                meas.append(chi_expect)
            else:
                chi_real = float(np.clip(chi_expect.real, -1.0, 1.0))
                p_plus = 0.5 * (1.0 + chi_real)
                meas.append(1.0 if rng.random() < p_plus else -1.0)
        meas = np.array(meas, dtype=complex)
        denom = sample_weights
        reward_terms = meas * np.conjugate(target_values) / denom
        reward = float(
            reward_scale * (sample_area / np.pi) * np.mean(reward_terms).real
        )
        if reward_norm is not None:
            reward = float(reward / reward_norm)
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

    fidelity = float(dq.expect(target_rho, rho_boson).real)
    return reward, fidelity, rho_boson, target_rho


def trapped_ion_cat_sim_batch(
    phi_r,
    phi_b,
    amp_r=None,
    amp_b=None,
    n_boson=20,
    omega=2 * np.pi * 0.002,
    t_step=1.0,
    n_times=None,
    alpha_cat=2.0,
    cat_parity="even",
    sample_mode="cat",
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
    reward_norm=None,
):
    rng = np.random.default_rng(seed)

    phi_r = np.asarray(phi_r, dtype=float)
    phi_b = np.asarray(phi_b, dtype=float)
    if phi_r.shape != phi_b.shape:
        raise ValueError("phi_r and phi_b must have the same shape.")

    if amp_r is None:
        omega_r = omega
    else:
        omega_r = omega * np.asarray(amp_r, dtype=float)
    if amp_b is None:
        omega_b = omega
    else:
        omega_b = omega * np.asarray(amp_b, dtype=float)

    rho_boson, rho_qubit = simulate_boson_state(
        phi_r,
        phi_b,
        n_boson=n_boson,
        omega_r=omega_r,
        omega_b=omega_b,
        t_step=t_step,
        n_times=n_times,
    )

    target = _cat_state(alpha_cat, n_boson, parity=cat_parity)
    target_rho = target @ target.dag()

    if sample_points is None:
        if sample_mode == "cat":
            sample_points = _cat_focus_points(alpha_cat)
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

        disp_ops = dq.displace(n_boson, alphas)
        meas = dq.expect(disp_ops, rho_boson)
        if meas.shape[0] == len(sample_points):
            meas = jnp.swapaxes(meas, 0, 1)

        if n_shots > 0:
            chi_real = np.clip(np.array(meas.real), -1.0, 1.0)
            p_plus = 0.5 * (1.0 + chi_real)
            meas = np.where(rng.random(size=p_plus.shape) < p_plus, 1.0, -1.0)
            meas = jnp.asarray(meas, dtype=jnp.complex64)

        reward_terms = meas * jnp.conjugate(target_vals) / weights
        reward = reward_scale * (sample_area / np.pi) * jnp.mean(reward_terms, axis=1).real
        if reward_norm is not None:
            reward = reward / reward_norm
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

    fidelity = np.array(dq.expect(target_rho, rho_boson).real, dtype=float)
    return reward, fidelity, rho_boson, target_rho
