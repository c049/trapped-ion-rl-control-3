import numpy as np
import qutip as qt


def _cat_state(alpha, n_boson, parity="even"):
    psi_p = qt.coherent(n_boson, alpha)
    psi_m = qt.coherent(n_boson, -alpha)
    if parity == "odd":
        psi = psi_p - psi_m
    else:
        psi = psi_p + psi_m
    return psi.unit()


def _parity_operator(n_boson):
    try:
        return qt.parity(n_boson)
    except AttributeError:
        diag = [1 if (n % 2 == 0) else -1 for n in range(n_boson)]
        return qt.Qobj(np.diag(diag))


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
    n_boson = rho.dims[0][0]
    disp = qt.displace(n_boson, alpha)
    displaced = disp * rho * disp.dag()
    return (2.0 / np.pi) * qt.expect(parity_op, displaced).real


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


def _as_time_array(value, n_steps, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_steps, float(arr))
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size != n_steps:
        raise ValueError(f"{name} must be scalar or length {n_steps}.")
    return arr


def simulate_boson_state(
    phi_r,
    phi_b,
    n_boson,
    omega_r,
    omega_b,
    t_step,
    n_times=None,
):
    phi_r = np.asarray(phi_r, dtype=float)
    phi_b = np.asarray(phi_b, dtype=float)
    if phi_r.shape != phi_b.shape:
        raise ValueError("phi_r and phi_b must have the same shape.")

    n_steps = phi_r.size
    t_duration = n_steps * t_step
    ts = np.linspace(0.0, t_duration, n_steps)
    if n_times is None:
        n_times = max(3 * n_steps, n_steps + 1)
    tlist = np.linspace(0.0, t_duration, n_times)

    a = qt.tensor(qt.qeye(2), qt.destroy(n_boson))
    a_dag = a.dag()
    sigma_p = qt.tensor(qt.sigmap(), qt.qeye(n_boson))
    sigma_m = qt.tensor(qt.sigmam(), qt.qeye(n_boson))

    omega_r = _as_time_array(omega_r, n_steps, "omega_r")
    omega_b = _as_time_array(omega_b, n_steps, "omega_b")
    coeff_r = 0.5 * omega_r * np.exp(1j * phi_r)
    coeff_b = 0.5 * omega_b * np.exp(1j * phi_b)

    coeff_r_func = qt.interpolate.Cubic_Spline(ts[0], ts[-1], coeff_r)
    coeff_r_conj = qt.interpolate.Cubic_Spline(ts[0], ts[-1], np.conj(coeff_r))
    coeff_b_func = qt.interpolate.Cubic_Spline(ts[0], ts[-1], coeff_b)
    coeff_b_conj = qt.interpolate.Cubic_Spline(ts[0], ts[-1], np.conj(coeff_b))

    H = [
        [sigma_p * a, coeff_r_func],
        [sigma_m * a_dag, coeff_r_conj],
        [sigma_p * a_dag, coeff_b_func],
        [sigma_m * a, coeff_b_conj],
    ]

    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(n_boson, 0))
    result = qt.sesolve(H, psi0, tlist=tlist)
    psi_final = result.states[-1]
    rho_boson = qt.ptrace(psi_final, 1)
    rho_qubit = qt.ptrace(psi_final, 0)
    return rho_boson, rho_qubit


def wigner_grid(rho, xvec, yvec):
    return qt.wigner(rho, xvec, yvec)


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
    w_target = wigner_grid(target.proj(), axis, axis)
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
    n_shots=1,
    seed=None,
    return_details=False,
):
    """
    Simulate trapped-ion state preparation with RSB/BSB controls and return
    a measurement-based reward derived from sampled Wigner values.
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

    parity_op = _parity_operator(n_boson)
    if sample_points is None:
        if sample_mode == "cat":
            sample_points = _cat_focus_points(alpha_cat)
        elif sample_mode == "random":
            sample_points = _random_points(n_sample_points, sample_extent, rng)
        else:
            sample_points = _sample_points(sample_grid, sample_extent)
    else:
        sample_points = list(sample_points)

    target = _cat_state(alpha_cat, n_boson, parity=cat_parity)
    target_rho = target.proj()
    target_wigner = _target_wigner_values(target_rho, sample_points, parity_op)

    w_meas = []
    for alpha in sample_points:
        parity_expect = _wigner_at_point(rho_boson, alpha, parity_op) * (np.pi / 2.0)
        parity_sample = _sample_parity(parity_expect, n_shots, rng)
        w_meas.append((2.0 / np.pi) * parity_sample)
    w_meas = np.array(w_meas, dtype=float)

    denom = float(np.mean(target_wigner ** 2))
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    reward = float(np.mean(target_wigner * w_meas) / denom)
    reward = float(np.clip(reward, -1.0, 1.0))
    if not return_details:
        return reward

    fidelity = float((target.dag() * rho_boson * target).full()[0, 0].real)
    return reward, fidelity, rho_boson, target_rho
