"""Composite derivatives of the current 3-D polar and stretch-clamp maps.

The caller must initialize Taichi with default_fp matching its matrix fields.
Forward operations use Taichi's unmodified SVD/polar decomposition. The VJPs
below differentiate the mathematical composite maps, not Taichi's finite-iteration
SVD algorithm or independently chosen singular vectors. Taichi 1.7.4's default
fp64 SVD can differ from a converged factorization near repeated stretches by
about 1e-10 on unit-scale inputs; tiny-step differences of that forward algorithm
can therefore disagree with the composite derivative. Tests quantify this error
separately. These VJPs are intended for orientation-preserving matrices with
positive singular values bounded away from zero; the solver must reject
unsupported states before invoking these functions.

At a clamp threshold the derivative from the capped side is used: input-stretch
sensitivity is zero and bound sensitivity is one. If the bounds coincide, the
lower bound takes precedence at equality. The Von Mises map likewise uses the
active-return derivative at its yield threshold. There is no classical derivative
at these thresholds. Repeated stretches use analytic divided-difference limits.
"""

import taichi as ti


@ti.func
def polar_forward(F):
    rotation, _ = ti.polar_decompose(F)
    return rotation


@ti.func
def clamp_forward(F, lower, upper):
    U, sig, V = ti.svd(F)
    yielded = 0
    for i in ti.static(range(3)):
        original = sig[i, i]
        projected = ti.min(ti.max(original, lower), upper)
        if ti.abs(original - projected) > 1e-6:
            yielded = 1
        sig[i, i] = projected
    return U @ sig @ V.transpose(), yielded


@ti.func
def polar_vjp(F, R_bar):
    U, sig, V = ti.svd(F)
    K = U.transpose() @ R_bar @ V
    local_bar = F * 0.0
    for i, j in ti.static(ti.ndrange(3, 3)):
        if ti.static(i != j):
            local_bar[i, j] = (K[i, j] - K[j, i]) / (sig[i, i] + sig[j, j])
    return U @ local_bar @ V.transpose()


@ti.func
def clamp_vjp(F, lower, upper, Y_bar):
    U, sig, V = ti.svd(F)
    K = U.transpose() @ Y_bar @ V
    projected = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
    region = ti.Vector([0, 0, 0])
    lower_bar = F[0, 0] * 0.0
    upper_bar = F[0, 0] * 0.0
    for i in ti.static(range(3)):
        projected[i] = ti.min(ti.max(sig[i, i], lower), upper)
        if sig[i, i] <= lower:
            region[i] = -1
            lower_bar += K[i, i]
        elif sig[i, i] >= upper:
            region[i] = 1
            upper_bar += K[i, i]

    local_bar = F * 0.0
    for i, j in ti.static(ti.ndrange(3, 3)):
        if ti.static(i == j):
            if region[i] == 0:
                local_bar[i, i] = K[i, i]
        else:
            symmetric_coefficient = F[0, 0] * 0.0
            # The equal-region formulas also handle repeated stretches exactly.
            if region[i] == 0 and region[j] == 0:
                symmetric_coefficient = 1.0
            elif region[i] != region[j]:
                symmetric_coefficient = (projected[i] - projected[j]) / (sig[i, i] - sig[j, j])
            skew_coefficient = (projected[i] + projected[j]) / (sig[i, i] + sig[j, j])
            local_bar[i, j] = (
                symmetric_coefficient * (K[i, j] + K[j, i]) * 0.5
                + skew_coefficient * (K[i, j] - K[j, i]) * 0.5
            )
    return U @ local_bar @ V.transpose(), lower_bar, upper_bar


@ti.func
def von_mises_forward(F, youngs_modulus, poisson_ratio, yield_stress):
    """Perfect isotropic J2 return in principal Hencky-strain coordinates."""
    U, sig, V = ti.svd(F)
    epsilon = ti.Vector([0.0, 0.0, 0.0])
    mean = F[0, 0] * 0.0
    for i in ti.static(range(3)):
        epsilon[i] = ti.log(sig[i, i])
        mean += epsilon[i] / 3.0
    deviator = epsilon - mean
    radius = deviator.norm()
    limit = yield_stress * (1.0 + poisson_ratio) / youngs_modulus
    yielded = 0
    if radius >= limit:
        yielded = 1
        corrected_epsilon = mean + limit * deviator / radius
        for i in ti.static(range(3)):
            sig[i, i] = ti.exp(corrected_epsilon[i])
    return U @ sig @ V.transpose(), yielded


@ti.func
def von_mises_vjp(F, youngs_modulus, poisson_ratio, yield_stress, Y_bar):
    """VJP of :func:`von_mises_forward`, including projection bars for E and nu."""
    U, sig, V = ti.svd(F)
    epsilon = ti.Vector([0.0, 0.0, 0.0])
    mean = F[0, 0] * 0.0
    for i in ti.static(range(3)):
        epsilon[i] = ti.log(sig[i, i])
        mean += epsilon[i] / 3.0
    deviator = epsilon - mean
    radius = deviator.norm()
    limit = yield_stress * (1.0 + poisson_ratio) / youngs_modulus
    young_bar = F[0, 0] * 0.0
    poisson_bar = F[0, 0] * 0.0
    F_bar = Y_bar
    if radius >= limit:
        normal = deviator / radius
        corrected = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            corrected[i] = ti.exp(mean + limit * normal[i])

        strain_jacobian = F * 0.0
        stretch_jacobian = F * 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            centering = (1.0 if i == j else 0.0) - 1.0 / 3.0
            strain_jacobian[i, j] = (
                1.0 / 3.0
                + limit / radius * (centering - normal[i] * normal[j])
            )
            stretch_jacobian[i, j] = corrected[i] * strain_jacobian[i, j] / sig[j, j]

        K = U.transpose() @ Y_bar @ V
        diagonal_bar = ti.Vector([K[0, 0], K[1, 1], K[2, 2]])
        stretch_bar = stretch_jacobian.transpose() @ diagonal_bar
        limit_bar = F[0, 0] * 0.0
        for i in ti.static(range(3)):
            limit_bar += K[i, i] * corrected[i] * normal[i]
        young_bar = -limit_bar * limit / youngs_modulus
        poisson_bar = limit_bar * limit / (1.0 + poisson_ratio)

        local_bar = F * 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            if ti.static(i == j):
                local_bar[i, i] = stretch_bar[i]
            else:
                scale = ti.max(1.0, ti.max(ti.abs(sig[i, i]), ti.abs(sig[j, j])))
                symmetric_coefficient = F[0, 0] * 0.0
                if ti.abs(sig[i, i] - sig[j, j]) <= 1e-6 * scale:
                    symmetric_coefficient = stretch_jacobian[i, i] - stretch_jacobian[i, j]
                else:
                    symmetric_coefficient = (corrected[i] - corrected[j]) / (sig[i, i] - sig[j, j])
                skew_coefficient = (corrected[i] + corrected[j]) / (sig[i, i] + sig[j, j])
                local_bar[i, j] = (
                    symmetric_coefficient * (K[i, j] + K[j, i]) * 0.5
                    + skew_coefficient * (K[i, j] - K[j, i]) * 0.5
                )
        F_bar = U @ local_bar @ V.transpose()
    return F_bar, young_bar, poisson_bar
