"""Shape-descriptor loss wrapper for TaichiDough rollouts (no change to the experiment tree).

Descriptors per frame: soft width along x, soft width along z (soft max - soft min, temperature tau), soft height
(soft max of y). Recorded descriptors come from the recorded clouds in the scene frame. The loss is the mean over
scored frames of sum_d ((d_sim - d_rec) / scale)^2 (optionally on changes relative to the first scored frame, which
cancels a constant reconstruction offset), weighted per frame, plus `orig_weight` times the original observation
loss. Returns LossRecord(value, dL/dpositions, components, diagnostics) like ObservationLoss."""
import numpy as np

def _band_mean(c, lo_frac, hi_frac, sign=1.0):
    """Mean of the values ranked between lo_frac and hi_frac from the top (sign=+1) or bottom (sign=-1); gradient 1/k on
    the band members (piecewise linear, robust to the extreme (1-hi_frac) fraction of outliers)."""
    n = c.shape[0]; order = np.argsort(sign * c); lo, hi = int(lo_frac * n), int(hi_frac * n)  # ascending order of sign*c: top band = ranks [lo_frac, hi_frac)
    idx = order[lo:hi]; k = len(idx); m = float(c[idx].mean()); g = np.zeros_like(c); g[idx] = 1.0 / k
    return m, g

def descriptors(x, tau=None, band=(0.90, 0.975)):
    """x: (N,3) scene-frame positions. d = [width_x, width_z, height] from band means on each side; dd/dx (3,N,3)."""
    d = np.zeros(3); g = np.zeros((3, x.shape[0], 3))
    for j, axis in enumerate((0, 2)):
        mx, gmx = _band_mean(x[:, axis], *band, +1.0); mn, gmn = _band_mean(x[:, axis], *band, -1.0)
        d[j] = mx - mn; g[j, :, axis] = gmx - gmn
    mh, gh = _band_mean(x[:, 1], *band, +1.0); d[2] = mh; g[2, :, 1] = gh
    return d, g

class ShapeDescriptorLoss:
    def __init__(self, base_loss, recorded, *, tau=0.004, scale=0.01, shape_weight=1.0, orig_weight=0.0, relative=False, frame_weights=None, initial_positions=None):
        self.base, self.rec, self.tau, self.scale = base_loss, recorded, tau, scale  # rec: {frame_index: np.array(3,)}
        self.ws, self.wo, self.relative, self.fw = shape_weight, orig_weight, relative, frame_weights or {}
        self.d0_sim = descriptors(np.asarray(initial_positions, dtype=np.float64), tau)[0] if (relative and initial_positions is not None) else None
        self.d0_rec = recorded.get(0) if relative else None

    def value_and_grad_positions(self, positions, observation, compute_grad=True):
        x = np.asarray(positions, dtype=np.float64); fi = int(observation.frame_index)
        d, g = descriptors(x, self.tau); target = self.rec[fi]
        if self.relative: d = d - self.d0_sim; target = target - self.d0_rec
        r = (d - target) / self.scale; w = self.fw.get(fi, 1.0)
        shape_value = w * float((r * r).sum()); grad = (w * 2.0 / self.scale) * np.einsum("k,kni->ni", r, g) if compute_grad else None
        components = {"shape": shape_value, "width_x_mm": float(d[0] * 1000), "width_z_mm": float(d[1] * 1000), "height_mm": float(d[2] * 1000), "rec_width_x_mm": float(target[0] * 1000), "rec_width_z_mm": float(target[1] * 1000), "rec_height_mm": float(target[2] * 1000)}
        value = self.ws * shape_value; diagnostics = {"loss_version": "shape-descriptors-v1", "frame_index": fi, "step": int(observation.step), "frame_weight": w}
        gradient = self.ws * grad if compute_grad else None
        if self.wo > 0:
            rec = self.base.value_and_grad_positions(positions, observation, compute_grad=compute_grad)
            value += self.wo * rec.value; components.update({"orig_" + k: v for k, v in rec.components.items()}); diagnostics.update({"orig_" + k: v for k, v in rec.diagnostics.items()})
            if compute_grad: gradient = gradient + self.wo * np.asarray(rec.gradient, dtype=np.float64)
        from experiments.differentiable_mpm.loss import LossRecord
        return LossRecord(float(value), None if gradient is None else gradient.astype(np.asarray(positions).dtype), components, diagnostics)

if __name__ == "__main__":  # finite-difference check of the descriptor gradient
    rng = np.random.default_rng(0); x = rng.normal(size=(2000, 3)) * 0.03; tau = None
    d, g = descriptors(x, tau); eps = 1e-7; err = 0
    for _ in range(40):
        i, a = rng.integers(2000), rng.integers(3); xp = x.copy(); xp[i, a] += eps; dp = descriptors(xp, tau)[0]
        num = (dp - d) / eps; err = max(err, np.abs(num - g[:, i, a]).max())
    print("descriptors (mm):", np.round(d * 1000, 2), "| max |FD - analytic| gradient error:", err)
