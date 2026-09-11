"""Bounded coordinates for fitting material and effective contact parameters."""
from collections.abc import Mapping, Sequence
import math

import numpy as np

from .state import PARAMETER_NAMES, validate_parameters


DEFAULT_BOUNDS = {
    "youngs_modulus": (100.0, 1.0e7),
    "poisson_ratio": (0.01, 0.49),
    "viscosity": (0.0, 1000.0),
    "plastic_min": (0.5, 1.0),
    "plastic_max": (1.0, 1.5),
    "tool_retention": (0.0, 1.0),
    "floor_retention": (0.0, 1.0),
}
DEFAULT_SCALES = {
    "poisson_ratio": 0.1,
    "viscosity": 100.0,
    "plastic_min": 0.1,
    "plastic_max": 0.1,
    "tool_retention": 1.0,
    "floor_retention": 1.0,
}


class PhysicalParameterSpace:
    """Transform fitted values without changing any fixed physical parameters.

    E uses log(E / scale); viscosity and retention use value / scale; the
    lower/upper plastic bounds use (1 - lower) / scale and (upper - 1) / scale.
    Nu uses a linear scaled coordinate with explicit bounds below 0.5. Projection
    belongs to the optimizer: ``pullback`` does not differentiate projection.
    The viscosity zero boundary is represented exactly.
    """

    def __init__(self, initial: Mapping[str, float], fit: Sequence[str], plasticity: str,
                 bounds: Mapping[str, Sequence[float]] | None = None,
                 scales: Mapping[str, float] | None = None):
        validate_parameters(initial)
        self.initial = {name: float(initial[name]) for name in PARAMETER_NAMES}
        if isinstance(fit, (str, bytes)):
            raise ValueError("fit must be a nonempty sequence of parameter names")
        self.fit = tuple(fit)
        if not self.fit or len(set(self.fit)) != len(self.fit):
            raise ValueError("fit must be nonempty and contain no duplicates")
        unknown = set(self.fit) - set(PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"Unknown fitted parameters: {sorted(unknown)}")
        if plasticity not in {"none", "stretch-clamp"}:
            raise ValueError("plasticity must be none or stretch-clamp")
        self.plasticity = plasticity
        if plasticity == "none" and {"plastic_min", "plastic_max"}.intersection(self.fit):
            raise ValueError("Fitting plastic limits requires explicit stretch-clamp plasticity")
        supplied_bounds = dict(bounds or {})
        supplied_scales = dict(scales or {})
        for name, supplied in (("bounds", supplied_bounds), ("scales", supplied_scales)):
            unknown = set(supplied) - set(PARAMETER_NAMES)
            if unknown:
                raise ValueError(f"Unknown {name} parameters: {sorted(unknown)}")
        self.bounds = dict(DEFAULT_BOUNDS)
        for name, values in supplied_bounds.items():
            a = np.asarray(values, dtype=np.float64)
            if a.shape != (2,) or not np.isfinite(a).all():
                raise ValueError(f"{name} bounds must be two finite numbers")
            self.bounds[name] = tuple(float(v) for v in a)
        for name, (lower, upper) in self.bounds.items():
            if lower >= upper:
                raise ValueError(f"{name} bounds must have positive width")
            self._validate_bound(name, lower, upper)
            if (name in self.fit or name in supplied_bounds) and not lower <= self.initial[name] <= upper:
                raise ValueError(f"Initial {name} is outside the configured fitting bounds")
        self.scales = {"youngs_modulus": self.initial["youngs_modulus"], **DEFAULT_SCALES}
        self.scales.update({name: float(value) for name, value in supplied_scales.items()})
        if not all(np.isfinite(value) and value > 0 for value in self.scales.values()):
            raise ValueError("All parameter scales must be finite and positive")
        encoded_bounds = [sorted(self._encode(name, value) for value in self.bounds[name]) for name in self.fit]
        self.lower = np.asarray([values[0] for values in encoded_bounds], dtype=np.float64)
        self.upper = np.asarray([values[1] for values in encoded_bounds], dtype=np.float64)
        if not (np.isfinite(self.lower).all() and np.isfinite(self.upper).all()):
            raise ValueError("Parameter bounds and scales produced nonfinite coordinates")

    @staticmethod
    def _validate_bound(name, lower, upper):
        valid = True
        if name == "youngs_modulus":
            valid = lower > 0
        elif name == "poisson_ratio":
            valid = -1 < lower < upper < 0.5
        elif name == "viscosity":
            valid = lower >= 0
        elif name == "plastic_min":
            valid = 0 < lower < upper <= 1
        elif name == "plastic_max":
            valid = 1 <= lower < upper
        else:
            valid = 0 <= lower < upper <= 1
        if not valid:
            raise ValueError(f"Physically invalid fitting bounds for {name}")

    def _encode(self, name, value):
        scale = self.scales[name]
        if name == "youngs_modulus":
            return math.log(value) - math.log(scale)
        if name == "plastic_min":
            return (1.0 - value) / scale
        if name == "plastic_max":
            return (value - 1.0) / scale
        return value / scale

    def _vector(self, coordinates):
        vector = np.asarray(coordinates, dtype=np.float64)
        if vector.shape != (len(self.fit),) or not np.isfinite(vector).all():
            raise ValueError(f"Require {len(self.fit)} finite parameter coordinates")
        return vector

    def coordinates(self):
        """Return the initial fitted coordinates, not optimizer-owned state."""
        return np.asarray([self._encode(name, self.initial[name]) for name in self.fit], dtype=np.float64)

    def project(self, coordinates):
        return np.clip(self._vector(coordinates), self.lower, self.upper)

    def physical(self, coordinates):
        u = self._vector(coordinates)
        if np.any(u < self.lower) or np.any(u > self.upper):
            raise ValueError("Parameter coordinates are outside their bounds; project proposals first")
        result = dict(self.initial)
        for name, value in zip(self.fit, u):
            if name == "youngs_modulus":
                physical = math.exp(float(value) + math.log(self.scales[name]))
            elif name == "plastic_min":
                physical = 1.0 - self.scales[name] * float(value)
            elif name == "plastic_max":
                physical = 1.0 + self.scales[name] * float(value)
            else:
                physical = self.scales[name] * float(value)
            # Avoid a one-ulp bound overshoot when decoding a boundary coordinate.
            result[name] = float(np.clip(physical, *self.bounds[name]))
        validate_parameters(result)
        return result

    def pullback(self, coordinates, physical_gradient: Mapping[str, float]):
        physical = self.physical(coordinates)
        unknown = set(physical_gradient) - set(PARAMETER_NAMES)
        missing = set(self.fit) - set(physical_gradient)
        if unknown or missing:
            raise ValueError(f"Invalid gradient keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
        if not all(np.isfinite(value) for value in physical_gradient.values()):
            raise ValueError("Physical parameter gradients must be finite")
        gradient = []
        for name in self.fit:
            derivative = self.scales[name]
            if name == "youngs_modulus":
                derivative = physical[name]
            elif name == "plastic_min":
                derivative = -derivative
            gradient.append(float(physical_gradient[name]) * derivative)
        result = np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("Parameter gradient transform overflowed")
        return result

    def settings(self):
        return {
            "schema_version": 1,
            "initial": dict(self.initial),
            "fit": list(self.fit),
            "plasticity": self.plasticity,
            "bounds": {name: list(self.bounds[name]) for name in PARAMETER_NAMES},
            "scales": dict(self.scales),
        }

    def serialize(self):
        return self.settings()
