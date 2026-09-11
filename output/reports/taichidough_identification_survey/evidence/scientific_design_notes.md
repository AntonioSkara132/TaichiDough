# Scientific analysis to test against the source review

These are analytical deductions and proposed experiments, not reported TaichiDough results or priority claims.

## The inverse problem is not defined by the optimizer

Let theta contain mechanical parameters and psi contain reconstruction, contact, registration and discretization assumptions. Recorded tool pose u(t) is prescribed input; camera observations are y_t = H(S(theta, psi; u) at t) + noise. A forward sweep can solve the same objective as a gradient optimizer. Its scientific validity depends on whether the model can be distinguished by available observations, whether the objective is appropriate, and whether the estimates predict independent trials. It does not depend on differentiability.

## Scale ambiguity with displacement-controlled loading

For small-strain elasticity with fixed Poisson ratio, sigma = E * Cbar:epsilon. In a quasistatic experiment without meaningful known body forces, all external boundary conditions consisting of prescribed displacements and homogeneous tractions, scaling E by c preserves the displacement solution and only changes the unobserved reactions. This is an explicit counterexample to the idea that depth plus known tool displacement necessarily identifies absolute E.

For a linear Kelvin-Voigt material sigma = E epsilon + eta epsilon_dot, multiplying E and eta by the same positive factor preserves analogous force-free quasistatic displacement histories. During unforced recovery in the scalar idealization, epsilon(t) = epsilon(0) exp(-t E/eta), which identifies a timescale eta/E rather than two absolute parameters. This example must be labeled scalar/linear/idealized, not a description of the whole current MPM material.

Known inertia/density, gravity-induced deformation, or measured forces can add an absolute scale, but practical signal strength matters. In dynamics, E/rho and eta/rho are relevant combinations. Independently measured mass plus uncertain reconstructed volume only partially resolves rho. In an elastoplastic model the ratio sigma_y/E can likewise be more directly visible than either stress scale separately. A singular-value clamp is a strain-limit parameter and must not be mislabeled a yield stress.

## Distinguishing mechanisms

Fast/slow press: time dependence, but confounded by inertia, friction, depth temporal filtering, rate-dependent contact and history.
Partial unload/release with geometric observation: recovery; delayed recovery is not proof of permanent plastic strain.
Long recovery plus repeated compression/unload cycles at varied amplitude: candidate elastic/plastic distinction; irreversible flow, damage, adhesion and a too-short observation can still mimic plasticity.
Fixed indentation dwell: a depth camera cannot measure stress relaxation if the geometry is held fixed. Force sensing is needed for direct stress-relaxation curves; lateral deformation or post-release dynamics may still contain indirect information.
Gravity settling/free dynamics: potentially fixes elastic scale if density, initial stress/geometry and sampling rate are independently reliable.
Controlled sliding after bulk fit: isolates a tangential response better than mixing pressing/sliding, but current velocity damping need not be a Coulomb coefficient.

## Falsifiable candidate contributions

1. Information-aware action selection improves cross-trajectory prediction and reduces near-optimal parameter sets versus arbitrary recording under matched simulation budget.
2. Elastic + viscous + plastic model classes produce measurably different held-out recovery/rate-cycle predictions; added parameters retained only when independent prediction improves.
3. A single-camera, hidden-volume-aware calibration protocol achieves a quantified accuracy/uncertainty trade-off against additional views.
4. Calibrated simulator parameters remain useful (or demonstrably fail to transfer) across numerical resolutions and contact assumptions.
5. A public repeated-batch, real-dough benchmark documents material preparation and observation/kinematic uncertainty and allows testing the preceding questions.

These are research directions. They should not be claimed as existing algorithms, established results, or first-of-kind contributions without source comparison and experiments.
