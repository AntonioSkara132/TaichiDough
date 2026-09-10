% TaichiDough: Reconstructing and Simulating Deformable Dough
% Antonio Skara
% September 2026

# Abstract

TaichiDough is a research prototype for reconstructing, simulating, and evaluating deformable dough from recorded point clouds and the motion of two tools. The project combines a Taichi MLS-MPM simulator with a calibration and evaluation workflow intended to compare simulated deformation with recorded observations in metric scene coordinates. The current repository demonstrates a good legacy static initialization match and timestamped replay, while its longer proxy-tool rollout identifies the remaining work required for physically validated prediction.

# From Recorded Observation to Simulation

The workflow begins with recorded DeformPath point clouds and two captured tool trajectories. A scene calibration maps camera measurements into simulation coordinates. The intended metric procedure uses an AprilTag, exact camera intrinsics, a measured `scene_from_tag` transform, and a floor plane. The calibration collector checks the configured tag family and ID, tag edge length, timestamps, frame identifiers, and the TF path between the point-cloud and colour-camera frames. It then rejects pose outliers before writing a version-2 calibration record with transforms, intrinsics, diagnostics, and input fingerprints.

## Calibration Algorithm

AprilTag detection estimates a rigid pose $T_{C\leftarrow A}$ from the tag frame $A$ to the camera frame $C$ using the tag's known physical edge length and the camera intrinsics. The collector accepts multiple valid detections of the fixed tag, rather than treating one image as the calibration. Each pose is represented as a homogeneous transform,

$$
T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix},
$$

where $R \in SO(3)$ is rotation and $t \in \mathbb{R}^3$ is translation. It first takes the coordinate-wise median of the observed translations. Translation residuals are Euclidean distances from that median. Rotation residuals are geodesic angles on $SO(3)$, measured against the rotation with the smallest median pairwise angular distance.

For each residual type, the acceptance threshold is the smaller of a fixed safety limit and a robust median-absolute-deviation threshold:

$$
\tau = \min\left(\tau_{\max},\; \operatorname{median}(r) + 3.5 \times 1.4826 \times \operatorname{MAD}(r)\right).
$$

Detections must pass both the translation and rotation tests. The accepted translations are averaged arithmetically. Accepted rotations are averaged with the Markley quaternion method: normalized quaternions $q_i$ form $A=\sum_i q_iq_i^T$, and the unit eigenvector of $A$ belonging to its largest eigenvalue supplies the mean rotation. A final fixed residual check removes any remaining outlier before the pose is recomputed.

Given a measured scene-to-tag transform $T_{S\leftarrow A}$, the resulting metric camera transform is

$$
T_{S\leftarrow C}=T_{S\leftarrow A}T_{C\leftarrow A}^{-1}.
$$

If the point cloud has source frame $P$, TF provides $T_{C\leftarrow P}$ and the point cloud is mapped to the scene with $T_{S\leftarrow P}=T_{S\leftarrow C}T_{C\leftarrow P}$. This is a rigid calibration: it does not fit an axis permutation, scale, or post-hoc registration.

The checked-in Episode 18 result instead uses a legacy bootstrap calibration: an axis remapping, uniform scale, translation, and prescribed top-view camera. This is useful for reproducing existing benchmarks, but it is not an independently verified metric calibration. The repository contains the AprilTag pathway needed to replace it; however, no completed version-2 calibration result is committed.

After calibration, the reconstruction program converts the observed visible dough into a particle initialization. In the preferred floor-fill mode, it removes points at or below the calibrated floor clearance, groups retained points by occupied columns, and fills each column down to the floor. The resulting voxel count gives the reconstructed volume. A deterministic particle sample is drawn from that volume and is accompanied by metadata that records the calibration, floor, volume, particle count, and file fingerprints. When a measured dough mass is available, particle volume, density, and mass are derived from the reconstructed total volume and total mass, preserving those quantities when numerical resolution changes.

# MPM Simulator

The simulator uses three-dimensional Moving Least Squares Material Point Method (MLS-MPM) with Affine Particle-In-Cell (APIC) transfer. Each material particle stores position, velocity, deformation gradient, and affine velocity; optional plastic-volume state is also supported. Momentum and mass transfer between particles and a regular background grid at every step, while the grid integrates gravity and collision response.

The constitutive law combines corotated elastic stress with a simple viscous term. Young's modulus and Poisson ratio determine the elastic Lamé parameters. Plastic deformation can be represented by clamping the singular values of the deformation gradient, optionally with plastic-volume hardening and damping. This is a simplified dough-like model, rather than a completed rheological model of food dough.

## Core Simulation Equations

For a particle with deformation gradient $F_p$, APIC affine velocity matrix $C_p$, and timestep $\Delta t$, the deformation update is

$$
F_p^{n+1}=(I+\Delta t\,C_p)F_p^n.
$$

The simulator computes the Lamé parameters from Young's modulus $E$ and Poisson ratio $\nu$:

$$
\mu=\frac{E}{2(1+\nu)}, \qquad
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}.
$$

With polar decomposition $F_p=RS$, determinant $J=\det(F_p)$, and viscosity $\eta$, its stress-like term is

$$
P = \underbrace{2\mu(F_p-R)F_p^T+\lambda J(J-1)I}_{\text{corotated elastic contribution}}
+\underbrace{\eta(C_p+C_p^T)}_{\text{viscous contribution}}.
$$

For quadratic B-spline interpolation weight $w_{pi}$ between particle $p$ and grid node $i$, particle mass $m_p$, particle volume $V_p$, grid spacing $\Delta x$, and particle-to-node offset $d_{pi}$, the particle-to-grid update uses

$$
m_i \mathrel{+}= w_{pi}m_p,
\qquad
m_i v_i \mathrel{+}=w_{pi}\left[m_pv_p+
\left(-\Delta t\,V_p\,4\Delta x^{-2}P+m_pC_p\right)d_{pi}\right].
$$

The node velocity is normalized by its accumulated mass and updated with gravity. Grid velocities are then interpolated back to particles to update $v_p$ and $C_p$. If plasticity is enabled, $F_p=U\Sigma V^T$ is decomposed by singular value decomposition and each singular value is clamped to the configured interval $[\sigma_{\min},\sigma_{\max}]$ before rebuilding $F_p$.

For tool contact, the velocity at particle position $x$ is

$$
v_{\mathrm{tool}}(x)=v_{\mathrm{linear}}+\omega\times(x-c),
$$

where $c$ is the tool centre and $\omega$ is angular velocity. The simulator forms relative velocity $u=v_p-v_{\mathrm{tool}}(x)$, removes an inward normal component when $u\cdot n<0$, then applies the configured friction, absorption, and stickiness. A post-advection projection also moves particles out of the floor or tool volume when penetration remains.

A constant-height floor and two moving tools provide contact. For recorded replay, tools can use calibrated oriented boxes or signed-distance fields generated from the supplied UR and Kinova spatula meshes. The contact calculation accounts for both the linear and angular velocity of each tool. Recorded positions are interpolated linearly and orientations use shortest-path spherical interpolation, allowing tool states to be evaluated at simulation time. The simulator also produces rendered depth maps, occlusion-aware virtual point clouds, video, GUI output, UDP observations, and interfaces for Gymnasium and Stable-Baselines3 experiments.

# Related Work

TaichiDough builds on the Material Point Method (MPM), a hybrid particle-grid method that is well suited to large deformation and changing topology. The MPM formulation and transfer methods summarized by Jiang et al. provide the numerical basis for the MLS-MPM/APIC update used here, while the elasto-plastic MPM model of Stomakhin et al. established a practical computer-graphics treatment of yield behaviour through singular-value projection. Taichi provides the data-oriented, parallel programming system used to implement the simulator on CPU or GPU. More recently, EMPM (Embodied MPM) couples differentiable MPM with multi-view RGB-D observations to estimate geometry and physical parameters of deformable objects. TaichiDough instead uses recorded point clouds and measured tool trajectories with visible-geometry losses; it currently performs a non-differentiable search over an effective Young's modulus. In food engineering, inverse finite-element identification has long been used to estimate dough rheology from measured deformation. Fabbri and Cevoli identified semolina-dough parameters by minimizing disagreement between experiment and simulation with Levenberg-Marquardt optimization. This prior work motivates fitting to observations, but the present project evaluates a trajectory-conditioned MPM model with held-out deformation windows and does not claim that one fitted modulus is a universal dough property.

# Evaluation and Calibration Attempts

The static Episode 18 benchmark evaluates the reconstructed initial state under the bootstrap calibration. It contains 10,463 filtered observed points and compares the visible simulated and recorded geometry without translating, scaling, or registering either point set. The footprint intersection-over-union is 0.8808. The simulated-to-reference extent ratios are 1.058 and 1.052, the area ratio is 1.113, centroid offsets are below 0.5 mm on both top-view axes, and the 95th-percentile depth error is 2.0 mm. These results support the static visible-shape initialization, but they do not establish the accuracy of the material model.

The longer dynamic replay gives a more demanding result. It replays captured trajectories over 66 timestamp-paired observations using two identical 5 cm half-extent proxy boxes, a 3,000-particle simulation, a 24-cubed grid, a 0.2 ms timestep, and the default Young's modulus of 2,000 Pa. Across the 65 post-initial frames, mean pixel intersection-over-union is 0.4240, mean depth mean-absolute error is 95.95 mm, mean 95th-percentile depth error is 119.40 mm, and mean depth-change error is 83.83 mm.

The dynamic result explicitly records that physical fidelity is not validated. Its known limitations include unmeasured proxy tool geometry, an unverified camera calibration, no tool occlusion in the depth export, assumed hidden volume and material parameters, and visible-surface-only evaluation. It is therefore a replay and diagnostic result, not evidence of a calibrated physical prediction. A short implementation-smoke replay has substantially better early-frame agreement, but its limited duration only demonstrates that the replay path operates correctly.

# Material Identification and Next Steps

TaichiDough includes a reproducible procedure for identifying an effective Young's modulus. Each candidate begins from the same reconstructed particles, replays the same tool trajectory, and varies only Young's modulus. Candidate runs are compared using visible depth change, mask overlap, observed-to-simulated distance, and visible coverage. The search uses a logarithmic coarse grid, local refinement, boundary expansion, cached complete evaluations, and bootstrap intervals over deformation windows. Training windows select candidates, while held-out windows assess prediction without refitting.

No completed material-calibration output is included in this checkout. When obtained, the selected value must be reported as an effective Young's modulus for the selected constitutive model, viscosity, plasticity, contact method, reconstructed volume, mass assumption, and numerical resolution; it is not a universal constant for dough. The next experimental priorities are to collect a verified AprilTag metric calibration, measure the tool geometry and marker transforms, use measured dough mass, run held-out material calibration, and repeat the fit at several particle counts, grid resolutions, and timesteps.

# Conclusion

TaichiDough provides an end-to-end basis for data-conditioned dough simulation: calibration, floor-aware reconstruction, mass-preserving MPM initialization, timestamped two-tool replay, visible-geometry comparison, and material-parameter search. The static benchmark and replay implementation are established. The dynamic proxy result also makes the current limitations measurable and identifies the measurements needed before claiming physical agreement.
