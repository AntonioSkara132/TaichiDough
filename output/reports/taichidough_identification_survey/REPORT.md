---
title: "Dough Parameter Identification"
subtitle: "Depth recordings, tool trajectories, and the scientific case for TaichiDough"
author: "Critical literature survey and independent technical assessment"
date: "11 September 2026"
lang: en-GB
bibliography: references.json
link-citations: true
reference-section-title: "References"
geometry:
  - a4paper
  - margin=24mm
fontsize: 11pt
---

# Executive assessment

**Your idea is a legitimate scientific research programme:** use depth recordings and known tool trajectories to calibrate a model of dough's elastic, viscous, and plastic response, then ask whether that model predicts new manipulations. Forward parameter sweeps are a valid way to perform the inference. A differentiable simulator is neither a scientific requirement nor a prerequisite for a journal paper.

The broad idea is not new. Visual inverse simulation, robotic material identification, and inverse dough rheometry each have substantial prior work. The strongest opportunity is therefore **not** to claim the first camera-based dough simulator or the first MPM calibration. It is to answer a more precise question:

> Which elastic, rate-dependent, and non-recoverable behaviours can one calibrated depth camera and known two-tool motion distinguish in real dough—and which inferred parameter combinations remain predictive outside the calibration motions?

That question connects robotics, inverse problems, continuum simulation, and food rheology. A carefully designed study can contribute new empirical knowledge, a useful calibration method, or a benchmark without inventing a new MPM solver. The system becomes scientifically persuasive when its assumptions, limitations, and advantages are measured, not merely described.

## The strongest contribution to pursue

The recommended main contribution is an **observation- and motion-aware calibration protocol**. It would:

1. distinguish recoverable deformation, rate dependence, and persistent deformation through deliberately chosen loading, release, and recovery motions;
2. estimate only the parameters or parameter combinations that the observations constrain;
3. account for uncertainty in hidden volume, camera/tool registration, contact, and numerical discretisation;
4. test continuous predictions on complete, previously unused tool trajectories;
5. establish what changes when an extra view, an independent mass measurement, or a force trace is available.

The novelty would be the demonstrated result of that protocol: for example, that particular short probe sequences distinguish mechanisms that arbitrary shaping recordings cannot; that a single view is sufficient within a quantified operating range; or that apparently well-fitted material values actually depend on numerical contact settings. Each is a testable contribution. None is an established result of TaichiDough yet.

## What the fresh code assessment changes

This report uses the working tree at revision `cb76169` (full identifier in the evidence appendix), rather than treating the earlier conversation or report as evidence. It finds a substantive replay, reconstruction, evaluation, and forward-search implementation. It also finds three matters that affect interpretation:

- **The completed material search estimates only an effective elastic modulus.** Viscosity is fixed to zero and plastic projection is disabled in that experiment. Separate viscosity and contact-retention sweeps do not establish joint elastic–viscous–plastic identification.
- **The previous held-out comparison was wrong.** The default-modulus loss of 0.7997 is a *training* score. The recorded held-out comparison is 0.8281 for the selected modulus versus 0.8359 for a frozen-state baseline, approximately 0.93% lower. There is no stored default-modulus held-out score in that result. This small improvement has no uncertainty estimate and is not cross-trajectory validation.
- **An affine-transfer scaling inconsistency is present in the current solver.** An isolated algebraic test shows that its grid-to-particle expression reproduces an affine velocity gradient as $hA$, rather than $A$, where $h$ is grid spacing. Because that quantity drives deformation, viscous stress, and plastic activation, material interpretation requires resolving the inconsistency and rerunning calibration. No simulator code was changed for this report.

These findings do not make the research idea unscientific. They separate *verification of the numerical implementation* from *validation of a model of real dough*. Both are necessary, and they answer different questions.

# Scope, evidence, and how to read this survey

## What this review covers

The review brings together five bodies of work:

- visual inverse simulation and physical parameter identification;
- robotic manipulation and prediction of deformable materials;
- dough rheology and robotic or camera-assisted characterisation;
- MPM discretisation, constitutive modelling, and contact;
- identifiability, model discrepancy, uncertainty, and experimental design.

A recurring problem is that these literatures use the same words for different results. A paper may estimate a current shape, identify a physical parameter, fit a convenient simulator setting, or achieve a shaping task. All can be valuable, but they cannot be ranked using a single image error or the presence of automatic differentiation.

## Review method and limits

This is a **critical scoping review and research agenda**, not a registered systematic review or a proof of priority. Searches used combinations of dough, viscoelasticity, plasticity, depth/RGB-D, tool motion, robot manipulation, inverse simulation, system identification, contact, and parameter calibration. Exact-title follow-up used publisher records, proceedings, author manuscripts, DOI metadata, arXiv, and open full-text repositories. Topic evidence files retain search notes, access limitations, and page/section locations.

Technical claims are strongest when the relevant primary methods and results were inspected. Where only a publisher abstract or bibliographic record was available, the report limits itself accordingly. A feature marked *not established in the inspected material* is not claimed to be absent from the entire paper or codebase. Reviews are used for synthesis, not as proof of an uninspected experiment.

The report does not manufacture a screened-paper count, an acceptance probability, or a claim that no publication has ever combined a particular set of components. Existing numerical results were inspected, not independently reproduced by full simulation. The new affine-transfer check is a small algebraic calculation, explicitly distinguished from a Taichi rollout. Primary papers downloaded for reading are not bundled with the web companion.

## Four different achievements

| Achievement | What it establishes | Evidence it still needs for a stronger claim |
|---|---|---|
| Reconstruction or tracking | Agreement with currently observed geometry | Independent future prediction to establish dynamics |
| Effective simulator calibration | Parameters that improve a specified simulator's predictions | Numerical and experimental robustness before physical interpretation |
| Physical material identification | Mechanically interpretable parameters under stated conditions | Informative loading, uncertainty, and independent mechanical checks |
| Manipulation or control | Actions that achieve a target or task | Separate identification tests if claiming material recovery |

A good reconstruction is not a rheological measurement. A broad parameter uncertainty interval does not necessarily prevent accurate prediction. A successful controller may compensate for inaccurate physics through repeated observations. These distinctions are central to evaluating TaichiDough fairly [@review2020; @raue2011; @discrepancy2014].

# The inverse problem: what is being identified?

## A measurement-conditioned forward model

Let $z_t$ denote the simulated material state, $u_t$ the recorded tool poses and velocities, $\theta$ the material parameters, and $\psi$ uncertain non-material quantities. A useful formulation is

$$
z_{t+1}=\mathcal S_{h,\Delta t}(z_t,u_t;\theta,\psi),
\qquad
y_t=\mathcal H_{c,o}(z_t)+\delta_t+\epsilon_t.
$$

Here $h$ and $\Delta t$ describe numerical resolution; $\mathcal H$ projects the state through camera calibration $c$ and visibility/occlusion model $o$; $\delta$ is model discrepancy; and $\epsilon$ is measurement noise. The initial state must also be reconstructed or measured. The target is not simply a parameter vector fitted to an arbitrary point cloud: it is a model of the entire experiment and observation process.

A forward sweep estimates a parameter set by minimising a declared training objective,

$$
\widehat\theta\in\operatorname*{arg\,min}_{\theta\in\Theta}
\sum_{e\in\mathcal D_{\mathrm{train}}}w_e
\sum_{t\in\mathcal T_e}\ell\!\left(
\mathcal H(\mathcal S^{0:t}(z_{0,e},u_e;\theta,\psi)),y_{t,e}
\right).
$$

The optimiser can be a grid, a coarse-to-fine sweep, random or space-filling search, evolutionary search, Bayesian optimisation, finite-difference optimisation, or an adjoint method. These choices affect computational cost and exploration, not whether the inverse problem is scientifically meaningful. Finite-difference Levenberg–Marquardt is *not autodiff*, but it is not strictly derivative-free either. This distinction matters when comparing food-engineering and robotics methods.

## Separate parameter families

| Family | Examples | Why the distinction matters |
|---|---|---|
| Bulk material | $E$, $\nu$, viscosity $\eta$, relaxation times, yield/hardening parameters | These describe a specified constitutive law, not the word “dough” in general |
| Interface | Tangential slip law, adhesion, floor interaction | Incorrect interfaces can be compensated by apparent bulk stiffness or dissipation |
| Geometry and sensing | Hidden volume, mass/density, tool registration, camera pose, timing | These alter inferred strain, force scale, contact onset, and visual residuals |
| Numerical | Grid spacing, time step, particle count, transfer, SDF resolution, contact padding | A fitted value that changes with these settings may describe the numerical model rather than the material |

Treating all these quantities as “dough parameters” prevents meaningful physical interpretation. Conversely, holding them fixed does not remove their uncertainty. It makes every material estimate conditional on those choices [@kennedy2001; @discrepancy2014].

## A precise meaning of scientific value

A scientific framework states a hypothesis, makes assumptions explicit, produces reproducible predictions, and includes experiments that could contradict its claim. For TaichiDough, a useful claim is not “the animation looks realistic.” It is, for example:

> Under a specified recipe, sensor configuration, and contact model, parameters fitted from multi-rate pressing and recovery predict unseen dual-tool deformation more accurately than parameters fitted from a single monotonic push, under the same forward-simulation budget.

Failure to confirm that hypothesis can still provide useful knowledge if it reveals an observation limit, a constitutive-model failure, or a dependence on tool contact. An isolated failed software launch, by contrast, says nothing about rheology.

# Dough mechanics and constitutive interpretation

## Constitutive models define the parameters

Elasticity describes recoverable response. Viscosity describes stress associated with a deformation rate or flow. Viscoelasticity introduces a time-dependent combination of elastic and dissipative response, often with internal state or memory. Plasticity commonly describes irreversible deformation with a yield condition and loading history. Food papers sometimes use “viscoplastic” more broadly for lasting viscous flow. A manuscript must give the equation, not rely on the label.

For a scalar linear Kelvin–Voigt element,

$$
\sigma=E\varepsilon+\eta\dot\varepsilon.
$$

For a scalar Maxwell element,

$$
\dot\varepsilon=\frac{\dot\sigma}{E}+\frac{\sigma}{\eta}.
$$

They contain similarly named parameters but make different predictions. At held homogeneous strain, Kelvin–Voigt's rate term vanishes after the ramp; it does not produce a sustained Maxwell-like exponential stress relaxation. A standard-linear-solid, generalised Maxwell model, Burgers model, or fractional/power-law model has a different response and parameter interpretation. Large-strain dough may additionally exhibit strain hardening, structural change, rupture, and interface slip [@yazar2023; @ng2006; @sun2020].

The choice need not be the most complicated model. A restricted model can be scientifically useful if its operating range and failures are quantified. But adding more sweep values cannot make an incapable constitutive family reproduce missing physics.

## Real dough is not plasticine

Plasticine is a valuable reproducible proxy for robot shaping, but successful fitting on plasticine is not validation for flour dough. Flour formulation, hydration, mixing, temperature, resting, and previous handling affect response. Small-deformation stiffness need not predict large-deformation behaviour. A sample may become easier or harder to deform depending on preparation history; experiments themselves change it [@yazar2023].

The food literature already includes nonlinear rheology, formulation-dependent viscoelastic models, rate-independent hysteretic descriptions of mixing, and experiments/simulation of bread-dough rolling [@ng2006; @sun2020; @anderssen2013; @rolling2009]. These references prevent a claim that modelling dough with several mechanical parameters is itself new. Their exact constitutive domains also warn against importing fitted numbers from extrusion or starch-pasting tests into room-temperature robot manipulation.

## Residual deformation does not prove yield plasticity

A remaining deformation after a short wait could result from a yielded solid, a flowing dashpot, very slow recovery, structural damage, adhesion to the table, or gravity. A three-second recording cannot distinguish a permanent shape change from a recovery that takes much longer without additional evidence.

The fondant study by Cocuzza and Yan makes this issue concrete. It identifies a three-element spring–dashpot model, including a series dashpot, before evaluating gravity and tool-induced sheet deformation. The series dashpot produces non-recoverable strain without an explicit yield threshold. The study adds a different known-force recovery test when tensile data cannot determine the third parameter reliably [@cocuzza2018]. This is a useful precedent for *changing the experiment to expose a parameter*, rather than trusting a precise optimiser output.

For TaichiDough, plasticity experiments should include unloading, varied peak deformation, repeated cycles or matched fresh samples, and sufficiently long recovery. If the chosen model uses principal-stretch limits, report those dimensionless limits. Do not rename them a measured yield stress in pascals.

## Interfaces are part of the experiment

Dough can adhere, slip, and stretch during tool separation. Ghorbel and Launay compare different probe materials and hydration conditions and report rate-dependent adhesive separation behaviour [@adhesion2014]. Their result is not a contact law for TaichiDough, but it demonstrates why bulk viscosity and tool adhesion can be confused in a depth sequence.

Pressing, sliding, and withdrawal should be evaluated separately before a complex two-tool sequence is used to estimate everything simultaneously. Tool material, moisture, cleanliness, and contact geometry need documentation. A signed-distance field describes geometry; it does not select the correct normal, tangential, or adhesive response.

# Food-engineering and robotic rheometry precedents

## Inverse semolina-dough FEM

Fabbri and Cevoli use extrusion measurements and forward FEM to identify a power-law consistency index $k$ and flow index $n$ in $\tau=k\dot\gamma^n$ [@fabbri2015]. These are flow parameters, not simultaneous elasticity and plasticity estimates. Known piston/flow motion and die geometry are combined with measured force-derived pressure.

The inverse method uses Levenberg–Marquardt with finite-difference Jacobians. It compares its results against conventional capillary rheometry with several dies and repeated measurements. The reported disagreements, 3.6% for $k$ and 2.0% for $n$, compare characterisation procedures—not unseen robotic trajectories or camera prediction. The authors do not claim that agreement identifies which procedure is closer to physical truth.

**Relevance:** inverse simulation of dough and numerical parameter estimation are established. TaichiDough's possible distinction is partial visual observation, volumetric tool-conditioned prediction, and an explicit study of what the measurements identify. Independent mechanical comparison would substantially strengthen any claim to physical parameters.

## MIRANDA and RELAPP

Monleón-Getino and colleagues combine computer vision and robotic material testing, including flour-dough experiments [@miranda2025]. MIRANDA tracks visible landmarks and derives recovery descriptors; RELAPP supplies robotic deformation and force-related measurements. Regression links descriptors to laboratory flour-quality quantities such as alveograph strength and RVA final viscosity.

This is close enough that “first vision-and-robotics framework for dough characterisation” is not a defensible claim. It is also different from TaichiDough's proposed result: video recovery descriptors and regressed industrial indices are not a recovered three-dimensional continuum material law. RVA final viscosity is not interchangeable with a room-temperature continuum shear viscosity.

A crucial validation detail is explicit in the full article, §2.5: its regression data were **not split into training and test sets**. The reported fits therefore do not establish withheld-motion or independent-batch prediction. This creates a useful comparison question, not a reason to dismiss the work: can a mechanistic, trajectory-conditioned model make independently tested geometric predictions from similarly accessible measurements?

## Rheological robot-shaping and RGB-D tracking

Cocuzza and Yan identify a rheological model of fondant icing using tensile and known-force tests, then separately assess sheet deformation under gravity and a shaping tool [@cocuzza2018]. The article predates recent visual MPM systems and shows that robotic food modelling has an experimental identification history outside MPM.

Petit and colleagues use RGB-D sensing with non-rigid FEM-based tracking for a pizza-chef application [@pizza2017]. Tracking updates the state using current observations. A fitted model predicting an unseen future trajectory without those updates answers a different and stronger dynamics question. The proposed TaichiDough paper should state explicitly whether visual observations initialise, calibrate, continually correct, or only evaluate the simulation.

Together these studies suggest a productive contribution: connect the food literature's controlled preparation and mechanical tests to robotics' partial sensing and action-conditioned prediction. Hardware uniqueness alone is not the argument.

# Visual and action-conditioned inverse simulation

## DPSI: the closest elastoplastic MPM identification precedent

Yang, Ji, and Lai's DPSI directly addresses physics-based system identification for robotic manipulation of elastoplastic materials [@dpsi2025]. A KUKA robot executes known tool trajectories on plasticine. A Zivid camera is moved to six viewpoints before and after each manipulation, and the resulting point clouds are fused. This is **one physical camera but multi-view endpoint observation**, not a continuous fixed-view depth sequence.

The simulator combines fixed-corotated elasticity, von Mises plasticity, MLS-MPM/APIC, and rigid-tool SDF contact. Its lower-contact-complexity experiment fits $E$, $\nu$, yield stress, and density; its higher-contact-complexity experiment also fits table and manipulator friction. Viscosity is not an explicit parameter in this stated model. Missing underside geometry is completed under a table-contact assumption. The tools are coated with flour to reduce adhesion, making the interface preparation part of the experiment rather than an incidental detail.

DPSI compares endpoint Chamfer and Earth Mover's Distance objectives against observed or volume-filled targets, and uses Adam. It evaluates a heightmap error separately. The heightmap metric is a summed residual, not mean depth error in millimetres. Its reported best parameters are selected using in-distribution validation evaluated during optimisation; that validation should not be described as an untouched final test.

Crucially, DPSI already tests six examples of longer, unseen tool motions, including rolling, repeated poking, and rotation. Those tests provide genuine action-transfer evidence, although fitted parameters do not uniformly outperform handpicked ones. The paper also reports flat loss regions, different low-loss parameter combinations, incomplete elastic recovery, and contact parameters compensating for material-model error.

**Implication for TaichiDough:** known tool motion, incomplete volumetric reconstruction, elastoplastic MPM fitting, contact calibration, and unseen-motion evaluation are established. A useful advance would be to show what continuous depth adds beyond endpoints, and whether rate-dependent and persistent response can be separated in *real flour dough*. Merely replacing Adam with a sweep would not establish that advance.

## EMPM: already bimanual and already bread dough

Chen and colleagues' EMPM uses three RealSense D455 cameras, action-conditioned MPM, offline human demonstrations, and online experiments with two Franka arms [@empm2026]. Its materials explicitly include bread dough. It is therefore a direct counterexample to a broad claim that bimanual, visually calibrated MPM for dough is unexplored.

The formulation includes fixed-corotated elasticity, von Mises plasticity, and Coulomb table/gripper friction. Its general parameter notation lists $E$, $\nu$, density, and yield stress. However, the experimental implementation explicitly states that it optimises homogeneous **Young's modulus and Poisson's ratio**. A general parameter vector is not evidence that every entry was jointly identified in the reported bread-dough experiment. An explicit viscosity parameter is not part of the listed formulation.

Offline fitting uses point-cloud and tracked-point discrepancies with AdamW. The paper also explicitly provides **CMA-ES on the forward simulator**. This is an especially relevant precedent: derivative-free embodied MPM identification is already a legitimate published option, not a scientific deficiency requiring TaichiDough to implement autodiff.

Online fitting omits persistent object tracks and updates near quasi-static states, comparing short rollouts with observed geometry. Its bread-dough results measure alignment with and without ongoing correction. Such adaptation can be useful for robotics, but it is different from freezing a fitted material law and predicting an independent deformation rate or new trajectory. The inspected version does not establish that latter dough result.

**Implication:** TaichiDough should compare its proposed one-view measurement requirement, time-resolved loss, explicitly defined rate/plastic parameters, and frozen-parameter predictions with EMPM's actual experiment. Neither “two arms,” “bread dough,” nor “forward optimisation” alone establishes novelty.

## DiffCal and DiffCloud: depth is already an identification input

Arnavaz and colleagues' *Differentiable Depth*, implemented in DiffCal, uses one Intel L515 depth camera with known silicone-specimen geometry and clamp conditions [@diffcal2023]. A stable neo-Hookean FEM with strain-rate damping is fitted directly against rendered-versus-observed depth images. Poisson's ratio is fixed. One dynamic experiment jointly fits **Young's modulus, damping, and density from twelve depth images at 30 Hz**, following synchronised support removal.

This is clear prior evidence for single-depth-camera elastic/dissipative calibration. It is also instructive about physical interpretation: the paper explicitly treats fitted quantities as parameters of the selected discrete simulator, and coarse/fine discretisations produce different stiffness and damping estimates. Its damping coefficient should not be renamed a dough shear viscosity without matching the equation and units. Known reference geometry, short dynamic sequences, and largely fitting-oriented evaluation distinguish it from arbitrary dough with changing contact and hidden initial volume.

DiffCloud uses two calibrated RealSense cameras, known initial cloth geometry and grasp location, and recorded robot motion [@diffcloud2022]. It fits stiffness and mass multipliers in a differentiable thin-shell simulator. Its real experiments commonly minimise a one-way squared point-distance loss at one selected informative frame rather than supervise every frame. Folding also uses a visibility-related sampling heuristic.

DiffCloud demonstrates useful real observation alignment. A target sequence that is subsequently fitted is not an independent frozen-parameter prediction test, and such a real cross-trajectory test was not located in the inspected version. Its selected-frame approach is nevertheless a relevant baseline for TaichiDough: **does the extra information in intermediate depth observations actually improve parameter discrimination and future prediction?** That is an experiment, not an assumption that more frames must help.

## Earlier mechanical identification: Wang and Hahn

Wang and colleagues combine three Kinect cameras, a high-quality initial scan, fixed boundaries, and dynamic release under gravity to estimate soft-object models [@wang2015]. Corotated FEM uses $E$, $\nu$, and Rayleigh damping parameters $\alpha$ and $\beta$. Their material-fitting subproblem uses **gradient-free Nelder–Mead**, within an alternating procedure that also estimates tracking and reference geometry. Calling the material fit derivative-free is accurate; calling every stage gradient-free would not be.

Separate static loading and dynamic comparisons provide stronger validation than fitting the original images alone. The paper also recognises discretisation-dependent apparent stiffness, flat or multimodal objectives, and the limitations of 30 Hz sampling for damping. It demonstrates that visual material calibration with derivative-free optimisation and mechanical validation predates recent differentiable MPM frameworks.

Hahn and colleagues use ten-camera marker motion capture, known geometry, and measured clamp motion to identify neo-Hookean elasticity with power-law viscosity [@hahn2019]. Sensitivity/adjoint gradients and L-BFGS fit Lamé parameters, a viscosity coefficient, and a rate exponent. This is not a markerless depth method, but it is central to the *mechanics* of the proposed problem.

Their experiments show why loading diversity matters. Narrow-frequency motion can poorly constrain a damping law; changing clamp conditions and combining motions improves estimation; large bending or snapping can expose constitutive-model limitations. Independent rheometer measurements use different strain/frequency conditions and are not automatically exact ground truth for the manipulation experiment.

Together, these studies support a stronger research argument than simply showing an optimiser converges: demonstrate informative excitation, disclose numerical dependence, and validate under physically different conditions.

## PAC-NeRF and GIC: distinguish synthetic scope from real evidence

PAC-NeRF identifies continuum-model parameters from posed multi-view RGB videos, using differentiable MPM coupled to a radiance-field representation [@pacnerf2023]. Its synthetic studies include elastic, plastic, Newtonian, granular, and non-Newtonian viscoplastic laws. The last includes shear/bulk response, yield stress, and plastic viscosity. Therefore a visual inverse problem containing elastic, viscous, and plastic parameters is already represented in the literature.

The real demonstration is much narrower: a falling deformable ball recorded by four cameras, using **RGB rather than their depth measurements**. Synthetic recovery under known parameter-generating models and qualitative real reconstruction should not be described as experimentally validated joint rheological identification of manipulated dough.

GIC uses multi-view Gaussian reconstruction to construct geometry and infer continuum parameters through temporal point-distance and mask losses [@gic2024]. Synthetic experiments likewise cover multiple constitutive families, including viscoplasticity. A future-frame benchmark tests same-sequence continuation. For sparse-view real data, the described variant fits masks rather than using the complete dynamic reconstruction objective. Real grasp demonstrations add useful application evidence, including transfer of inferred properties to a different simulation framework, but they are not matched-condition rheometry.

These systems are strong prior art for visual continuum identification. They also demonstrate why a survey should separate **what a formulation permits**, **what synthetic experiments recover**, and **what real experiments establish**. TaichiDough's possible contribution is the latter: an experimentally supported result about rate, recovery, and residual deformation in real dough under known tools and limited sensing.

## Learned constitutive laws and action-conditioned twins

MASIV learns neural elasticity and plasticity mappings, rather than recovering a compact vector of $E$, viscosity, and yield coefficients [@masiv2025]. Multi-view reconstructed trajectories and silhouettes supervise differentiable MPM. Initialisation and mechanical priors constrain the learned functions; interior trajectories are inferred rather than directly observed. It is a relevant alternative when a fixed constitutive family is too restrictive, but the scientific target differs from interpretable parameter calibration. Its main-paper evidence includes synthetic prediction and qualitative transfer, not new real-dough rheometry.

PhysTwin records one- and two-hand interactions with three RGB-D cameras and fits a spring–mass model with stiffness, damping, collision, and control-interaction parameters [@phystwin2025]. Zero-order initial fitting is followed by gradient refinement. A single-image geometry prior does not make the complete identification method single-camera.

PhysTwin separately reports reconstruction, future-frame continuation, and unseen interactions. In the latter, the fitted physical model is reused but registered to the new episode's initial observation. This is meaningful transfer evidence with an explicit initial-state requirement. It shows that bimanual visual fitting and new-interaction evaluation are established outside MPM as well. The distinction for TaichiDough must concern the dough law, measurements, or demonstrated result—not just two moving effectors.

## Forward and hybrid fitting are established options

Matl and colleagues identify granular contact coefficients using depth-derived pile statistics and BayesSim likelihood-free inference [@matl2020]. No simulator gradients are required. Repeated calibration pours at one height are followed by different pour heights and robotic tasks, with uncertainty and model mismatch examined. These are granular rather than dough parameters, but the study is a strong methodological precedent for forward-only calibration aimed at predictive macroscopic behaviour.

Yoon and Lim combine Bayesian optimisation with later gradient descent for robot-manipulated cloth [@yoon2025]. Their experiments include held-out draping and specimen-size transfer. The accessible publisher text supports a **hybrid** method, not a purely derivative-free one, and comparisons also change simulation resolution. It should not be cited as isolated evidence that a particular optimiser is superior.

For TaichiDough, method choice can be practical and transparent: a small joint grid or space-filling search, selective refinement, and a plausible-parameter ensemble. Compare its cost and predictive quality under equal budgets if claiming an inference advantage. A scientific identification study can instead keep the optimiser conventional and make the experimental finding its main contribution.

## Evidence matrix: compare questions, not score magnitudes

The following table summarises the primary distinctions. It does not rank performance across incompatible tasks or declare uninspected capabilities absent.

| Study | What the inspected real experiment identifies or fits | What makes its validation different |
|---|---|---|
| DPSI | Plasticine elastic/yield/density and, in a separate contact level, interface friction; multi-view endpoints | Validation used in parameter selection; additional unseen longer actions |
| EMPM | Stated experimental $E,\nu$ fit; three views; bread dough and two arms | Online alignment while adapting is not frozen-parameter dough transfer |
| DiffCal | Single-view silicone stiffness/damping/density with known geometry | Strong discretisation/fitting study; independent dynamic transfer not located |
| DiffCloud | Cloth stiffness/mass multipliers from two views and robot motion | Selected-frame fitting; independent real frozen-parameter transfer not located |
| Wang et al. | Elasticity and Rayleigh damping from tracked 3D dynamics | Separate static loading and dynamic checks; derivative-free material fit |
| Hahn et al. | Elasticity and power-law viscosity from marker motion | Different motions/clamps reveal both information gains and model limits |
| PAC-NeRF / GIC | Real evidence narrower than synthetic multi-law parameter coverage | Keep rendering, synthetic recovery, temporal continuation, and grasp tasks separate |
| PhysTwin | Effective spring/damping/contact model from three-view hand interactions | Future-frame and separate interaction transfer; new initial state aligned |
| Fabbri–Cevoli | Semolina flow consistency and exponent from extrusion pressure | Independent rheometry comparison, not camera-guided action transfer |
| MIRANDA / RELAPP | Recovery/force descriptors and regressed flour-quality indices | Explicitly no regression train/test split |
| Proposed TaichiDough study | A defined effective dough law from continuous single-view depth and recorded tools | Must demonstrate independent prediction and uncertainty; not yet achieved |

Three comparisons are particularly valuable for a TaichiDough paper: **endpoint versus time-resolved observation**; **arbitrary shaping versus designed rate/recovery probes**; and **fit quality versus independent frozen-parameter prediction**. These comparisons provide testable questions that remain meaningful even when the optimiser and simulator are established methods.

# Numerical foundations and what MPM contributes

## MLS-MPM, APIC, and a material law are different choices

The original MLS-MPM paper derives a moving-least-squares formulation and efficient stress-divergence computation; its CPIC extension addresses displacement discontinuities and two-way rigid coupling [@mls2018]. APIC describes the locally affine velocity representation used in particle–grid transfers [@apic2015]. Neither name specifies the material's rheology.

MPM is attractive for large deformation because material state is carried by particles and computation uses a background grid rather than a permanently deforming body-fitted mesh. That advantage does not automatically supply physically correct cutting, fracture, thin-tool contact, or topology changes. Those require additional formulations and verification.

The snow MPM model is important because it uses corotated elasticity, principal-stretch plastic projection, and plastic-volume-dependent hardening [@snow2013]. Reusing related mechanisms for dough can be a useful approximation, but the parameters inherit the meaning of that particular law. A snow-style dimensionless compression limit is not a direct rheometer yield-stress measurement.

## Contact geometry versus contact mechanics

An SDF provides signed distance and, where well-defined, a normal. It can support detecting penetration and projecting velocity or position. It does not define compliance, restitution, slip, adhesion, or the relationship between normal and tangential forces.

For example, ILS-MPM explicitly specifies unilateral contact, normal/tangential penalty treatment, and Coulomb sticking/sliding for deformable particulate contact [@ils2020]. Its evolving level-set formulation is not identical to sampling a static rigid-tool SDF. It is relevant prior art for contact, but comparison should name the shared geometric idea and the different mechanical and numerical treatments.

For physical Coulomb friction, tangential traction is bounded relative to normal traction. A multiplicative reduction of velocity without normal-force dependence is a different model. It can be calibrated as an effective numerical response, but the fitted coefficient must not be interpreted as a measured Coulomb coefficient.

## Numerical dissipation can imitate viscosity

PIC/APIC transfer, grid resolution, time integration, projection, and repeated contact damping can all affect dissipation. If a per-contact multiplier $r$ is applied each time step, its repeated effect scales like $r^{T/\Delta t}$ in a simplified continuous-contact example. Its equivalent decay rate is

$$
\gamma=-\frac{\log r}{\Delta t}.
$$

This is not a constitutive viscosity law. It shows why an unchanged numerical contact parameter need not represent unchanged dissipation after a time-step change. In the actual simulator, contact activation and multiple update stages make the dependence more complicated still.

Likewise, two padding studies answer different questions: holding padding fixed in metres while refining the grid, or holding padding/grid-spacing fixed so its physical thickness changes. A paper should report both quantities and distinguish these experiments. Material and prediction robustness must be assessed under numerical changes that preserve mass, initial volume, tool motion, and the intended physical contact assumptions.

## Differentiability is an inference option

ChainQueen establishes differentiable MLS-MPM for soft-robot applications, while DiffTaichi supplies an efficient differentiable programming framework [@chainqueen2019; @difftaichi2020]. These are foundational references for adjoint-based optimisation, not evidence that every scientifically useful MPM study must use an adjoint.

A derivative-free search can handle a small number of parameters transparently and work with discontinuous contact or image operations. It still suffers from expensive evaluations, parameter coupling, and ambiguous data. Gradients can accelerate search, but do not create missing information or guarantee the correct constitutive model. Both approaches need validation against observations not used for fitting.

# Prediction, manipulation, and benchmark literature

## Learned dynamics is a different scientific target

RoboCraft reconstructs particle representations from RGB-D, learns graph dynamics, and uses predictive control for elastoplastic shaping [@robocraft2022]. RoboCook adds diverse tools and long-horizon real dough tasks, with transfer demonstrations to other materials [@robocook2023]. **Real flour-dough robotic manipulation is therefore not an unexplored application.**

Their value as comparators depends on the claim. If TaichiDough claims more accurate forward dynamics, a learned model trained on the same observations is useful. If it claims interpretable material parameters or a measurement protocol, an appropriate mechanical/inverse-calibration comparison may be more important. A complete controller is not mandatory for a narrowly framed identification paper.

RoboCook's CEM+MPM versus learned-model/planner comparisons change more than just calibration. They should not be read as proof that a carefully calibrated MPM model inherently loses to learned dynamics. Nor should a TaichiDough static reconstruction IoU be compared to a final target-shaping IoU as if they measured the same task.

## Synthetic and single-view prediction benchmarks

PlasticineLab supplies differentiable elastoplastic manipulation tasks and is useful for controlled synthetic identification experiments [@plasticinelab2021]. Recovering parameters from the exact model that generated the observations is a valuable implementation test, but it can be an *inverse crime*: the test omits the model mismatch that makes physical inference difficult. Add noisy sensing, imperfect registration, altered contact, and ideally a different observation or forward generator.

DoughNet predicts geometry and topology from a single RGB-D observation and tool actions [@doughnet2024]. It does not establish physical parameter identification in the inspected project description, but it prevents treating single-view input alone as a new capability. TaichiDough's distinction would be the identifiability, physical interpretation, numerical robustness, or experimental repeatability it demonstrates.

A useful future benchmark should document more than many depth frames. It should include independent specimens, preparation metadata, calibration, tool motion, contact conditions, recovery durations, split definitions, and reference measurements. This would connect robotics datasets to food-rheology reproducibility.

# What depth and tool motion can identify

## A simple counterexample to absolute-stiffness identification

Consider homogeneous small-strain quasistatic elasticity, fixed Poisson ratio, no body forces, prescribed displacement on part of the boundary, and zero traction on the remainder:

$$
\nabla\cdot\left[E\,\overline{\mathsf C}(\nu):\varepsilon(u)\right]=0.
$$

For any constant $a>0$, replacing $E$ with $aE$ leaves the displacement solution unchanged after dividing equilibrium by $a$. Reaction forces change, but a depth camera observing displacement does not measure them. Therefore **known tool displacement does not, by itself, guarantee identification of the absolute stiffness scale**.

This is an analytical example with stated assumptions, not a theorem that all depth-based dough calibration fails. Known gravity, inertia with independently constrained mass/density, nonzero measured loads, or force sensing can provide an absolute scale. Whether they provide a strong enough signal in a particular recording is an experimental question.

If density is unknown as well, combinations such as $E/\rho$ can be easier to infer from dynamics than $E$ and $\rho$ separately. Supplied mass helps, but density computed as mass divided by reconstructed volume inherits the uncertainty in hidden geometry.

## A camera may identify timescales better than separate constants

For the idealised scalar Kelvin–Voigt element during unloaded recovery,

$$
\eta\dot\varepsilon+E\varepsilon=0,
\qquad
\varepsilon(t)=\varepsilon(0)\exp[-tE/\eta].
$$

This recovery identifies the ratio $\eta/E$ if the assumptions hold. It does not necessarily identify both absolute parameters. More complex spatial deformation, inertia, gravity, or additional loads can add information, but the lesson remains: look for identifiable *combinations*, not only individual coefficients.

A fixed-displacement hold is especially easy to misinterpret. If the visible geometry is held still while force relaxes, depth alone cannot directly measure the stress-relaxation curve. A load cell or independent force measurement is needed for that direct measurement. Lateral motion or subsequent release can offer indirect information, but it should not be described as a measured force-relaxation experiment.

## Structural identifiability, practical sensitivity, and prediction

Structural identifiability concerns what ideal observations and the assumed model can uniquely determine. Practical identifiability concerns finite, noisy, incomplete measurements. A flat sampled loss curve is evidence of weak sensitivity under the chosen experiment and objective; it is not by itself a proof of structural non-identifiability [@raue2009; @raue2011].

There is also **predictive identifiability**: different parameter combinations may predict the required output almost identically. A broad parameter set can therefore support accurate forecasts for one action class, yet diverge under another. The correct response is to propagate the plausible set through the intended held-out actions, not automatically discard the framework because one modulus is uncertain [@raue2011].

Model discrepancy further complicates interpretation. Missing adhesion, wrong hidden volume, or numerical errors can be absorbed into fitted mechanical parameters. A narrow numerical optimum is not proof of physical truth. Explicit discrepancy models can help but also introduce further confounding; independent information is needed to separate those effects [@kennedy2001; @discrepancy2014].

## Conditional sweeps are not joint parameter profiles

A curve $L(E,\eta_0,p_0)$ varies elasticity while keeping viscosity and plasticity fixed. It measures a conditional slice. A profile instead permits the others to change:

$$
L_{\mathrm{profile}}(E)=\min_{\eta,p,\psi}L(E,\eta,p,\psi).
$$

The latter can reveal compensating combinations hidden by one-at-a-time sweeps. A joint grid, space-filling search, or repeated conditional optimisation can approximate a profile without autodiff. A set defined as “within 2% of the best geometric loss” is a *loss-tolerance set*, not a 95% confidence interval. Formal intervals need a defensible likelihood/noise model or an empirically justified resampling procedure.

With one physical trajectory, resampling the same complete trajectory cannot estimate between-trajectory uncertainty. Thousands of adjacent depth frames are not thousands of independent experiments. The sampling unit must match the claim: trajectory, specimen, batch, or preparation day.

## Information-aware motion selection without gradients

One possible method contribution is to choose a small set of motions that produces distinguishable outcomes across candidate models. Forward simulations can estimate feature sensitivities:

$$
J_{jk}\approx\frac{f_j(\theta^{(k,+)})-f_j(\theta^{(k,-)})}
{2\epsilon\,s_j},
$$

where $\theta^{(k,\pm)}$ equals $\theta$ except that its positive $k$th parameter is multiplied by $e^{\pm\epsilon}$. Here $f_j$ is a measured trajectory feature and $s_j$ a noise or scale estimate. Nearly parallel columns suggest confounded effects; small columns suggest poor signal. This approximation needs checks across perturbation sizes and contact regimes. It is a diagnostic, not a guarantee of global identifiability or a calibrated confidence interval.

Alternatively, no derivative estimate is required: simulate a plausible parameter ensemble and select motions where their predicted observable histories disagree most, subject to safe and repeatable contact. Raue and colleagues provide the general model-based experiment-design rationale [@raue2011]. The proposed TaichiDough contribution would be a concrete real-dough demonstration with matched acquisition and computation budgets, not a claim to have invented experiment design.

# TaichiDough: current implementation and scientific interpretation

## What is implemented

The audited code supplies much of the infrastructure needed for the proposed study:

- metric-camera and floor-plane representations;
- single-view volumetric reconstruction by filling from observed points toward the floor;
- particle initialisation that preserves the reconstructed volume and supplied mass when changing particle count;
- two-tool replay with timestamp checks, translation interpolation, quaternion SLERP, and angular contact-point velocity;
- box, no-tool, and generated-solid SDF collision options;
- visible-depth, mask, coverage, and point-distance evaluation without post-hoc per-frame registration or time warping;
- coarse-to-fine modulus search, invalid-candidate handling, fingerprints, and declared training/validation windows;
- separate tool-retention and viscosity sweep runners.

These are implemented capabilities, not yet a demonstration that all three material mechanisms are recoverable. Calibration records refer to metric-v2 transformations, but a schema check is not an independent metrology error estimate. Generated watertight solids are usable collision geometry, but their topology tests do not independently verify physical tool dimensions or marker registration.

## The actual material approximation

For deformation state $F$, polar rotation $R$, and $J=\det F$, the code uses

$$
\tau_e=2\mu(F-R)F^T+\lambda J(J-1)I,
\qquad
\mu=\frac{E}{2(1+\nu)},\quad
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}.
$$

This is a corotated stress-like quantity $PF^T$, not the first Piola stress itself. When plastic-volume hardening is enabled, both Lamé coefficients are additionally multiplied by $\exp[b(1-J_p)]$, where $b$ is the hardening setting. The code adds

$$
\tau_v=\eta(C+C^T),
$$

where $C$ is intended to represent an affine velocity gradient. It is an instantaneous rate addition, with no generalised Maxwell branch or relaxation spectrum. At finite deformation, directly adding a rate term to $PF^T$ requires a stated stress convention; a spatial Cauchy viscosity and a Kirchhoff viscosity are not automatically interchangeable.

Optional plastic projection clips the principal stretches into `plastic_min`/`plastic_max`. Optional `Jp` stores a scalar plastic-volume-related history for hardening, not a full general plastic state. The limits are dimensionless. The flag `--pure-viscoelastic` disables this projection: sweeping plastic bounds while retaining that flag would provide no evidence about plasticity.

Source locations: `scripts/taichi_viscoelastic_mpm_scene.py:1268–1303`, `1397–1407`, and `2013–2039`. The corresponding reading of snow-style plasticity should use its explicit stretch/energy equations rather than assume a yield-stress model [@snow2013].

## Verified affine-transfer inconsistency

For an interior quadratic B-spline stencil with physical offsets $d_i=x_i-x_p$,

$$
\sum_iw_i d_i=0,
\qquad
\sum_iw_i d_i d_i^T=\frac{h^2}{4}I.
$$

An affine grid velocity $v_i=b+Ad_i$ should be reconstructed by

$$
C=\frac{4}{h^2}\sum_iw_i v_i d_i^T=A.
$$

The audited source instead uses physical `dpos=(offset-fx)*dx` with coefficient `4*inv_dx`, which produces $C_{\mathrm{code}}=hA$. The coefficient $4/h$ is appropriate if the offset is dimensionless; it is inconsistent with the physical offset used here.

An isolated NumPy reproduction of the exact weights and expression confirms the issue:

| Grid resolution | Input gradient component | Reconstructed component in current expression |
|---|---|---|
| 24 | $3\ \mathrm{s}^{-1}$ | 0.125 |
| 48 | $3\ \mathrm{s}^{-1}$ | 0.0625 |
| 96 | $3\ \mathrm{s}^{-1}$ | 0.03125 |

The corrected physical-offset formula reproduces $A$ to floating-point precision in the same calculation. This is an algebraic verification, not a measured dough result or a full Taichi regression test. Source: `scripts/taichi_viscoelastic_mpm_scene.py:1374–1378`; reproducible check: `evidence/check_affine_transfer.py`.

Because $C$ drives the update $(I+\Delta tC)F$, the viscous term, and plastic activation, the discrepancy affects more than a cosmetic naming convention. The stored modulus experiment's simulator hash matches the audited source. **Resolve and test this before interpreting fitted SI parameters or collecting an expensive calibration dataset.** A correction changes dynamics and requires new fits; dividing an old fitted modulus by grid resolution is not a justified substitute. The source code has been left unchanged because this task is an assessment, not an implementation request.

## Contact “friction” is velocity retention

After removing inward normal relative velocity, the tool update multiplies the remaining relative velocity by

$$
r(1-a)(1-s),
$$

where $r$ is named `tool_contact_friction`, $a$ is absorption, and $s$ is stickiness. This scales outward normal as well as tangential relative components and has no normal-traction-based Coulomb bound. Lower $r$ means more attenuation; $r=0$ makes the corrected particle velocity follow the tool. This direction is not the interpretation commonly attached to increasing a physical friction coefficient.

The appropriate current description is **effective contact-velocity retention**. Contact geometry, retention, absorption, stickiness, grid/particle projections, and floor interaction can compensate for viscosity in a visual loss. Source: `scripts/taichi_viscoelastic_mpm_scene.py:1335–1343` and the particle-contact functions. The SDF checks remain useful, but they validate signed geometry, not a physical friction law.

## Hidden volume and observation assumptions

The floor-fill reconstruction assumes that the unobserved volume between the visible top and floor is occupied. It does not measure hidden cavities, folded undersides, or detached material. Supplied object mass divided by this volume determines density. A wrong volume therefore changes both geometry and mechanical scale. In the inspected modulus run, the supplied 0.25 kg and reconstructed volume of approximately $8.8317\times10^{-4}$ m³ imply about 283 kg/m³. That is a derived simulation input, not an independent density measurement; the mass and reconstruction should be checked against the actual specimen.

The current objective averages depth-change, mask-IoU loss, observed-to-simulated visible-point distance, and real-coverage loss. These are dimensionless normalised components. Their aggregate is not a depth error in metres. Dough-only rendering omits tool occlusion, and the depth-change term uses the intersection of valid initial/current masks. That can exclude newly exposed or disappearing areas; support and visibility need separate reporting.

These are reasonable starting assumptions for a restricted task, provided their effects are measured. Single-view efficiency becomes a contribution only when the accuracy and uncertainty costs of these assumptions are quantified.

## What the stored experiments actually show

The available modulus experiment uses 24,000 particles, grid 48, $\Delta t=0.0002$ s, SDF resolution 64, contact padding $1/384$ m, supplied mass 0.25 kg, fixed $\nu=0.3$, zero viscosity, and disabled plasticity. It selects $E=130{,}579.320726$ Pa on source frames 1–60, but reports `needs_review` and `accepted: false` because its sampled near-minimum region is broad.

| Setting | Training frames 1–60 | Later frames 61–97 |
|---|---:|---:|
| Selected modulus | 0.361059 | 0.828075 |
| Default $E=2000$ Pa | 0.799745 | Not stored in this result |
| Frozen initial state | 0.343191 | 0.835860 |

All entries are the same declared aggregate loss; lower is better. The frozen-state baseline is slightly better on training than the selected simulation. The selected simulation is about 0.93% better on the later window than frozen state, without repetitions or a confidence interval. These values do not establish a unique modulus or transfer to a new action. They also **do not** support the earlier claim that the fitted model is worse than default $E$ on held-out data: that claim incorrectly compared different splits.

Ten sampled modulus values between approximately 91.4 and 186.5 kPa fall within the configured 2% loss tolerance. This is weak practical discrimination for that objective and experiment, not proof of universal non-identifiability. Its whole-window bootstrap has one training window; all resamples repeat it, so its zero-width interval provides no between-trial precision evidence.

The newer `_local` contact-retention sweep completed, unlike the older failed directory. Its training losses are 0.359191, 0.361067, and 0.361463 for $r=0.1,0.2,0.3$. The smallest tested value gives the smallest training score, with only about 0.52% improvement relative to 0.2. There is no held-out stage or uncertainty estimate. This is a completed sensitivity result, not a physical-friction measurement.

A viscosity runner now tests 0, 1, 2.5, 5, and 10 under the parameter's nominal Pa·s interpretation. The available `viscosity_sweep2` results failed during CUDA initialisation with `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`. They are startup failures, not evidence that the viscosities are physically wrong or numerically unstable. No successful joint elasticity–viscosity–plasticity fit was found.

Current metric-v2 static records show a close initial visible-depth match, but they use the observation from which the initial state was reconstructed. Older long proxy-tool rollouts are diagnostic evidence for their own configurations, not a valid measure of current solid-SDF accuracy. The evidence appendix identifies the files and split locations.

# A falsifiable experimental programme

## Start with the claim, not all available knobs

The first study should target **predictive calibration of a restricted material model**, not simultaneous recovery of every rheological constant. State the preparation, deformation range, loading speeds, observation duration, tools, and required predictions. Decide whether the aim is a batch-specific model, transfer between specimens of one recipe, or transfer between recipes. These are different claims.

A sensible initial model comparison is:

- elastic-only;
- elastic plus a specified rate-dependent term;
- elastic plus a specified plastic mechanism;
- the combined model, if the observations justify its extra parameters.

Retain added complexity only when it improves independent prediction or explains a mechanically verified effect. More parameters and a lower training loss do not establish better science.

## Motion primitives and what they measure

| Motion or test | Useful information | Key confound or limitation |
|---|---|---|
| Small press, unload, and release | Reversible response; contact geometry; recovery | Displacement alone may not fix absolute stiffness |
| Same path at multiple speeds | Rate dependence and candidate timescales | Inertia, contact, timing, and specimen history also change response |
| Fixed-displacement hold | Force relaxation if force is recorded | Nearly fixed visible geometry may contain little information |
| Known-force creep and recovery | Compliance, retarded response, lasting flow | A position-controlled robot is not a constant-force apparatus |
| Load–unload at several amplitudes | Onset of persistent deformation; model selection | Slow recovery, damage, and adhesion can imitate plasticity |
| Gravity sag or slump | Known body-force scale and contact-light response | Requires reliable mass/geometry and visible deformation above noise |
| Controlled slide and withdrawal | Interface slip and adhesion | Contact damping and bulk dissipation remain coupled |
| New two-tool sequence | Practical action-transfer prediction | Must remain untouched during model and parameter selection |

Use a pilot to choose speeds, amplitudes, and recovery durations. Values should resolve the material response above sensor noise while avoiding unmodelled rupture, uncontrolled adhesion, or tool-tracking errors. A numerical list chosen merely because its values are convenient is not a justified physical search range.

One specific warning follows from the current nominal settings: $E\approx130.6$ kPa and $\nu=0.3$ give $\mu\approx50.2$ kPa. Ratios $\eta/\mu$ for $\eta=1$–10 Pa·s are approximately 20–200 microseconds, compared with a 200-microsecond simulation step and roughly 33-millisecond recorded frame spacing. This is only a dimensional diagnostic; it is not the exact recovery time of the full model. It does show why the viscosity range needs an argument based on observable response times, especially after the affine-transfer inconsistency is resolved.

## Stage 1: verify the numerical and measurement chain

Before parameter interpretation:

1. add affine-field reproduction and rigid-motion tests for transfer;
2. verify mass/volume invariance across particle counts and grids;
3. test simple elastic, rate-dependent, and plastic cases with known expected behaviour;
4. check camera projection, floor coordinates, tool pose conventions, interpolation, and timing independently;
5. quantify tool-to-marker and camera-to-scene uncertainty;
6. test collision solids and padding without confusing topological validity with physical alignment;
7. test parameter recovery under the actual depth/visibility operator, then deliberately introduce noise and model mismatch.

Report failures separately as launch/environment failures, numerical failures, missing-observation cases, and physically poor predictions. Only the last group measures predictive inadequacy. A verification test does not replace physical validation, but it prevents a search from compensating for a coding error.

## Stage 2: fit a small, justified joint parameter set

Select bounds from physical response scales, reference tests, and pilot sensitivity. Positive scale parameters often benefit from logarithmic sampling; dimensionless stretch bounds require their own physically meaningful intervals. Represent zero viscosity as an explicit model case rather than taking its logarithm.

A forward-only protocol can use:

1. a coarse joint grid or space-filling set;
2. evaluation of every candidate under identical trajectories, numerical settings, and visibility rules;
3. refinement around several promising regions, not only the first apparent minimum;
4. pairwise loss plots and approximate profiles to identify compensation;
5. propagation of all plausible parameter sets to validation predictions;
6. a final locked evaluation on untouched trials.

A grid with five levels in three parameters has 125 combinations before repeats, numerical-sensitivity studies, or multiple specimens. This arithmetic is a compute-planning example, not a proposed universal budget. Compare sampling strategies under equal forward-evaluation and physical-data budgets. Record failed evaluations, warm-up/compilation, simulator time, evaluator time, and total wall time.

One-at-a-time sweeps remain useful diagnostics. They should not be presented as evidence that elasticity, viscosity, and plasticity were jointly separated. Similarly, selecting a contact parameter after examining a former holdout turns that holdout into development data; a new final test is then needed.

## Stage 3: separate three kinds of validation

**Temporal forecast:** fit an early interval and propagate continuously from the initial state into later frames. Do not reset the state at the boundary. This tests prediction under the same specimen and trajectory history.

**Cross-trajectory prediction:** fit on one set of motions and predict entirely different recorded motions without refitting. Vary speed, orientation, indentation, or tool coordination to test the claimed operating range. This is the central evidence for useful material calibration.

**Across-specimen or batch transfer:** test independently prepared samples or days, with recipe and preparation controlled. A batch-specific calibration paper does not need to promise universal recipe transfer, but it must state that limit. If recipe transfer is claimed, evaluate it explicitly.

An extra camera or a geometric scan can test hidden-volume assumptions. A force trace or conventional material test can test the mechanical scale. Those measurements may be reserved for evaluation while the operational estimator uses one camera. That makes the low-sensor claim testable rather than assumed.

## Stage 4: use metrics that reveal different errors

Report camera-space depth error with valid support, silhouette IoU, bidirectional coverage, and symmetric visible-point distance or a thresholded F-score. Add geometry features relevant to the task: height, extent, centroid, spread, or recovered thickness. Force/torque error belongs in the report when force is available and mechanically relevant.

Always distinguish:

- current-state reconstruction from future prediction;
- visible geometry from complete volume;
- optimised loss from independent evaluation metrics;
- frame-wise curves from episode-level uncertainty;
- a mean error from the coverage on which it was computed.

Do not perform per-frame alignment or time warping during final prediction evaluation. If input registration is uncertain, calibrate it independently or analyse its uncertainty, rather than aligning away the prediction error. Include error versus forecast horizon and contact phase; a final-frame value can conceal wrong intermediate dynamics.

At minimum compare a frozen initial state, unfitted/default mechanics, and the selected model. For a contact claim, compare calibrated box/solid-SDF/no-tool cases under otherwise matched settings. For a multi-parameter claim, compare nested material models. A learned baseline is important for a learned-versus-physics performance claim, but not automatically mandatory for a focused identification experiment.

## Stage 5: uncertainty and numerical transfer

Use the physical trajectory, specimen, or batch as the resampling unit. Choose sample size after a pilot establishes variability and the smallest practically important paired improvement; this report does not prescribe an arbitrary number as a journal requirement. Repeats of interpolation frames do not increase the independent sample count.

Report plausible parameter sets and their predicted outcome spread. Test camera/tool perturbations, hidden-volume alternatives, contact-retention assumptions, SDF resolution, particle count, grid spacing, and time step. Preserve physical mass and geometry when changing resolution. Separate physical-padding-fixed and grid-relative-padding-fixed tests.

Two outcomes can both be valuable:

- stable parameters and improved independent predictions support physical/effective identification within the stated conditions;
- unstable individual parameters but stable useful predictions support a narrower predictive calibration claim.

If both parameter values and predictions change materially under plausible numerical or measurement choices, report that limitation. More printed decimal places do not resolve it.

# Where a defensible novel contribution could lie

## Novelty is a result, not a list of components

The existing pieces—depth cameras, recorded tool poses, MPM, SDFs, loss minimisation, and parameter sweeps—are established. Their combination can still enable a meaningful contribution if it answers a question the closest studies do not answer in the same experimental conditions. A narrow hardware conjunction is weak evidence of novelty; a measured advantage or a carefully established limitation is much stronger.

The following are **prospective contributions**, ranked here by their fit to the proposed framework rather than claimed uniqueness in the entire literature.

| Candidate contribution | Testable claim | Evidence that would distinguish it from an implementation report |
|---|---|---|
| Motion-dependent identification | Designed rate/recovery/amplitude probes distinguish mechanisms better than arbitrary shaping recordings | Matched-budget comparison, joint parameter sensitivity, and untouched action-transfer predictions |
| Single-view measurement sufficiency | One view provides useful predictions within a defined operating range | Independent-view or volume checks; quantify what extra sensing changes |
| Material/interface separation | Independent contact/timing characterisation reduces false viscosity or plasticity estimates | Controlled contact perturbations and improved cross-tool or cross-motion transfer |
| Numerical robustness of inferred response | Some parameter combinations or predictions remain stable after refinement | Correct transfer tests and refits across grid/time-step/particle/contact settings |
| Real-dough benchmark | Controlled repeated specimens connect rheology to robot prediction | Released preparations, calibrations, depth, trajectories, optional forces, and locked evaluation splits |

None requires a new adjoint solver. A more capable constitutive model or a new search strategy could be an additional contribution, but would require separate comparison. It is better to demonstrate one principal contribution clearly than to claim all five on one recording.

## Recommended primary hypothesis

**H1:** Multi-rate loading and sufficiently long release/recovery observations provide more useful constraints on a restricted elastic–viscous–plastic model than a single monotonic deformation, leading to better prediction on new two-tool trajectories.

Support would require improvement over a matched-data, matched-compute baseline, reproducible across independent trials, with the added model mechanisms actually enabled and numerically verified. H1 is weakened if the apparent improvement disappears after registration/contact uncertainty is included, if the inferred response changes with time step, or if multiple model classes remain observationally equivalent on new motions.

A complementary hypothesis is that **not all individual parameters need be recoverable**: a narrower set of parameter combinations may suffice for a particular prediction task. Report this as a positive predictive result with a bounded physical claim, not as complete rheological recovery.

## Strong and weak manuscript claims

**A strong claim, if experimentally supported:**

> We establish when single-view depth and recorded dual-tool kinematics are sufficient to calibrate a compact model of real dough response. A forward-only search over explicitly defined elastic, rate, and residual-deformation parameters is evaluated on independent tool trajectories, with contact, hidden-volume, and numerical sensitivity quantified.

**A strong narrower claim, if parameters remain ambiguous:**

> Several mechanically different parameter sets fit the observed manipulations. We identify which additional motions or measurements distinguish them, and which predictions remain reliable despite that ambiguity.

**Claims not supported by the present evidence:** first visual material identification; first robotic dough rheometry; a novel MLS-MPM method; physical Coulomb-friction measurement from the current retention coefficient; joint rheological identification from the completed elasticity-only run; or universal dough constants.

“An effective parameter for a specified model and preparation” is not a concession that makes the work unpublishable. It is an accurate statement of the inverse problem. It becomes valuable when the resulting predictions, uncertainty, or limitations are useful to other researchers.

# Journal suitability and the scientific case

## Yes, the framework can support a journal paper

A scientific paper does not have to introduce a new numerical solver. Official RA-L scope includes innovative ideas, theoretical findings, application results, and case studies. It does not require gradients or a new optimiser [@ral-scope]. RA-P explicitly accommodates practical investigations, reproducible advances, and case studies [@rap-scope]. Research-software significance can also include substantial engineering rather than algorithmic novelty, although JOSS has its own maturity, research-use, licensing, documentation, and testing requirements [@joss-scope].

These scope statements are not acceptance predictions. They show why “uses known methods” and “uses sweeps” are not sufficient reasons to reject the idea. Reviewers will still ask what was learned, how it differs from close prior work, whether experiments test the claims, and whether another group can reproduce the result.

## Choose the paper type to match the evidence

| Paper type | Plausible main contribution | What must exist before submission |
|---|---|---|
| Robotics method/experimental paper | Partial-observation calibration with demonstrated action-transfer or information-aware probing | Verified implementation, closest-method comparison, independent physical tests, uncertainty |
| Applied robotics systems paper | Reliable real-dough measurement/replay/calibration procedure with useful practical findings | Repeated real experiments, specified calibration, failure analysis, reproducible release |
| Food-engineering/rheology paper | Mechanically interpretable response under controlled preparation and loading | Independent force/rheometry reference, correct constitutive interpretation, batch/repeatability evidence |
| Simulation/computational-methods paper | New discretisation, contact, constitutive or inference treatment | A genuinely new method plus accuracy/stability/efficiency evaluation |
| Dataset or scientific-software paper | A reusable calibrated resource or research tool | Data rights/license, documented protocol, working release, reproducible use beyond one bespoke run |

For the proposed framework, a focused robotics experimental or applied-systems article is the most natural direction. A food-rheology article becomes more plausible with independent mechanical measurements. A numerical-methods paper requires a numerical contribution; merely implementing MPM is not one.

A top-level submission should not depend on a long feature list—GUI, network adapters, reinforcement-learning interfaces—unless those features are used in an experiment. They may belong in documentation. The paper's space belongs to the inference problem, measurements, model, uncertainty, and evidence.

## What makes this survey itself useful

Broad deformable-object surveys already organise representations, models, estimation, and control [@review2020]. Food reviews already organise dough rheometry and nonlinear behaviour [@yazar2023]. A useful survey at their intersection should organise studies by **parameter, observation, and excitation**, rather than by simulator brand or optimiser alone.

This report's proposed synthesis asks:

1. What physical or effective quantity is estimated?
2. What information is measured—shape, motion, force, or a task outcome?
3. What loading and history make that quantity observable?
4. Which sensor, contact, geometry, and numerical assumptions remain fixed?
5. What independent prediction or physical measurement validates the result?

That taxonomy makes meaningful comparisons possible between an extrusion FEM inverse problem, a point-cloud MPM fit, a video-recovery descriptor, and a learned robot dynamics model. It also makes misleading comparisons easier to identify. To submit this document itself as a formal review article would require a more reproducible database/search protocol, systematic inclusion/exclusion accounting, complete bibliographic checking, and author-led critical revision. This report is a substantial starting survey, not a claim that those review-publication requirements have already been met.

# Recommended research sequence

## Immediate verification and clarification

1. Resolve the affine-transfer inconsistency and add an affine-reproduction regression test; rerun affected calibration.
2. Define the intended constitutive law and units. Decide whether the first paper estimates effective rate and stretch-limit parameters or physically interpretable rheological constants.
3. Keep elasticity-only, viscosity, plasticity, and combined-model experiments distinct. Ensure parameter switches actually activate the tested mechanism.
4. Rename or clearly document the contact-retention coefficient so it cannot be mistaken for Coulomb friction.
5. Preserve current logs, exact splits, and source hashes. Do not rewrite failed or superseded experiments as successful evidence.

## First physical study

Use controlled pressing/release, multiple speeds, and several amplitudes on independently prepared specimens. Preserve the depth and pose time bases, initial mass and volume assumptions, tool calibration, and preparation history. Use an independent force or geometric reference on a subset if possible. Compare conditional sweeps with a small joint search and propagate plausible parameters into unseen motions.

The first result should answer a limited question well: for example, whether release and recovery observations distinguish elastic-only from rate-dependent or plastic models better than pressing alone. It need not recover every parameter simultaneously to be publishable.

## Full framework study

Extend only after the basic identification tests work: cross-trajectory two-tool prediction, independent specimens/batches, interface perturbations, numerical refits, and a packaged benchmark. Report both successful and ambiguous cases. Publish the operational single-camera pipeline separately from any evaluation-only force/extra-camera measurements so the claimed sensing requirement remains clear.

## Suggested reading order

Read by the question each paper can answer, rather than attempting the whole bibliography in date order.

| Priority | Paper or pair | What to extract for your study |
|---|---|---|
| 1 | DPSI [@dpsi2025] | Closest elastoplastic MPM calibration; contact preparation; endpoint fitting; unseen actions and failures |
| 2 | EMPM [@empm2026] | Existing bimanual bread-dough work; actual fitted parameters; adaptive alignment versus prediction |
| 3 | DiffCal [@diffcal2023] and Hahn et al. [@hahn2019] | What dynamic visual observations constrain; damping definitions; motion diversity and numerical dependence |
| 4 | Yazar [@yazar2023] | Dough preparation, nonlinear response, appropriate loading, recovery and interface conditions |
| 5 | Fabbri–Cevoli [@fabbri2015] and Cocuzza–Yan [@cocuzza2018] | Independent mechanical checks; change the experiment when a parameter remains unconstrained |
| 6 | Raue et al. [@raue2011] | Conditional slices versus profiles; uncertain parameters versus reliable predictions |
| 7 | MLS-MPM and snow MPM [@mls2018; @snow2013] | Transfer/stress equations and the precise meaning of stretch-based plasticity |
| 8 | PAC-NeRF and GIC [@pacnerf2023; @gic2024] | Joint viscoplastic parameterisation; synthetic versus real evidence |
| 9 | Matl et al. [@matl2020] and Wang et al. [@wang2015] | Forward-only material fitting, uncertainty and independent validation |
| 10 | MIRANDA/RELAPP, RoboCook and PhysTwin [@miranda2025; @robocook2023; @phystwin2025] | Adjacent food measurement, practical shaping, and new-interaction prediction |

The reading catalogue records a primary link, the source version inspected, and whether access was full, partial, abstract-level, or metadata-only. That distinction is especially important before copying equations, reported parameter values, or evaluation claims into a manuscript.

# Evidence appendix

## Current repository records

The audit is tied to revision `cb76169436d334c127f165ce5c3f779100bb82d4`. Relative paths below avoid dependence on a particular machine's home directory.

| Record | What was checked |
|---|---|
| `scripts/taichi_viscoelastic_mpm_scene.py:1268–1407` | Material terms, plastic switches, transfers, and contact/velocity updates |
| `scripts/deformpath_dynamics.py:390–516` | Timestamped two-tool interpolation and velocities |
| `scripts/reconstruct_voxel_dough_from_deformpath.py:223–372` | Floor-fill and hidden-volume assumptions |
| `scripts/material_calibration.py:164–332` | Normalised loss, support, and validity rules |
| `scripts/calibrate_youngs_modulus.py:1184–1379` | Search, selection, split handling, and reporting |
| `data/single_episode_calibration/episode18_kugla_fixed_collision_11_9/material_calibration_result.json` | Correct training/held-out/default/frozen-state comparisons |
| `output/episode18_e130579_dx8_tool_friction_sweep_local/friction_sweep.json` | Completed training-only retention sweep |
| `output/episode18_e130579_dx8_viscosity_sweep2/viscosity_sweep.json` | Failed startup cases; stderr identifies CUDA initialisation failure |

In the large material-result JSON, selection/flat-minimum/one-window bootstrap are at lines 227–289; the held-out aggregate is at 86830–86837; the default comparison explicitly carries the training split at 93862–93899. The full local evidence note contains additional locations and interpretation. Numeric reports under `data/` and external recorded inputs require an archive or download procedure; source code alone is not a complete reproduction package.

## What was and was not executed

Executed for this assessment: read-only source/result inspection, primary-literature retrieval, bibliography preparation, the isolated affine-weight calculation, and document builds. **Not executed:** a new material calibration, a new robot experiment, a full Taichi regression suite, or an independent reproduction of the stored real-data results. No simulator or sweep source files were modified.

The accompanying evidence files distinguish full-text inspection, abstract-level evidence, and metadata-only reading leads. The bibliography identifies the particular editions cited; related conference/journal titles and online/issue years may differ. The older report remains a separate file and should not be used for its erroneous mixed-split comparison.

## Literature access and version qualifications

| Source group | Scientific text inspected and limit |
|---|---|
| DPSI; EMPM | Full arXiv v3 (February 2025) and v1 (January 2026), respectively; final journal publication metadata checked separately |
| PAC-NeRF; GIC; DiffCloud | Full arXiv versions v1 (2023), v3 (2024), and v2 (2025); DiffCloud's conference publication is 2022 |
| PhysTwin; MASIV | CVF accepted main papers; separate supplements not read |
| DiffCal; Cocuzza–Yan | Full early-online/accepted author manuscripts; final issue metadata checked separately |
| Wang; Hahn; Matl; Fabbri; MIRANDA; Yazar; RoboCook | Full papers or relevant primary full-text methods/results inspected |
| MLS-MPM; snow MPM; ILS-MPM; Raue 2011; Arriola-Rios review | Relevant primary full-text numerical/methodological sections inspected |
| APIC; RoboCraft; PlasticineLab; DoughNet; ChainQueen; DiffTaichi | Primary abstracts/project descriptions and publication records; detailed results not inferred from those alone |
| Raue 2009; Kennedy–O'Hagan; Brynjarsdóttir–O'Hagan; Anderssen–Kružík; Ghorbel–Launay | Abstract-level evidence and metadata; used for general methodological or physical context |
| Ng; Sun; Mitsoulis–Hatzikiriakos; Petit | Metadata/discovery excerpts; reading leads and broad subject scope, not quantitative validation evidence |
| Yoon–Lim | Partial publisher-text access through §5.4.2; complete limitations unavailable |

The source-version qualifications apply to the technical descriptions throughout this report. In particular, final publication metadata do not prove that a preprint's wording is identical to the published version. Detailed locators and search queries are in `evidence/visual_inverse.md`, `evidence/dough_rheology.md`, and `evidence/methods_evaluation.md`. The CSV catalogue provides primary links and access levels for each reference.

## Bottom line

**The scientific opportunity is real, but it is an experimental and inverse-problem question—not a novelty claim about using Taichi or parameter sweeps.** A framework that establishes when depth and tool kinematics can distinguish dough response, quantifies what remains ambiguous, and predicts new manipulations would be a credible contribution. The strongest report of that work will connect numerical verification, measurement physics, food preparation, and independent prediction in one transparent argument.
