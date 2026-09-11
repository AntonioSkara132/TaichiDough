# Visual and action-conditioned material identification

**Evidence review date:** 11 September 2026.  
**Scope:** identification of deformable-object simulation parameters from visual/depth measurements and known motion or boundary conditions, with particular attention to the proposed TaichiDough experiment: one depth camera, two moving tools, real dough, and forward-simulation fitting of elastic, viscous, and plastic response.  
**Research status:** targeted primary-literature review, not an exhaustive systematic review. Twelve targeted web searches were followed by primary-paper reading and bibliographic verification. Eleven available full papers were read; one additional article was examined through an incomplete publisher-text extraction. No simulator code was audited for this evidence note.

## 1. Main findings

The proposed inverse problem is scientifically legitimate, but its broad ingredients have strong precedents. A useful contribution must establish what can be identified and predicted for real dough under the proposed observation and contact conditions, rather than claim novelty from assembling depth sensing, recorded tool motion, MPM, and parameter optimization.

1. **DPSI is the closest direct precedent for action-conditioned elastoplastic MPM calibration from real 3D observations.** It uses a KUKA robot, endpoint point clouds, known tool trajectories, fixed-corotated elasticity, von Mises plasticity, and frictional contact. Its lower-contact-complexity experiment fits four parameters: Young's modulus, Poisson's ratio, yield stress, and density. Its higher-contact-complexity experiment also fits table and manipulator friction. It evaluates longer, unseen tool motions. Its main material is plasticine, not bread dough. [DPSI, §§4–5 below.]
2. **EMPM already includes real bread dough and two robot arms.** It uses three RGB-D cameras and an action-conditioned MPM simulator. Its formal parameter set contains Young's modulus, Poisson's ratio, density, and yield stress, but the experimental implementation explicitly optimizes homogeneous Young's modulus and Poisson's ratio. The paper also explicitly offers **CMA-ES on the forward simulator**. Its bread-dough online results measure alignment while adapting, not a demonstrated frozen-parameter, independent cross-rate dough prediction test. [EMPM.]
3. **One depth camera plus dynamic elastic/viscous parameter fitting is already demonstrated by Differentiable Depth / DiffCal.** A single Intel L515 observes silicone specimens; a dynamic experiment jointly fits Young's modulus, damping, and density from twelve depth images. Thus neither single-view depth nor adding a viscosity-related parameter is, by itself, a defensible broad novelty claim. [DIFFCAL.]
4. **Visual estimation of elastic, yield, and viscous parameters already appears in synthetic MPM studies.** PAC-NeRF and GIC include non-Newtonian/viscoplastic parameter sets containing shear modulus, bulk modulus, yield stress, and plastic viscosity. Their synthetic coverage must be distinguished from their much narrower real-data evidence. [PACNERF; GIC.]
5. **Derivative-free calibration is an established scientific method.** Wang et al. use Nelder–Mead for the material-parameter subproblem; Matl et al. use likelihood-free Bayesian inference over forward simulations; EMPM provides CMA-ES; PhysTwin and Yoon–Lim use hybrid zero-order/global and gradient-based fitting. A forward sweep can be appropriate without being a new optimization algorithm. [WANG; MATL; EMPM; PHYSTWIN; YOON.]
6. **Low observation error does not establish unique or physically exact parameters.** Several papers demonstrate model-resolution dependence, parameter coupling, alternative low-loss solutions, or failures on changed deformation modes. Optimizer convergence, observed-geometry alignment, future prediction, and intrinsic material-property measurement are distinct claims. [DPSI; DIFFCAL; WANG; HAHN; MATL.]
7. **The most promising contribution is an experimentally demonstrated identification result for real dough under restricted sensing.** It could quantify whether continuous single-view depth and carefully chosen dual-tool motions separate recoverable elastic deformation, rate-dependent dissipation, and permanent deformation, after accounting for contact and hidden initial geometry. Independent cross-trajectory and cross-rate prediction would be essential evidence. This is a research direction supported by the comparison, not an established unoccupied gap or a verified first claim.

## 2. Evidence policy and source versions

**FT** means the available full paper was read, including appendices present in that file. It does not imply that a separate supplement, code repository, or final publisher version was read. **PT** means partial publisher-text access. Bibliographic metadata and scientific-method evidence are recorded separately: a publisher DOI record can verify publication status without verifying a method described only in a preprint.

| Key | Source actually read | Access level and publication distinction |
|---|---|---|
| DPSI | `dpsi_full.pdf` / `dpsi_full.txt`; arXiv:2411.00554 v3, 18 February 2025 | FT, including included appendices. Published IJRR 2025 metadata verified separately. Final publisher PDF not read. |
| EMPM | `empm_full.pdf` / `empm_full.txt`; arXiv:2601.17251 v1, 24 January 2026 | FT. Published RA-L 2026 volume, issue, pages and DOI verified separately. Final IEEE typeset version not read. |
| DIFFCLOUD | `diffcloud_full.pdf` / `diffcloud_full.txt`; arXiv:2204.03139 v2, 13 May 2025 | FT. Conference publication is IROS 2022; the downloaded revision is later. |
| DIFFCAL | `diffcal_full.pdf` / `diffcal_full.txt`; author-hosted Wiley early-online PDF | FT. Read version says 2022, volume 0, pages 1–14; final record is CGF 42(1), 277–289, 2023. |
| PACNERF | `pacnerf_full.pdf` / `pacnerf_full.txt`; arXiv:2303.05512 v1, 9 March 2023 | FT, including Appendix A. Published at ICLR 2023. |
| GIC | `gic_full.pdf` / `gic_full.txt`; arXiv:2406.14927 v3, 31 October 2024 | FT, 22-page arXiv version including appendices. Final NeurIPS proceedings pagination is different; final proceedings PDF not read. |
| MATL | `matl_full.pdf` / `matl_full.txt`; arXiv:2003.08032 v4, 5 November 2020 | FT. ICRA 2020 metadata verified separately. |
| WANG | `wang2015_full.pdf` / `wang2015_full.txt`; author-hosted SIGGRAPH/TOG 2015 PDF | FT, final-looking author paper. DOI and journal metadata verified separately. |
| HAHN | `hahn2019_full.pdf` / `hahn2019_full.txt`; ETH author-hosted TOG 2019 paper | FT. DOI and journal metadata verified separately. |
| PHYSTWIN | `phystwin_full.pdf` / `phystwin_full.txt`; CVF ICCV 2025 accepted main paper | FT of accepted main paper. Separate supplement not read. DOI not verified in this review. |
| MASIV | `masiv_full.pdf` / `masiv_full.txt`; CVF ICCV 2025 accepted main paper | FT of accepted main paper. Separate supplement not read. DOI verified separately. |
| YOON | Oxford Academic publisher-page extraction | PT, through §5.4.2 in the available extraction. Conclusion, full limitations, and some optimizer details were unavailable. No local full-paper PDF was read. |

All local line locators below refer to the accompanying `pdftotext -layout` files, not source-code lines. Printed section names, equations, tables, and figures are also given where useful. Two-column extraction can interleave nearby paragraphs. These locators make the claims checkable without treating PDF-extracted equations as authoritative typesetting.

## 3. Comparative overview

| Study | Actual observations and known inputs | Parameters/law demonstrated | Inference and main evaluation | Relevance to proposed TaichiDough work |
|---|---|---|---|---|
| DPSI | One Zivid camera moved to six viewpoints for each initial/final state; KUKA tool motion and geometry | Four material parameters at contact level 1; six including contact friction at level 2; fixed-corotated + von Mises MPM | Adam; endpoint CD/EMD fitting; selected using in-distribution validation; longer unseen tool motions | Direct precedent for elastoplastic MPM, tool conditioning, incomplete geometry, contact fitting, and unseen-motion evaluation; endpoint multi-view plasticine differs from continuous single-view dough |
| EMPM | Three D455 RGB-D cameras; tracked hands offline or two Franka grippers online | Formal E, ν, density, yield; experiments explicitly fit E and ν; fixed-corotated + von Mises MPM | AdamW; optional forward CMA-ES; offline geometry/tracks and online quasi-static correction | Already real bread dough and bimanual action-conditioned MPM; no explicit viscosity parameter in the listed formulation; joint E/viscosity/yield calibration is not demonstrated by the stated experimental parameter set |
| DiffCal | One L515; known specimen geometry, clamps, gravity, synchronized support removal | Stable neo-Hookean FEM; E, strain-rate damping, density in dynamic experiment; ν fixed | Autodiff + Adam; raw depth-image L2 fitting; model/discretization studies | Direct precedent for single-camera elastic/viscous dynamic calibration; silicone, known geometry, no plasticity in described experiment |
| DiffCloud | Two D435 cameras; robot motion, initial mesh, grasp location | Thin-shell simulation; stiffness and mass multipliers | Differentiable simulation/rendering + Adam; usually one selected target frame | Direct depth/action fitting precedent; primarily cloth and fitting accuracy, not independent real dough prediction |
| PAC-NeRF | Posed multi-view RGB, known rigid boundary geometry; initial velocity inferred | Several MPM laws; elastic, plastic, Newtonian, viscoplastic, granular parameters on synthetic data | Rendering loss; L-BFGS for initial velocity, Adam for material parameters | Joint visual viscoplastic inference already exists synthetically; real test is a falling ball using four cameras' RGB, not depth-guided tool manipulation |
| GIC | Multi-view RGB; known camera calibration and material family | Several MPM constitutive families, including viscoplastic parameters synthetically | Surface CD + mask L1; real sparse-view variant uses masks only; synthetic continuation and real grasp demonstrations | Strong visual MPM precedent; measured depth is not the real identification input described here |
| Wang et al. | Three Kinects for dynamics, Artec initial scan; fixed boundaries and released motion under gravity | Corotated FEM; E, ν, Rayleigh damping α and β; reference geometry also estimated | Nelder–Mead material fitting within alternating tracking/rest-state estimation; separate static/dynamic validation | Direct non-gradient elastic/damped-solid calibration precedent |
| Hahn et al. | Ten-camera marker mocap; known geometry and measured moving clamps | Neo-Hookean FEM + power-law viscosity; Lamé parameters, viscosity coefficient, exponent | Sensitivity/adjoint gradients + L-BFGS; cross-motion and material comparisons | Strong evidence on excitation, damping coupling, model dependence; not markerless/depth calibration |
| Matl et al. | Overhead D435 final piles; known funnel and pouring protocol; grain mass/size | Nonsmooth DEM; sliding friction, rolling friction, restitution | BayesSim forward-simulation posterior; independent pour heights and robotic pours | Direct likelihood-free depth calibration and uncertainty precedent; granular contact coefficients are not dough rheology |
| PhysTwin | Three D455 RGB-D cameras; one/two tracked hands; reconstructed geometry | Spring stiffness, dashpot/drag and collision/control parameters | Zero-order initialization + gradient refinement; 7:3 future-frame split and separate unseen interactions | Already sparse-view bimanual action-conditioned visual fitting and transfer; effective spring model, no explicit yield law identified in read main text |
| MASIV | Multi-view videos and inferred dense 3D trajectories; gravity | Neural elasticity/plasticity weights and initial velocity | Differentiable MPM, trajectory and silhouette losses; reconstruction, synthetic prediction, qualitative geometry transfer | Relevant alternative to fixed constitutive families; not direct recovery of E, viscosity, yield coefficients |
| Yoon–Lim | Two L515 cameras; UR10 trajectories; known cloth layout and grasp | Mass–spring–damper cloth; stretch, bend, damping, friction parameters | 50 BO iterations including 10 random, then 50 gradient epochs; held-out draping and sample-size transfer | Non-autodiff-only workflow is established; hybrid, not purely derivative-free |

## 4. Closest action-conditioned MPM studies

### 4.1 DPSI — Yang, Ji and Lai

**Bibliographic identity:** [DPSI] in §9. Published IJRR article; scientific details below were verified in the available arXiv v3 full text.

**Observation and input protocol.** The platform is a KUKA iiwa LBR 14 with a Zivid One+ medium camera and interchangeable rectangular, cylindrical-roller, and bullet/round end effectors. For each object state, the camera is moved to six poses and the point clouds are fused. Only the states before and after manipulation are observed; intermediate object deformation is not used as supervision. Thus this is physically one camera, but its inverse problem uses multi-view endpoint reconstruction, not one continuously fixed depth view. The main material is plasticine. Cloud slime and soil are additional generalization examples. The tool trajectories are reconstructed from time-stamped MoveIt trajectories, retaining motion duration and resampling at 0.01-second intervals. The paper does not justify treating that trajectory interval as the simulator's complete physical substep specification.

**Initial geometry and contact.** The unobserved underside is completed by projecting points toward the bottom, under an object/table contact-angle assumption of at least 90 degrees. A ball-pivoted watertight mesh is filled to create particles. Rigid tool SDFs are precomputed. SDF contact is handled at both grid and particle stages. The friction response depends on relative normal and tangential velocity, including a sticking threshold; it is not a constant tangential velocity multiplier. To prevent adhesion from dominating the experiment, the tools are coated with a thin layer of flour before each interaction. The fitted tool-friction coefficient therefore describes the flour-coated interface.

**Model and parameters actually fitted.** The simulator is Taichi/DiffTaichi MLS-MPM with APIC-style transfers, fixed-corotated elastic energy, and von Mises plasticity. The experimental distinction matters:

- **Contact level 1:** fit E, ν, yield stress σy, and density ρ using vertical pokes.
- **Contact level 2:** additionally fit table friction μt and manipulator friction μm using poke-and-shift motions; the three tool types are assumed to share the manipulator-friction coefficient.
- There is no explicit viscosity coefficient in this stated six-parameter model. That statement is about the described parameterization, not proof that every possible dissipation source is absent from the code.

**Objective and optimization.** Four endpoint losses are compared: Chamfer and Earth Mover's Distance, each against either the fused point cloud or a volume-filled reconstructed particle target. Although normalized point-set equations are introduced, the implementation omits averaging over point sets to preserve gradient magnitude. A separate 32×32 heightmap covers a 0.11×0.11 m² area. Its reported error is a **sum of scalar height residual norms**, not per-pixel mean depth error. Direct optimization of that heightmap objective performed poorly, so it is used for evaluation rather than main parameter fitting. Adam performs 100 updates for each loss/dataset pairing, with three random seeds. A training update averages parameter gradients across the fitting examples.

**What fitting, validation and unseen-motion results mean.** Each contact level has 12-example and 6-example mixed-tool fitting sets and three alternative one-example fitting sets. Each example is a before/after observation pair and its tool motion, not a continuous deformation sequence. In-distribution validation uses additional initial object configurations under the same motion family. The text reports twelve in-distribution validation datapoints across its described two-level collection; its wording should not be rewritten as twelve per level. Validation is evaluated after every gradient update, and the parameter values summarized as best are selected by the lowest validation heightmap error across seeds. This selection must be reported when interpreting the validation numbers.

The out-of-distribution set comprises six examples, two for each of three longer motions: roller flattening, triple poking, and a poke followed by 180-degree tool rotation. Their durations are approximately 3.74, 4.55, and 6.23 seconds, versus about 0.86–1.52 seconds for fitting motions. These are meaningful unseen-action tests. They are not uniform successes: handpicked parameters outperform the fitted alternatives in two of six object simulations, and the best in-distribution configuration does not consistently give the best out-of-distribution outcome.

**Failures and interpretation.** The paper reports flat loss regions, different low-loss parameter combinations, initialization-dependent minima, insufficient elastic recovery, sharp creases, overly local deformation, and floating particles. For materials poorly represented by its constitutive model, fitted contact parameters can compensate for material-law error. Its pairwise loss plots fix all remaining parameters; they are useful sensitivity illustrations but not full joint identifiability proofs. The preprint contains apparent equation/units inconsistencies, including suspect physical-unit headings in parameter tables. Do not copy the extracted Lamé equations or convert its table values into validated physical moduli without additional verification.

**Exact overlap with TaichiDough.** Known rigid-tool motion, volumetric particle reconstruction from incomplete 3D data, differentiable MPM, elastic and plastic response, contact-parameter fitting, SDF tools, rotation, endpoint geometric objectives, and unseen-motion testing are already present. Continuous single-view observation of real dough and explicit rate-dependent identification would be meaningful changes to the experiment, but their scientific value must be demonstrated. A different optimizer alone is insufficient.

**Evidence locators:** `dpsi_full.txt` lines 127–161, 277–313, 356–431; “Real-to-sim object reconstruction,” “Real-to-sim trajectory reconstruction,” lines 433–495 and Table 1; “Loss functions,” lines 498–582; “Experiment design,” lines 550–642; Figure 3 and lines 645–665; parameter selection lines 721–735; failure examples lines 1065–1135; unseen motions lines 1138–1279; material mismatch lines 1303–1329; initialization study lines 1331–1421; limitations lines 1525–1557; heightmap-optimization appendix lines 1761–1811. Four-versus-six fitted parameters are explicitly supported by lines 438–478 and the four/six parameter trajectories described at lines 724–725.

### 4.2 EMPM — Chen et al.

**Bibliographic identity:** [EMPM] in §9. Published RA-L metadata verified; method details read in the January 2026 arXiv v1.

**Observation and control.** Three RealSense D455 cameras provide multi-view RGB-D. Segmentation and depth back-projection produce fused point clouds. Offline demonstrations use human hands; hand motion and tracked object points are reconstructed from the videos. Online robotic experiments use two Franka arms with their original grippers and record gripper positions as control inputs. Bread dough, pita bread, plasticine, cloth, rope, and an elastic toy are included. The presence of **real bread dough and a bimanual robot setup is explicit**, not inferred from a figure caption about generic soft objects.

**Mechanics and actual fitted parameters.** Warp implements a differentiable MPM simulator with APIC transfers. Controller motion is imposed through grid Dirichlet boundary velocities. The paper explicitly specifies fixed-corotated elastic stress, multiplicative elastic/plastic decomposition with von Mises return mapping, and Coulomb table/gripper friction.

The formal offline parameter vector is θ = {E, ν, ρ, y}. However, the implementation section states: “We optimize the Young’s modulus E and Poisson’s ratio ν of the objects and assume the parameters remain constant accross the material field.” The experiments therefore support **homogeneous E and ν optimization**, not an unqualified claim of demonstrated joint E, ν, density and yield identification. No explicit viscosity parameter is included in that listed formulation. The paper does not provide evidence here that every material parameter in the general notation was independently calibrated for bread dough.

**Offline loss and optimizer.** The objective sums a 3D Chamfer term and squared L2 tracked-particle errors over time, excluding invalid tracked points through occlusion masks. Warp differentiation is coupled to AdamW; the stated default learning rate is 10^-4 and E is scaled by 10^6. Crucially, §3.3 also states: “we also provide an option of zero-order optimization using CMA-ES on the forward simulator.” This is direct evidence that forward derivative-free fitting is already part of an embodied MPM framework.

**Online loss and adaptation.** Object-point tracking is omitted online because persistent tracks become unreliable under occlusion. Parameter updates are restricted to relatively quasi-static states to avoid unstable gradients. Short forward simulations under held/current controls are compared to observed geometry with Chamfer and mask terms. In the reported examples an update is attempted every five stream steps when the object is relatively steady, using ten forward simulation steps per update.

**What the reported evaluation establishes.** Offline results compare simulated geometry, tracking, masks, and rendered appearance with PhysTwin and PGND. The available text does not clearly specify an independent offline cross-trajectory split or physical units for every distance metric; those should not be invented. Online rope and bread-dough results compare alignment with and without correction. For bread dough, the reported distance changes from 0.0060 to 0.0059 and the mask term from 0.0031 to 0.0024. These numbers are the paper's loss values; they are not verified millimetres, rheometer errors, or frozen-parameter predictive accuracy. Online adaptation can reduce current discrepancy while leaving the identifiability of individual constitutive parameters unresolved.

The runtime table also needs care: the authors state that PhysTwin uses both CMA-ES and gradient optimization but report only its gradient training time in that comparison. Autonomous model-based manipulation is discussed as a future application rather than established by the reported identification metrics. Long-lived point tracks under large deformation and occlusion remain a limitation.

**Exact overlap with TaichiDough.** Real bread dough, two arms, action-conditioned continuum simulation, RGB-D reconstruction, material fitting, and a forward CMA-ES option overlap directly. Potential differences are the use of one continuous depth view rather than three cameras, explicit calibration of rate-dependent and permanent behavior rather than the reported E/ν fit, and independent prediction after freezing parameters. These differences are experimentally important but do not establish a first claim by themselves.

**Evidence locators:** `empm_full.txt` §3.1–3.2, lines 178–212; mechanics lines 234–247; offline Eq. (8), lines 223–249; formal parameter vector lines 265–276; online §3.4, lines 250–309; setup and actual optimized parameters lines 310–329; materials lines 335–351; online schedule lines 373–425; Tables 1–3, lines 435–477; applications/discussion lines 481–551.

## 5. Depth-based calibration and experimental identification

### 5.1 Differentiable Depth / DiffCal — Arnavaz et al.

**Observation and inputs.** A single Intel L515 LiDAR/depth camera measures fabricated silicone specimens made from Ecoflex-50 and MoldStar-15. CAD/reference geometry and clamp boundary conditions are known. ArUco markers on rigid clamps serve scene/camera calibration; there are no deformation markers on the soft specimen and no measured forces in the described calibration. The distinction between reference-geometry knowledge and marker-free observation matters when comparing to arbitrary dough whose hidden volume must be reconstructed.

**Model and fitted parameters.** The main model is stable neo-Hookean FEM, with a strain-rate dissipation potential controlled by a scalar damping coefficient β. The framework supports per-element E; experiments include homogeneous and heterogeneous stiffness. Poisson's ratio is fixed at 0.49 in all reported experiments. Semi-implicit Euler and reverse-mode autodiff with checkpointing are used. The authors explicitly describe the fitted values as parameters of the chosen discrete simulator, affected by mesh, constitutive model, and deformation modes.

**Objective.** The method minimizes a mean L2 discrepancy between observed raw depth images and rendered simulation depth, using differentiable ray tracing with treatment of discontinuity edges. This is a depth-image objective, unlike point-cloud fitting after full 3D reconstruction. Background pixels contribute to the loss, which complicates interpreting a small whole-image error as uniformly accurate object deformation. Adam uses decaying momentum; repeated random initializations are evaluated.

**Decisive dynamic experiment.** The paper jointly estimates E, β, and density from **twelve depth images at 30 Hz, approximately 0.4 seconds**, after synchronized support removal. It reports different parameter solutions for coarse and fine discretizations: approximately E = 223 kPa versus 277 kPa, β = 11.1 versus 8.8, and density = 1050 versus 1057 in the paper's density units. β should not be relabeled a dough shear viscosity without matching the constitutive definition and units. This experiment directly establishes prior single-depth-camera calibration of elastic and damping/density response.

**Evaluation and limitations.** The paper emphasizes fitting convergence, known-material fabricated examples, combined bending/twisting information, discretization changes, and constitutive-model comparisons. No independent unseen dynamic trajectory test was located in the read paper. Limited depth resolution, reflectance noise, known reference geometry, multiple material distributions consistent with similar deformations, and the memory-limited short dynamic horizon are important qualifications. Its abstract's self-described “first marker-free” claim is an author claim, not independently established priority in this review.

**TaichiDough comparison.** Single-view depth and an explicit rate-dependent response parameter are already present. Real dough, plastic deformation, changing tool contact, and incomplete initial volume present a different identification problem. Demonstrating that those additional difficulties can be resolved is more defensible than claiming the general sensor/optimizer combination is new.

**Evidence locators:** `diffcal_full.txt` lines 37–69 and 99–140; objective Eq. (1), lines 167–188; FEM/rendering lines 191–294; damping and integration lines 298–313; setup and fixed ν lines 339–365; combined motions lines 433–477; dynamic experiment lines 458–488; model comparisons lines 499–525; discussion/limitations lines 541–639.

### 5.2 DiffCloud — Sundaresan, Antonova and Bohg

**Observation and inputs.** A Kinova Gen3 with a Robotiq 2F-85 manipulates cloth. Two RealSense D435 cameras, overhead and side, provide calibrated point clouds. Robot geometry is used for masking. Initial object geometry and grasp location are known; simulated anchors replay the recorded end-effector motion. This is not geometry-agnostic single-view inference.

**Model and parameters.** DiffCloud uses differentiable thin-shell simulation through DiffSim/ARCSim with implicit Euler and rigid contact. It estimates two scalar multipliers on basis stiffness and mass parameters, each constrained to [0.1, 10] in simulation units through a sigmoid parameterization. These are not direct measurements of continuum E and specimen mass. The real cloth mesh has 7×7 vertices, or 49 nodes.

**Loss and optimization.** The point-set objective is one-way, simulated-to-real, squared nearest-neighbor Chamfer. The direction is intended to reduce attraction to real sensor outliers. Differentiable sampling maps mesh triangles to points. Although a sequence-level problem is formulated, practical real experiments usually evaluate the loss at one selected informative frame: the end of lifting or a middle folding frame. For folding, simulated sampling is heuristically restricted to the upper half. Adam runs for up to fifty iterations at learning rate 0.2.

**What the experiments test.** Five fabrics and three trajectories per fabric are used for lifting and folding, with approximately 2.5-second recordings sampled at 10 Hz. Results demonstrate real observation alignment and qualitative recovery of material categories. Thin paper-towel examples expose perception failures. MLP, PointNet++, and MeteorNet baselines are inverse models trained on simulated trajectories; the stated synthetic train/test split pertains to those data-driven baselines. Additional pole-contact examples are synthetic optimization targets. A held-out target sequence for which parameters are subsequently fitted is different from fitting once and predicting an independent real trajectory. No independent real cross-trajectory frozen-parameter evaluation was located in the read version.

**Limitations and comparison.** Full visibility reasoning, unknown initial geometry, and longer multistage manipulation remain open in the described method. It is a close precedent for depth/action-conditioned fitting, but not for calibrated real dough viscoplasticity. Its selected-frame strategy is a useful baseline for asking whether continuous depth adds identification information in TaichiDough.

**Evidence locators:** `diffcloud_full.txt` Eq. (2), lines 153–172; known geometry/anchors and simulator lines 174–208; sampling lines 210–237; sequence formulation lines 239–272; setup lines 288–305; fitted multipliers and data lines 304–347; informative-frame/occlusion choices lines 362–403; real results lines 404–419; synthetic benchmarks lines 402–469; limitations lines 430–458.

### 5.3 Wang et al. — Deformation Capture and Modeling of Soft Objects

**Observation and inputs.** Three Kinect cameras record dynamic point clouds at 30 Hz. An Artec Eva scan provides a high-quality initial object geometry. Fixed boundary assignments and primarily gravity-driven motion after twisting/pulling and release are used. Density is obtained from measured total mass and volume. This is a kinematic identification method, not a force-sensor fitting pipeline.

**Model and actual unknowns.** Coarse tetrahedral, corotated linear FEM is embedded in a detailed surface representation. Rayleigh damping is D = αM + βK. Material parameters are p = (E, ν, α, β); reference geometry is also estimated. The method alternates physics-based probabilistic tracking and model estimation, with homogeneous or spatially blended material parameterizations.

**Optimization distinction.** The material subproblem explicitly uses **gradient-free Nelder–Mead** against squared discrepancies between simulated and physics-tracked nodal trajectories. The reference-geometry subproblem uses mechanics and Jacobian information. Therefore the material calibration is a direct derivative-free precedent, while calling the entire pipeline wholly gradient-free would be inaccurate. Modal-frequency initialization and sampled stiffness estimates help initialize the search.

**Validation and failures.** Synthetic examples permit parameter recovery checks. Real silicone/elastomer examples include independent static loading under different weights and dynamic load-release comparisons. This is stronger evidence than fitting pictures alone, although it does not make every inferred coefficient an intrinsic, model-independent measurement. Coarse FEM introduces stiffness bias, multiple minima and flat regions occur, high-quality initial geometry/boundaries require manual work, and 30 Hz misses some high-frequency response relevant to damping. Contact-rich parameter inference is discussed as future work. Distance tables use normalized object coordinates; arbitrary conversion into millimetres is inappropriate.

**TaichiDough comparison.** This establishes early visual elastic/damped-solid identification with derivative-free material optimization and separate mechanical validation. Explicit plasticity, dough, and continuous contact manipulation are not part of its stated constitutive experiment.

**Evidence locators:** `wang2015_full.txt` lines 143–207; observation/tracking lines 174–207, 237–283; squared trajectory objective lines 261–277; rest-state optimization lines 297–331; Nelder–Mead lines 333–402; initialization and ambiguity lines 404–470; sensor setup lines 526–540; synthetic and real validation lines 565–634; normalized metrics lines 641–665; limitations lines 667–714.

### 5.4 Hahn et al. — Real2Sim: Visco-elastic parameter estimation from dynamic motion

**Observation and inputs.** Ten OptiTrack Prime 13 cameras track reflective markers on deformable objects and their rigid clamps. Measured clamp motion becomes Dirichlet boundary motion. Reference geometry is known. This study is highly relevant to mechanical identification but is not a depth-camera or marker-free method.

**Model and estimation.** Neo-Hookean FEM is combined with power-law viscosity. Fitted quantities include Lamé parameters λ and μ, a viscosity coefficient, and a power-law exponent h. The paper's ν in the viscosity formula is not Poisson's ratio. The objective is time-integrated least-squares marker trajectory error. Continuous direct-sensitivity and adjoint derivations are developed, including parameter-dependent initial equilibrium. BDF2 reduces numerical damping relative to less accurate integration, and L-BFGS optimizes log-parameterized positive quantities.

**What validation reveals.** The work examines synthetic recovery, real foam dynamics, new motions, heterogeneous parameter fields, and applications to soft structures. A limited-frequency motion poorly constrains a two-parameter damping law. Different clamp configurations and combined motions improve estimation of spatially varying parameters; a single motion can produce implausible distributions. Strong bending or snapping can require different apparent parameters from lower-deformation motions. Independent rheometer results are discussed, but their frequency/strain conditions differ from the visual experiment and are not exact matched-condition ground truth.

**TaichiDough lesson.** Motion diversity must provide information that separates parameters; more frames of nearly the same response may not. This paper also supports treating discretization and constitutive approximation as part of the interpretation. History-dependent plasticity is left for future work in the described framework, so the study does not resolve dough's permanent-deformation identification problem.

**Evidence locators:** `hahn2019_full.txt` lines 1–87; sensitivity/adjoint and integration lines 162–274; initial equilibrium lines 277–340; sensors, objective, and law lines 342–374; optimizer lines 470–500; real motion and material comparisons lines 505–626; multiple-boundary heterogeneous experiments lines 706–743; limitations lines 788–813.

## 6. RGB-based continuum identification

### 6.1 PAC-NeRF — Li et al.

**Observation and assumptions.** PAC-NeRF estimates geometry and dynamics from calibrated multi-view RGB videos. Camera intrinsics/extrinsics, foreground masks, rigid collision geometry, and the material family are assumed available. Geometry is initialized from the first frame, initial velocity is estimated from the first few frames, and material fitting then proceeds through a differentiable MPM/radiance-field model.

**Constitutive coverage and parameters.** Synthetic experiments include elastic objects with E and ν; plasticine with E, ν and yield stress; Newtonian fluids with viscosity and bulk modulus; non-Newtonian viscoplastic material with shear modulus, bulk modulus, yield stress and plastic viscosity; and sand with friction angle. The appendix describes neo-Hookean elasticity, StVK/Hencky-strain-based plasticine response with von Mises return mapping, a J-based Newtonian fluid law with viscous stress, a finite-rate viscoplastic return toward yield, and Drucker–Prager sand. Some PDF equations have typographic inconsistencies, so the physical categories are more reliable than blind transcription of extracted formulas.

**Optimization and losses.** RGB rendering error drives geometry/appearance and physical estimation. Initial velocity uses L-BFGS; material parameters use Adam through DiffTaichi MPM. The benchmark includes eleven synthetic viewpoints and known reference parameters. These synthetic parameter errors test recovery under a specified scene/material generator, not agreement with real rheometry.

**Real-data scope and evaluation.** The real example is a falling deformable ball filmed by four synchronized RealSense D455 cameras at 60 fps; the method explicitly uses RGB, **not the cameras' depth measurements**. The real result mainly demonstrates qualitative reconstruction/alignment, with geometry errors under limited views. The synthetic parameter tables include appreciable failures for some parameters despite generally favorable aggregate language. Per-sequence fitting is not independent cross-interaction real-world validation. Robot manipulation is discussed as a possible application rather than the real experiment performed.

**Exact comparison.** Joint visual inference of elastic/yield/viscous parameters is already represented here, so a three-component parameter vector is not new by itself. Real dough under prescribed tools and partial single-view depth remains a different and harder experimental identification setting.

**Evidence locators:** `pacnerf_full.txt` lines 168–217; initialization/known inputs lines 255–312; implementation lines 314–331; synthetic setup lines 348–366; parameter sets and optimizers lines 386–407; real RGB experiment lines 411–428; parameter tables/evaluation lines 430–525; limitations lines 528–544; Appendix A.1, lines 668–747; rigid boundaries lines 750–768.

### 6.2 GIC — Cai et al.

**Observation and reconstruction.** GIC combines dynamic Gaussian reconstruction with a continuum model. Its inputs are multi-view videos with known camera calibration and material family. Dynamic surface reconstruction and interior filling provide a simulation-ready representation. Rendered depth may be used during reconstruction, but that must not be confused with measured depth-camera observations in the real identification experiment.

**Physical objective and law.** Differentiable MPM parameters are fitted using temporal surface Chamfer distance plus multi-view mask L1 error. Material categories and parameters include elastic, von-Mises-plastic, Newtonian, viscoplastic, and Drucker–Prager granular response. The synthetic non-Newtonian cases fit modulus, yield, and plastic-viscosity quantities. Initial velocity is estimated using the first three frames; Adam is used for velocity and material parameters.

**What each evaluation means.** PAC-NeRF synthetic examples use eleven viewpoints and approximately fourteen frames. Spring-Gaus synthetic examples use ten views and thirty frames. One synthetic future-prediction test fits the first twenty frames and evaluates the final ten: this is same-trajectory temporal continuation. The real Spring-Gaus cases have three dynamic views plus fifty to seventy static images. Because the dynamic three-view observations are insufficient for the full dynamic reconstruction method, the real “Ours*” system-identification variant uses **mask-only fitting**, with static geometry aligned to the first dynamic frame.

Real grasp demonstrations use a UR10 and Robotiq 140. The appendix reports a transfer from the MPM identification model to a FEM-based Isaac Gym grasp simulation with mass and pose aligned. These demonstrations therefore do not establish that the same numerical simulator predicts independently calibrated force measurements. The paper's qualitative statements about grasp widths and forces should be kept distinct from a controlled constitutive-validation experiment.

**Limitations and comparison.** Multi-view calibration, a selected material law, reconstruction quality, and substantial computation remain important dependencies. Parameter errors on synthetic data, Chamfer reconstruction scores, mask losses, and real grasp outcomes measure different things. GIC is strong prior work for visual continuum identification and synthetic joint viscoplastic fitting, but not evidence of single-depth-camera real-dough parameter identification under known dual-tool trajectories.

**Evidence locators:** `gic_full.txt` lines 168–175; reconstruction lines 231–374; physical objective Eq. (6), lines 378–393; datasets/metrics lines 395–422; synthetic identification lines 470–518; temporal prediction lines 558–575; real “Ours*” and demonstrations lines 577–614; limitations lines 617–634; initial velocity lines 990–994; Appendix A.5, lines 1023–1059; constitutive families lines 1156–1262.

### 6.3 MASIV — Zhao et al.

**What is identified.** MASIV learns neural constitutive functions rather than a compact directly interpretable vector of E, viscosity and yield parameters. A neural elastic function maps deformation to stress; a neural plastic function maps trial deformation to an updated elastic state. Initial velocity is also learned. Frame indifference and undeformed equilibrium are incorporated as priors. Stable initialization uses a pretrained NCLaw constitutive model, so “material-agnostic” does not mean prior-free.

**Observation and objective.** Dynamic multi-view Gaussian reconstruction is used to infer dense pseudo-particle trajectories. Interior motion is inferred from coherent exterior motion and refinement; it is not directly observed internal material motion. L1 trajectory and silhouette losses supervise differentiable MPM learning. This changes the scientific target from identifying a few rheological coefficients to fitting a flexible constitutive mapping.

**Evaluation and qualifications.** PAC-NeRF and Spring-Gaus datasets support synthetic tests; the real examples use existing elastic-object data rather than new real dough trials. Real observed-state rendering, synthetic future prediction, and qualitative transfer to new geometry are reported. The main paper does not restate every future-frame split detail; the exact GIC 20/10 split should not be attributed to MASIV without checking its supplement. MASIV is not uniformly best on all measures: its aggregate future Chamfer distance is worse than GIC's in the reported comparison. The paper acknowledges overfitting with scarce observations, reliance on multi-view data, assumptions about interior motion, and lack of explicit separation of material properties from external forces; uniform gravity is assumed.

**TaichiDough comparison.** MASIV is relevant when a fixed material family is too restrictive. It does not supply direct evidence that visual data uniquely identify physical E, viscosity, and yield parameters in real dough. A report should distinguish interpretable parameter calibration from neural constitutive learning.

**Evidence locators:** `masiv_full.txt` lines 72–124 and 129–183; pseudo-trajectories lines 249–325; pretrained initialization lines 291–293; parameters/priors/losses lines 329–375; datasets lines 302–328; Tables 2–3 and prediction lines 397–443; training/overfitting lines 442–473; transfer and limitations lines 545–584. Separate supplement not read.

## 7. Forward, hybrid and likelihood-free identification

### 7.1 Matl et al. — granular material identification

**Observation and known inputs.** An overhead RealSense D435 observes final formations of couscous and barley produced by a known funnel/pouring setup. Grain size and mass, funnel geometry and height are treated as known. An ABB YuMi is used for robotic pouring tasks. The experiment uses a black velvet floor, and a validation bowl is velvet-lined to match the modeled contact conditions.

**Model and inference.** NVIDIA Isaac provides implicit nonsmooth discrete-element contact dynamics. Grains are modeled as identical rigid cohesionless spheres, with negligible drag, and grain–grain and grain–ground coefficients are equated. The unknowns are sliding friction, rolling friction and restitution. BayesSim learns a conditional posterior from forward simulations through a mixture-density random-Fourier-feature model. Simulator gradients are not required. The paper explicitly targets predictive macroscopic behavior rather than exact microscopic contact coefficients.

**Loss information and uncertainty.** Final depth formations are reduced to sixteen summary statistics describing spread, heights and distributions. The posterior is therefore conditioned on those summaries, not all raw depth pixels. Noise, observation blur, sample size and model mismatch are examined. Posterior uncertainty is propagated to macroscopic outputs.

**Independent evaluation.** Real calibration uses ten pours at 12 cm, each observation averaging ten depth frames. For reported deployment, five low-error parameter sets are selected and averaged. New pour heights of 2, 4, 6, 8 and 10 cm provide independent conditions. Robotic ring/bowl pours and predicted spilled grain amounts add task-level evaluation. The reported standardized summary-statistic L2 error is not a depth-pixel MAE and cannot be compared numerically to a dough heightmap error without reproducing the definitions.

**TaichiDough comparison.** This is a strong precedent for depth-driven likelihood-free calibration, uncertainty, and held-out actuation conditions. Granular contact mechanics is not viscoelastoplastic dough mechanics. Its use here is methodological, not a claim that the inferred coefficients or constitutive assumptions transfer between the materials.

**Evidence locators:** `matl_full.txt` lines 67–77; assumptions lines 119–152; BayesSim and parameters lines 155–182; geometry/sensors lines 183–241; summary statistics lines 242–260; uncertainty lines 261–295; synthetic/model mismatch lines 297–304 and following; real calibration and held-out heights lines 305–357; robotic tasks and limitations lines 359–379.

### 7.2 PhysTwin — Jiang et al.

**Observation and action.** PhysTwin uses three RealSense D455 RGB-D cameras to record one- or two-hand manipulations. A single RGB image is also used for a generated geometry prior through TRELLIS, followed by scale/pose/ARAP and depth/raycast alignment. That single-image geometry step does **not** make the full physical inference a single-camera method. Segmentation and depth-lifted hand tracking produce 3D control points; object tracking uses CoTracker3 lifted through depth.

**Model and parameter fitting.** The physical representation is a spring–mass graph with spring stiffness and rest length, dashpot and drag damping, collision response, and control-interaction parameters. A zero-order stage first searches homogeneous/global and topology quantities. A subsequent gradient stage refines dense spring stiffness and collision parameters. The main text places CMA-ES literature in this context, and EMPM explicitly reports using both CMA-ES and gradient optimization for its PhysTwin baseline. Exact iteration counts were not established from the read main paper. These coefficients are effective graph-simulator quantities, not direct calibrated continuum E, viscosity and yield values. No explicit plastic/yield constitutive law was identified in the read model description; this is not a statement that all spring–mass methods inherently cannot model plasticity.

**Objective.** Physics fitting combines a one-way Chamfer treatment of partial observations with depth-lifted motion tracking. Appearance is optimized separately using image L1 and D-SSIM with deforming Gaussian splats. Rendering quality and mechanical prediction should therefore be reported separately.

**Evaluation definitions.** The dataset contains twenty-two scenarios with 1–10-second videos of ropes, cloth, stuffed toys and packages, using lifting, stretching, pushing and squeezing. A 7:3 training/future-frame split evaluates same-sequence continuation. A separate paired-interaction dataset uses the same object with different motions and hand configurations, including transfer between single-hand lifting and two-hand stretching. For target interactions, the reused physical model is registered to the first target frame. This is genuine interaction-transfer evidence, but it assumes the new initial state is observed and aligned. Nine manually annotated tracking points per video support evaluation. Reconstruction, future prediction and unseen interaction have separate tables; image metrics are evaluated from the center camera.

**TaichiDough comparison.** Bimanual visual/action-conditioned identification, sparse-view depth, hybrid fitting and held-out interaction evaluation already exist. Real dough's irreversible and rate-dependent response under tools could justify a different model and experiment, but not a broad first claim for bimanual visual physical fitting.

**Evidence locators:** `phystwin_full.txt` lines 192–231; reconstruction/control/physics loss lines 237–291; zero-order and gradient stages lines 296–332; appearance lines 334–362; setup and splits lines 329–362; Tables 1–2, lines 405–440; applications lines 463–479; target-episode registration/transfer lines 489–497. Separate supplement not read.

### 7.3 Yoon and Lim — high-resolution cloth modeling

**Access qualification.** The following details are verified in the available Oxford Academic publisher-text extraction, not a complete locally downloaded paper. The extraction ends in §5.4.2. No conclusion or complete limitations section was available; absence claims would be unwarranted.

**Observation and mechanics.** Two Intel L515 cameras capture cloth manipulated by a UR10 with an OnRobot RG2-FT gripper. Grasped simulation nodes follow robot motion. Initial rotation, translation and uniform-scale alignment are reused. The Taichi simulator uses an 80×80, 6400-node mass–spring–damper grid with explicit Euler at 0.0005 seconds. Optimized coefficients concern stretching, bending, damping and friction. Stretching/shear share a coefficient rather than being independently identified; stiffness/damping parameterizations include mass normalization. The search-range magnitudes should not be presented as independently calibrated SI material properties.

**Optimizer and objective.** §3.1.3 specifies fifty Bayesian optimization iterations in total, including ten random initial evaluations, followed by fifty gradient-descent epochs. It is a **hybrid BO + GD method**, not a purely derivative-free method. A Gaussian-process surrogate is used. The exact gradient implementation, learning rate and acquisition details were not established from the available extraction. The fitting objective is a one-sided mean L1 nearest-neighbor term plus a weighted maximum-distance term. Evaluation uses Chamfer alone. Coordinate units and the objective weight were not verified and should not be supplied by analogy to another paper.

**Training and tests.** Nine fabrics and five actions yield forty-five fitting demonstrations on 240 mm square specimens. Nine center-lift draping demonstrations are held out. Forty-five 180 mm specimens are used for size transfer under the same fabric types. These tests concern new task/size conditions, not unseen material identities. Comparisons with DiffCloud also differ substantially in spatial resolution, 6400 versus 49 nodes; optimizer superiority cannot be isolated from that comparison alone.

**TaichiDough comparison.** This supports the validity of combining expensive forward evaluations with surrogate/global optimization and later gradient refinement. Its cloth model and tests are not evidence of volumetric dough parameter identification, and its hybrid design should not be cited as exclusively derivative-free.

**Evidence locators:** publisher article §§3.1.1–3.1.4, especially equations (5)–(11) and §3.1.3; §§4.1–4.2 for sensors/data; §5.2/Table 3 and §5.3/Table 4 for held-out draping and size transfer. Source URL in [YOON].

## 8. Consequences for a defensible TaichiDough contribution

This section is a synthesis of the papers, not a claim that the current repository already implements or validates the proposed experiments.

### 8.1 Claims that the reviewed evidence does not support

| Proposed broad claim | Relevant prior evidence | More defensible question |
|---|---|---|
| First calibration of deformable material parameters from visual/depth data and tool motion | DPSI, EMPM, DiffCloud, PhysTwin, Yoon–Lim | What additional information or robustness does the specific dough experiment provide? |
| First use of MPM for visual material identification | DPSI, EMPM, PAC-NeRF, GIC | Does the chosen constitutive model improve independent real-dough prediction? |
| First bimanual real-dough physical calibration framework | EMPM already uses bread dough and two Franka arms | Can a more restricted sensor setup identify rate-dependent and permanent behavior that its stated E/ν experiment does not establish? |
| First single-depth-camera elastic/viscous calibration | DiffCal | Can the method work with unknown hidden volume and changing contact rather than known silicone geometry/clamps? |
| Novel because fitting uses forward sweeps rather than gradients | EMPM CMA-ES, Wang Nelder–Mead, Matl BayesSim, hybrid PhysTwin/Yoon–Lim | Are the search method's accuracy, cost and uncertainty characterization appropriate and reproducible? |
| First simultaneous visual elasticity/plasticity/viscosity estimation | PAC-NeRF and GIC synthetic viscoplastic parameter sets | Can those responses be distinguished experimentally in real dough from contact and reconstruction error? |
| Novel because parameters are tested on new motions | DPSI, Hahn, Wang, PhysTwin and Matl already perform variants of predictive validation | Which new transfer result, failure analysis or experimental protocol advances knowledge for dough? |

A careful paper could state its concrete contribution without a priority adjective: **“We investigate the identifiability and predictive transfer of an effective viscoelastoplastic dough model from single-view depth observations under recorded dual-tool manipulation.”** That wording is appropriate only if identifiability and transfer are actually investigated. A method that fits one trajectory should instead claim calibration to that trajectory.

### 8.2 Separate the quantities being identified

A useful inverse-problem notation is:

- θ: constitutive parameters of the selected dough law, such as elastic stiffness, a defined rate-dependent coefficient, and yield/plastic-flow parameters;
- ξ: other unknown or uncertain quantities, including initial hidden volume, density if not measured, camera calibration, time alignment, tool pose, table contact, tool contact and adhesion;
- u(t): known imposed tool trajectories;
- H: the depth observation process, including visibility and background treatment.

The fitted objective compares observations to H applied to simulated states under θ, ξ and u(t). Changes in ξ can produce residual changes resembling changes in θ. This is an inference problem to measure, not a reason to assert that visual estimation is impossible. A known trajectory is a prescribed input; it need not be optimized or differentiated to identify material parameters.

The term “viscosity” must be tied to its stress/strain-rate law and units. Rayleigh damping, spring dashpots, power-law viscosity, plastic viscosity and a velocity-gradient viscous stress are not interchangeable coefficients. Similarly, a numerical singular-value clamp, von Mises yield stress and a learned plastic mapping have different parameter meanings. Contact friction and adhesion must be reported separately from intrinsic dough rheology.

### 8.3 What an identification experiment should establish

The following design implications are supported by the reviewed identification studies; they are recommendations, not results already demonstrated for TaichiDough.

1. **Independent response modes.** Include unloading/recovery, controlled changes in actuation rate, and deformation that leaves a residual state. Combine motions rather than assuming a long sequence of one motion identifies every parameter. Hahn's damping coupling and heterogeneous examples, DiffCal's bending/twisting comparison, and DPSI's contact-complexity experiments provide concrete precedents.
2. **Nuisance-parameter analysis.** Perturb plausible initial volume, camera extrinsics, time offset, density and contact assumptions. Determine whether the material estimates remain stable or whether different combinations fit equally well. DPSI's contact compensation and geometry completion, DiffCal's known geometry, and Wang's joint rest-state estimation show why this is necessary.
3. **Information from continuous depth.** Compare an endpoint-only objective with sampled intermediate frames and a fuller time-resolved objective, while holding the motion and training observations otherwise consistent. This tests a specific advantage over endpoint methods such as DPSI, rather than assuming more images automatically imply better identifiability.
4. **Frozen-parameter prediction.** Select parameters and hyperparameters using only fitting and validation data, then freeze them for independent tool trajectories, rates and initial configurations. State whether each target episode supplies a new initial-depth reconstruction. Report any online correction separately; EMPM's adaptive alignment and PhysTwin's target-initial-state registration illustrate the distinction.
5. **Model comparison.** Compare elastic-only, elastic-plus-dissipative, elastoplastic, and combined models under matched observation and computation budgets. Improvement should persist on independent manipulations, not only reduce fitting error through extra degrees of freedom.
6. **Parameter uncertainty and resolution sensitivity.** Use profile losses, repeated initializations, justified resampling units, or likelihood-free/posterior methods as appropriate. Refit at more than one numerical resolution. A confidence interval generated from effectively one independent experiment should not be treated as strong uncertainty evidence. DiffCal, Wang, Hahn, DPSI and Matl motivate these checks, but they do not prescribe one universally best uncertainty method.
7. **Physical interpretation proportional to evidence.** Cross-trajectory prediction can justify an effective simulator-parameter claim. Calling the estimates intrinsic rheological properties needs additional matched-condition physical measurement and careful interpretation of constitutive assumptions. Adjacent dough-rheology evidence is outside this note's scope.

### 8.4 Report metrics by what they measure

At least four result categories should remain distinct:

- **Fitting accuracy:** agreement with the observations used to choose parameters.
- **Predictive accuracy:** agreement under future frames or new interactions not used for fitting/selection, with the initial-state information disclosed.
- **Parameter recovery:** agreement with known synthetic parameters or independently measured physical quantities under matching definitions/conditions.
- **Task performance:** successful manipulation, grasping or pouring enabled by the model.

Metrics with different distance powers, point counts, visibility rules, normalizations or units cannot be compared by magnitude alone. In particular, DPSI's summed heightmap residual, DiffCloud's one-way squared Chamfer, DiffCal's whole-image L2 depth objective, EMPM's distance/mask terms, Matl's standardized summary statistics, and image-rendering scores are not interchangeable error measures. High rendering quality does not establish correct stress response, and low endpoint error does not establish a unique elastic/viscous/plastic decomposition.

### 8.5 Bounded novelty conclusion

No verified paper in this targeted set establishes the exact full experiment of continuous one-view depth, known dual-tool real-dough manipulation, a separately interpreted elastic/viscous/plastic parameterization, contact/geometry uncertainty analysis, and independent cross-rate/cross-trajectory prediction. **That observation is not proof that no such work exists.** The search was targeted, some source versions were preprints, and two separate supplements were not read.

The established overlap is already substantial enough that the report should avoid broad first claims. A publishable result could still come from a new inference method, a well-designed and reproducible experiment, demonstrated identifiability under a practical sensing restriction, a dataset/protocol with useful independent tests, or a convincing account of where and why common dough models fail. The scientific contribution is the new knowledge demonstrated by those results, not merely the choice of Taichi, MPM or a forward optimizer.

## 9. Verified bibliography and primary sources

Author names are taken from the papers when metadata conflicts with them. DOI verification establishes the publication record; the source-version table in §2 states which scientific text was actually read.

### [DPSI]

Xintong Yang, Ze Ji, and Yu-Kun Lai. **Differentiable physics-based system identification for robotic manipulation of elastoplastic materials.** *The International Journal of Robotics Research* 44(13), 2126–2155, 2025. DOI: [10.1177/02783649251334661](https://doi.org/10.1177/02783649251334661). Online publication 9 May 2025; print issue November 2025. Read version: [arXiv:2411.00554](https://arxiv.org/abs/2411.00554), v3, 18 February 2025. [Read PDF](https://arxiv.org/pdf/2411.00554). Final journal metadata verified through Crossref; publisher PDF not read.

### [EMPM]

Yunuo Chen, Yafei Hu, Lingfeng Sun, Tushar Kusnur, Laura Herlant, and Chenfanfu Jiang. **EMPM: Embodied MPM for Modeling and Simulation of Deformable Objects.** *IEEE Robotics and Automation Letters* 11(4), 4179–4186, 2026. DOI: [10.1109/LRA.2026.3664610](https://doi.org/10.1109/LRA.2026.3664610). April 2026 issue verified through Crossref; a precise earlier online-publication date was not independently established here. Read version: [arXiv:2601.17251](https://arxiv.org/abs/2601.17251), v1, 24 January 2026. [Read PDF](https://arxiv.org/pdf/2601.17251). [Project](https://embodied-mpm.github.io/).

### [DIFFCLOUD]

Priya Sundaresan, Rika Antonova, and Jeannette Bohg. **DiffCloud: Real-to-Sim from Point Clouds with Differentiable Simulation and Rendering of Deformable Objects.** *2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 10828–10835, 2022. DOI: [10.1109/IROS47612.2022.9981101](https://doi.org/10.1109/IROS47612.2022.9981101). Read version: [arXiv:2204.03139](https://arxiv.org/abs/2204.03139), v2, 13 May 2025. [Read PDF](https://arxiv.org/pdf/2204.03139). [Project](https://diffcloud.github.io/). Crossref misspells the last author as “Bohgl”; the primary paper gives “Bohg.”

### [DIFFCAL]

K. Arnavaz, M. Kragballe Nielsen, P. G. Kry, M. Macklin, and K. Erleben. **Differentiable Depth for Real2Sim Calibration of Soft Body Simulations.** *Computer Graphics Forum* 42(1), 277–289, 2023. DOI: [10.1111/cgf.14720](https://doi.org/10.1111/cgf.14720). Online publication 23 November 2022; print issue February 2023. Author initials follow the read paper. [Read author-hosted early-online PDF](https://erleben.github.io/pubs/2022/arnavaz.ea22/arnavaz.ea22.pdf). [Code](https://github.com/diku-dk/DiffCal). The read PDF's 2022 provisional pagination differs from the final 2023 record.

### [PACNERF]

Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, and Chuang Gan. **PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields for Geometry-Agnostic System Identification.** *International Conference on Learning Representations (ICLR)*, 2023. [Official OpenReview record](https://openreview.net/forum?id=tVkrbkz42vc). Read version: [arXiv:2303.05512](https://arxiv.org/abs/2303.05512), v1, 9 March 2023. [Read PDF](https://arxiv.org/pdf/2303.05512). No separate proceedings DOI was verified. Cite ICLR 2023, not the erroneous 2022 year found in some secondary citations.

### [GIC]

Junhao Cai, Yuji Yang, Weihao Yuan, Yisheng He, Zilong Dong, Liefeng Bo, Hui Cheng, and Qifeng Chen. **GIC: Gaussian-Informed Continuum for Physical Property Identification and Simulation.** *Advances in Neural Information Processing Systems* 37, 75035–75063, 2024. DOI: [10.52202/079017-2388](https://doi.org/10.52202/079017-2388). DOI and final pages verified through Crossref. Read version: [arXiv:2406.14927](https://arxiv.org/abs/2406.14927), v3, 31 October 2024. [Read PDF](https://arxiv.org/pdf/2406.14927). [Project](https://jukgei.github.io/project/gic). The downloaded 22-page version is not the final 29-page proceedings PDF.

### [MATL]

Carolyn Matl, Yashraj Narang, Ruzena Bajcsy, Fabio Ramos, and Dieter Fox. **Inferring the Material Properties of Granular Media for Robotic Tasks.** *2020 IEEE International Conference on Robotics and Automation (ICRA)*, 2770–2777, 2020. DOI: [10.1109/ICRA40945.2020.9197063](https://doi.org/10.1109/ICRA40945.2020.9197063). Read version: [arXiv:2003.08032](https://arxiv.org/abs/2003.08032), v4, 5 November 2020. [Read PDF](https://arxiv.org/pdf/2003.08032). [NVIDIA publication record](https://research.nvidia.com/labs/srl/publication/matl-2020-inferring/).

### [WANG]

Bin Wang, Longhua Wu, KangKang Yin, Uri Ascher, Libin Liu, and Hui Huang. **Deformation Capture and Modeling of Soft Objects.** *ACM Transactions on Graphics* 34(4), Article 94, 12 pages, 2015. DOI: [10.1145/2766911](https://doi.org/10.1145/2766911). SIGGRAPH 2015 paper. [Read author PDF](https://binwangbfa.github.io/publication/sig15_deformationcapture/SIG15_DeformationCapture.pdf). [Author publication page](https://binwangbfa.github.io/publication/sig15_deformationcapture/). Crossref records online publication on 27 July 2015; the paper's issue citation says August 2015.

### [HAHN]

David Hahn, Pol Banzet, James M. Bern, and Stelian Coros. **Real2Sim: Visco-elastic parameter estimation from dynamic motion.** *ACM Transactions on Graphics* 38(6), Article 236, 13 pages, 2019. DOI: [10.1145/3355089.3356548](https://doi.org/10.1145/3355089.3356548). [Read ETH author PDF](https://crl.ethz.ch/papers/Real2Sim.pdf). [Code](https://github.com/david-hahn/MyFEM). Crossref's short title is “Real2Sim”; the complete subtitle is verified in the paper.

### [PHYSTWIN]

Hanxiao Jiang, Hao-Yu Hsu, Kaifeng Zhang, Hsin-Ni Yu, Shenlong Wang, and Yunzhu Li. **PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos.** *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 7219–7230, 2025. [Read CVF accepted main paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_PhysTwin_Physics-Informed_Reconstruction_and_Simulation_of_Deformable_Objects_from_Videos_ICCV_2025_paper.pdf). [Project](https://jianghanxiao.github.io/phystwin-web/). Author list, title, status and pages verified from CVF paper; DOI not verified. Separate supplement not read.

### [MASIV]

Yizhou Zhao, Haoyu Chen, Chunjiang Liu, Zhenyang Li, Charles Herrmann, Junhwa Hur, Yinxiao Li, Ming-Hsuan Yang, Bhiksha Raj, and Min Xu. **Toward Material-Agnostic System Identification from Videos.** *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 5944–5956, 2025. DOI: [10.1109/ICCV51701.2025.00562](https://doi.org/10.1109/ICCV51701.2025.00562). [Read CVF accepted main paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhao_Toward_Material-Agnostic_System_Identification_from_Videos_ICCV_2025_paper.pdf). [Code](https://github.com/Skaldak/MASIV). Separate supplement not read.

### [YOON]

Kang-il Yoon and Soo-Chul Lim. **Real-to-sim high-resolution cloth modeling: Physical parameter optimization using particle-based simulation with robot manipulation data.** *Journal of Computational Design and Engineering* 12(8), 29–44, 2025. DOI: [10.1093/jcde/qwaf065](https://doi.org/10.1093/jcde/qwaf065). Online publication 18 July 2025; print publication 4 August 2025. [Publisher article](https://academic.oup.com/jcde/article/12/8/29/8206149). Bibliography verified through Crossref and publisher record; scientific evidence limited to the partial publisher-text extraction described above.

## 10. Search and retrieval log

### 10.1 Targeted web searches

The following twelve queries were used. Earlier work supplied names only as search leads; scientific claims were checked against the primary papers rather than inherited assessments. No further broad searches were used to finish this synthesis.

1. `"Differentiable Physics-based System Identification" elastoplastic materials Yang 2025`
2. `"EMPM" "Embodied MPM"`
3. `"DiffCloud" point clouds mass stiffness paper`
4. `"PAC-NeRF" "GIC" physical property identification`
5. `"Differentiable Depth" "Real2Sim" soft body calibration`
6. `"qwaf065"`
7. `"Inferring the Material Properties of Granular Media for Robotic Tasks"`
8. `"Toward Material-Agnostic System Identification from Videos"`
9. `"PAC-NeRF" arxiv`
10. `"GIC: Gaussian-Informed Continuum" arxiv`
11. `"Real2Sim" "Visco-elastic parameter estimation" Hahn pdf`
12. `"Deformation capture and modeling of soft objects" Wang 2015 pdf`

### 10.2 Additional targeted verification

- Crossref DOI records were queried for DPSI, EMPM, DiffCloud, DiffCal, Matl, Yoon–Lim, MASIV, GIC, Hahn and Wang. Metadata was reconciled with primary title pages where necessary.
- The Yoon–Lim publisher page was queried for methods/splits and again for optimizer and limitations details. The second extraction did not establish access beyond the available §5.4.2 endpoint.
- PhysTwin was located from primary-paper references and its public CVF paper. Its CVF HTML request returned HTTP 403, while direct access to the public PDF succeeded.
- Downloaded PDFs were checked as PDF files and converted with `pdftotext -layout`. The eleven PDF/text pairs listed in §2 are stored alongside this report.
- A Cardiff-hosted final DPSI PDF returned HTTP 403; the available arXiv v3 was read instead.
- OpenReview PDF endpoints for PAC-NeRF and GIC returned HTTP 403; their public arXiv PDFs were read instead.
- One DiffCal Crossref request returned HTTP 429; a subsequent targeted metadata request succeeded.

### 10.3 Limits of this evidence set

This review does not cover adjacent food rheology comprehensively, audit current TaichiDough implementation, or re-run any paper's experiments. Separate MASIV and PhysTwin supplements were not read. Yoon–Lim full-paper access was incomplete. Some final journal/proceedings metadata refer to a different version than the read preprint. Unclear distance units, split definitions and implementation details have been left explicitly unclear rather than filled in by analogy. A feature not located in a paper is not assumed absent from every version, supplement or code release.

## Sources

- [DPSI published record](https://doi.org/10.1177/02783649251334661) and [read full text](https://arxiv.org/pdf/2411.00554)
- [EMPM published record](https://doi.org/10.1109/LRA.2026.3664610) and [read full text](https://arxiv.org/pdf/2601.17251)
- [DiffCloud published record](https://doi.org/10.1109/IROS47612.2022.9981101) and [read full text](https://arxiv.org/pdf/2204.03139)
- [DiffCal published record](https://doi.org/10.1111/cgf.14720) and [read full text](https://erleben.github.io/pubs/2022/arnavaz.ea22/arnavaz.ea22.pdf)
- [PAC-NeRF official record](https://openreview.net/forum?id=tVkrbkz42vc) and [read full text](https://arxiv.org/pdf/2303.05512)
- [GIC published record](https://doi.org/10.52202/079017-2388) and [read full text](https://arxiv.org/pdf/2406.14927)
- [Matl et al. published record](https://doi.org/10.1109/ICRA40945.2020.9197063) and [read full text](https://arxiv.org/pdf/2003.08032)
- [Wang et al. published record](https://doi.org/10.1145/2766911) and [read full text](https://binwangbfa.github.io/publication/sig15_deformationcapture/SIG15_DeformationCapture.pdf)
- [Hahn et al. published record](https://doi.org/10.1145/3355089.3356548) and [read full text](https://crl.ethz.ch/papers/Real2Sim.pdf)
- [PhysTwin read CVF full text](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_PhysTwin_Physics-Informed_Reconstruction_and_Simulation_of_Deformable_Objects_from_Videos_ICCV_2025_paper.pdf)
- [MASIV published record](https://doi.org/10.1109/ICCV51701.2025.00562) and [read CVF full text](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhao_Toward_Material-Agnostic_System_Identification_from_Videos_ICCV_2025_paper.pdf)
- [Yoon–Lim published record](https://doi.org/10.1093/jcde/qwaf065) and [partially accessed publisher text](https://academic.oup.com/jcde/article/12/8/29/8206149)
