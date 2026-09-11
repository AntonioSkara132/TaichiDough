# Dough rheology, inverse measurement, and robotic experiment design

Fresh evidence review, 11 September 2026. Scope: real flour dough, related food materials, mechanical inverse identification, vision/robotic rheometry, and experiments that distinguish recoverable, rate-dependent, and non-recoverable deformation. DPSI, EMPM, and PAC-NeRF are intentionally left to the separate inverse-simulation review.

## 1. Main finding for the TaichiDough research question

The proposal is scientifically meaningful: infer a compact model of dough response from recorded depth geometry and known tool motion, then test whether it predicts new interactions. There is substantial food-engineering precedent for inverse simulation of dough, and robotics precedent for identifying rheological models before predicting deformation. Therefore neither inverse fitting nor robot-assisted dough rheometry is a new idea by itself.

There is nevertheless a credible research question between these fields: **under what loading, observation, and contact conditions can a single depth view identify an elastic–viscous–plastic simulator response that transfers to different tool motions?** A study answering that question with controlled actions, uncertainty, independent mechanical references, and withheld trajectories would provide more scientific information than an unqualified claim to have found three material constants. The opportunity is not established by the absence of a paper with exactly the same hardware; it needs a demonstrated advantage, a useful measurement limitation, or a reproducible benchmark.

Four distinctions should organize the report:

1. **A model parameter is not a material category.** Elastic modulus, shear viscosity, relaxation time, yield stress, and a dimensionless plastic strain limit are different quantities. Their units and constitutive equations must be given.
2. **Residual strain is not sufficient evidence of plastic yielding.** A dashpot produces lasting strain without a yield threshold; slow viscoelastic recovery may look permanent within a short recording. Dough damage, aging, drying, adhesion, and gravity also affect observed recovery.
3. **Tracking is not prediction.** Repeatedly fitting a mesh to each incoming depth image can track accurately even when the underlying material model cannot predict a withheld trajectory.
4. **Known displacement is not known force.** A recorded tool trajectory specifies kinematics. It does not necessarily give the absolute stress scale required to identify modulus and viscosity in physical units.

## 2. Source verification and limitations

Technical evidence levels used below:

- **Full text:** the original article XML or an open publisher/author PDF was read, including methods/results.
- **Primary abstract:** an original publisher or biomedical-index abstract was inspected, with DOI metadata independently checked.
- **Metadata plus discovery excerpt:** title/authors/venue/DOI were checked through Crossref/publisher records, but full methods were unavailable. Such papers are useful leads and must not support precise quantitative claims in the final survey.

Twelve targeted WebSearch queries were used, followed by original article pages, Crossref DOI records, Europe PMC full-text XML, and open author manuscripts. Some MDPI/ScienceDirect/PubMed web pages returned HTTP 403 or browser checks; Europe PMC XML and publisher PDFs supplied full text for the principal papers. No claim of a complete systematic review or exhaustive priority search is justified by this scoped search.

## 3. Closest food-engineering inverse-identification paper

### R1. Fabbri and Cevoli — inverse FEM rheometry of semolina dough

**Citation:** Angelo Fabbri and Chiara Cevoli. “Estimation of semolina dough rheological parameters by inversion of a finite elements model.” *Journal of Agricultural Engineering* 46(3), 95–99 (2015). DOI: [10.4081/jae.2015.469](https://doi.org/10.4081/jae.2015.469).

**Primary full text:** [Publisher PDF](https://www.agroengineering.org/jae/article/download/469/489/2564). Citation metadata independently checked with Crossref. **Evidence: full text.**

**What was actually identified:** consistency index \(k\), units Pa·s^n, and flow index \(n\) in the shear power law \(\tau=k\dot\gamma^n\). This is a non-Newtonian flow identification experiment, **not** a simultaneous estimate of Young’s modulus, yield stress, and viscoelastic relaxation. Do not turn its reported “rheological parameters” into an elasticity/plasticity claim.

**Specimen and experiment (pp. 96–97):** 1 kg semolina mixed with 0.5 kg water at 25°C for 15 minutes; 20 minutes of rest in a plastic enclosure at room temperature; approximately 20 g loaded into a laboratory extrusion apparatus based on a TA-HDi texture analyzer. The cylinder diameter was 20 mm; the identification die was 40 mm long with 5 mm radius; the load cell full scale was 2.5 kN. The imposed flow rate increased from zero to 3.6×10^-7 m³/s over 2 seconds. Measurements were repeated in triplicate.

**Known inputs and measured outputs:** piston motion/flow rate and die geometry were known. Force was measured and divided by piston area to obtain extrusion pressure. The inverse problem minimized disagreement between measured and FEM-predicted pressure along the flow-rate history. The model assumed incompressibility, no relative wall velocity, atmospheric outlet pressure, and density 1200 kg/m³ from a reference. It used 4,000 finite elements and remeshing as the piston advanced.

**Optimizer:** Levenberg–Marquardt, with Jacobian terms estimated by a 1% parameter perturbation. The article loosely calls this global optimization, but LM itself is a local nonlinear least-squares method. It is **not autodiff**, but it is also **not strictly derivative-free** because it uses a finite-difference Jacobian. This distinction matters when presenting it alongside TaichiDough sweeps.

**Validation:** a conventional capillary-rheometry procedure used 12 dies (four lengths × three radii), again with triplicate measurements, and Mooney–Rabinowitsch/Bagley corrections. Seven reported inverse iterations reached 3.6% disagreement for \(k\) and 2.0% for \(n\) relative to that reference. These percentages compare two characterization procedures, not unseen robot motions or image prediction. The authors explicitly note that agreement does not establish which procedure is closer to physical reality.

**Contribution relevant to TaichiDough:** established precedent for a simpler apparatus plus forward physical simulation and numerical parameter search, with an independent mechanical comparison. TaichiDough could add partial-depth observation and action-conditioned prediction, but should match this paper’s clarity about units, preparation, boundary assumptions, and what was measured.

## 4. Vision and robotic rheometry: the close contemporary food comparator

### R2. Monleón-Getino et al. — MIRANDA and RELAPP

**Citation:** Antonio Monleón-Getino, Víctor Madarnás-Gómez, Mario Cobos-Soler, Eduard Almacellas, Juan Ramos-Castro, Xavier Bielsa, Pere López-Brosa, Àngels Sahuquillo-Estrugo, Inés Marsà-González, and Alejandro Rodríguez-Mena. “Advancing Viscoelastic Material Characterization Through Computer Vision and Robotics: MIRANDA and RELAPP.” *Materials* 18(21), 4827 (2025), published 22 October. DOI: [10.3390/ma18214827](https://doi.org/10.3390/ma18214827).

**Primary full text:** [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12608870/fullTextXML); [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12608870/). **Evidence: full text, including §§2.2–2.5, §§3.3–3.4, Appendix A.**

**Important overlap:** it already combines computer vision, robotic loading, material recovery, and real flour-dough characterization. A claim that TaichiDough is the first camera-and-robot framework for dough rheology would be untenable.

**What it measures:** MIRANDA tracks selected visible landmarks from video and summarizes motion with a median and interquartile variability. Reported outputs include percentage recovery, final deformation, recovery time \(T_R\), and fitted curve parameters. The camera is described as high-speed/high-resolution; the methods inspected do **not** establish metric depth-camera reconstruction. RELAPP actively deforms samples and records force in newtons and displacement/strain-related quantities. Loading may also be manual for MIRANDA demonstrations. Thus the overall work is not evidence that absolute material parameters can all be recovered from depth alone.

**Materials:** the article reports 113 observations across 17 categories. Five wheat flours were mixed with salted water using standardized laboratory mixing and constant hydration; 14 flour samples were measured. The specific hydration percentage, mixing duration, rest time, camera specifications, and complete load calibration are not stated in the inspected methodological sections. Flour reference measurements came from the Mat Control laboratory’s cereal testing program, including alveograph and other industrial rheometry measurements.

**What “viscosity prediction” means here:** support-vector regression relates MIRANDA recovery metrics to laboratory reference quantities such as alveograph baking strength \(W\), tenacity \(P\), extensibility \(L\), and RVA final viscosity. RVA final viscosity is a protocol-specific starch/flour response; it is **not equivalent to a room-temperature dough continuum shear-viscosity coefficient**. Similarly, percentage recovery is not Young’s modulus.

**Validation limitation — essential to retain:** §2.5 explicitly states that the data were **not split into training and test sets**, and all models were fitted to the complete dataset. Reported \(R^2\) values of 0.594 for \(W\), 0.575 for \(P\), and 0.612 for RVA final viscosity are therefore not evidence of held-out prediction. The paper reports broad bootstrap intervals; repeated parts of a dough sample are not independent flour batches. It is a useful proof of concept, not a definitive solution to physical identification or generalization.

**Protocol ambiguity:** the text calls its test creep–recovery, while Appendix A describes a robot moved to a set position with compliance affecting the resulting load. Ideal constant-force creep, fixed-displacement stress relaxation, and recovery after unloading must be distinguished in TaichiDough rather than copied as interchangeable labels.

**Specific scientific opportunity:** TaichiDough could complement this work by predicting full time-resolved visible geometry using a specified continuum model, explicitly estimating identifiable parameters, and testing complete held-out motions/batches. That would be a distinct question from regressing flour-quality indices from video descriptors.

## 5. What food rheology says the model must distinguish

### R3. Yazar — nonlinear flour-dough rheology review

**Citation:** Gamze Yazar. “Wheat Flour Quality Assessment by Fundamental Non-Linear Rheological Methods: A Critical Review.” *Foods* 12(18), 3353 (2023). DOI: [10.3390/foods12183353](https://doi.org/10.3390/foods12183353).

**Primary full text:** [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10527890/fullTextXML); [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10527890/). **Evidence: full text, particularly §3.4 and §§4.1–4.2.**

The review explains why small-strain elastic measurements do not fully characterize the large deformation relevant to mixing, sheeting, and gas-cell expansion. Stress relaxation, creep–recovery, extension, lubricated squeezing flow, and large-amplitude oscillatory shear (LAOS) probe different aspects of the response. LAOS can separate amplitude/frequency-dependent nonlinear elastic and viscous features; extensional tests reveal strain hardening and failure that small-amplitude testing can miss.

Hydration, mixing state, resting history, and protein composition change the measured response. An undermixed dough can appear stiff under a small deformation because of insufficiently hydrated structures yet behave weakly under a large deformation. Resting can soften or strengthen depending on prior mixing. These are reasons to record preparation history and test more than one amplitude/speed, not reasons to fit an arbitrarily large parameter vector.

**Boundary condition warning:** wall slip can contaminate rheological tests, whereas lubricated squeezing deliberately changes the boundary condition to approximate biaxial extension. Neither “zero slip” nor “nonzero friction” is universally appropriate for dough–tool contact. The study must state the interface condition.

Useful primary references traced within the review (not independently full-text audited in this branch):

- Wang, F. C., and Sun, X. S. “Creep-recovery of wheat flour doughs and relationship to other physical dough tests and breadmaking performance.” *Cereal Chemistry* 79, 567–571 (2002). DOI [10.1094/CCHEM.2002.79.4.567](https://doi.org/10.1094/CCHEM.2002.79.4.567).
- Edwards, N. M., Dexter, J. E., Scanlon, M. G., and Cenkowski, S. “Relationship of creep-recovery and dynamic oscillatory measurements to durum wheat physical dough properties.” *Cereal Chemistry* 76, 638–645 (1999). DOI [10.1094/CCHEM.1999.76.5.638](https://doi.org/10.1094/CCHEM.1999.76.5.638).
- Ng, T. S. K., McKinley, G. H., and Ewoldt, R. H. “Large amplitude oscillatory shear flow of gluten dough: A model power-law gel.” *Journal of Rheology* 55, 627–654 (2011). DOI [10.1122/1.3570340](https://doi.org/10.1122/1.3570340).
- Yoshimura, A. S., and Prud’homme, R. K. “Wall slip effects on dynamic oscillatory measurements.” *Journal of Rheology* 32, 575–584 (1988). DOI [10.1122/1.549982](https://doi.org/10.1122/1.549982).

### R4. Ng, McKinley and Padmanabhan — linear to nonlinear response

**Citation:** Trevor S. K. Ng, Gareth H. McKinley, and Mahesh Padmanabhan. “Linear to Non-linear Rheology of Wheat Flour Dough.” *Applied Rheology* 16(5), 265–274 (2006). DOI [10.1515/arh-2006-0019](https://doi.org/10.1515/arh-2006-0019). An older DOI is also circulated, 10.3933/ApplRheol-16-265; use the verified current publisher DOI above.

**Evidence:** Crossref metadata and publisher-linked discovery summary; full article was not acquired. The reported theme is extending linear critical-gel relaxation descriptions toward nonlinear extensional behavior and strain hardening. **Do not cite detailed model accuracy or fitted constants without checking the full text.** Useful as a foundational reading alongside R3: a single linear spring plus dashpot is a restricted approximation, particularly when deformation becomes large or rupture begins.

### R5. Sun et al. — formulation-dependent viscoelastic models

**Citation:** Xinyang Sun, Filiz Koksel, Michael T. Nickerson, and Martin G. Scanlon. “Modeling the viscoelastic behavior of wheat flour dough prepared from a wide range of formulations.” *Food Hydrocolloids* 98, 105129 (2020). DOI [10.1016/j.foodhyd.2019.05.030](https://doi.org/10.1016/j.foodhyd.2019.05.030).

**Evidence:** exact metadata verified using Crossref and publisher record; full methods unavailable. Discovery excerpts compare power-law-gel and Burgers descriptions of rheometry and recovery, but numerical/model-superiority details must be treated as unverified until full text is read. This paper is important for the scope of the survey: formulation dependence and recovery modeling are not new. TaichiDough must say which physical regime its fitted parameters represent rather than imply one modulus/viscosity covers all recipes and deformation histories.

### R6. Anderssen and Kružík — dough mixing with hysteresis

**Citation:** Robert S. Anderssen and Martin Kružík. “Modelling of wheat-flour dough mixing as an open-loop hysteretic process.” *Discrete and Continuous Dynamical Systems – B* 18(2), 283–293 (2013). DOI [10.3934/dcdsb.2013.18.283](https://doi.org/10.3934/dcdsb.2013.18.283).

**Primary record:** [Publisher page](https://www.aimsciences.org/article/doi/10.3934/dcdsb.2013.18.283). **Evidence: primary abstract and exact DOI metadata.**

The work formulates dough mixing as a rate-independent finite-deformation elastoplastic/hysteretic process, with polyconvex energy and multiplicative elastic/nonelastic deformation. It is theoretical: existence results for incremental and energetic solutions, motivated by experiments. It is not evidence of a camera-based estimator or an experimentally verified universal dough constitutive law. It demonstrates that history-dependent irreversible response has a substantial theoretical literature, and should not be conflated with a scalar damping factor.

## 6. Physical interface behavior cannot be assigned to bulk viscosity

### R7. Mitsoulis and Hatzikiriakos — rolling bread dough

**Citation:** Evan Mitsoulis and Savvas G. Hatzikiriakos. “Rolling of bread dough: Experiments and simulations.” *Food and Bioproducts Processing* 87(2), 124–138 (2009). DOI [10.1016/j.fbp.2008.07.001](https://doi.org/10.1016/j.fbp.2008.07.001).

**Primary publisher record:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S096030850800059X). **Evidence:** DOI/title/authors verified; technical details available only from publisher-linked search excerpts, not full text. Those excerpts describe Herschel–Bulkley yield/shear-thinning behavior and a wall-slip law in rolling experiments/simulation. Do not quote numerical errors or detailed boundary equations without the paper.

Why relevant: tool-driven dough deformation and constitutive/interface calibration precede modern MPM systems. The physically interesting question is whether changing the tool contact treatment changes the inferred bulk parameters or held-out prediction—not simply whether a tool mesh was used.

### R8. Ghorbel and Launay — dough adhesion, hydration and separation rate

**Citation:** Dorra Ghorbel and Bernard Launay. “An investigation into the nature of wheat flour dough adhesive behaviour.” *Food Research International* 64, 305–313 (2014). DOI [10.1016/j.foodres.2014.06.045](https://doi.org/10.1016/j.foodres.2014.06.045). **Correction of a possible bibliographic trap:** its PubMed indexing date does not make this a 2018 paper; the journal publication is 2014.

**Primary abstract:** [PubMed](https://pubmed.ncbi.nlm.nih.gov/30011655/), successfully read via the [Europe PMC API](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID%3A30011655%20AND%20SRC%3AMED&format=json&resultType=core). **Evidence: primary abstract and DOI metadata.**

Six probe materials were tested: PMMA, stainless steel, polyethylene, PVC, PTFE, and polypropylene. Specific separation energy varied with water content and probe material. The abstract reports that the water-content effect exceeded that of probe surface tension. Dependence on withdrawal rate was represented as an interfacial work contribution times a viscoelastic rate-dependent contribution. The authors argue that separation energy per unit interface area, not simply peak tensile force, is the appropriate adhesion measure.

Implication: material seen stretching behind a withdrawing tool might reflect adhesion and debonding, bulk dissipation, or both. Depth-only inference can compensate for a wrong contact model by changing apparent viscosity or plasticity. A study should explicitly separate pressing, sliding, and withdrawal/release phases; control tool surface and moisture; and avoid identifying “friction” from a bulk-shape loss alone without sensitivity analysis.

## 7. Robotics precedents outside MPM

### R9. Cocuzza and Yan — identified rheological sheet model and deformation prediction

**Citation:** Silvio Cocuzza and X.-T. Yan. “First engineering framework for the out-of-plane robotic shaping of thin rheological objects.” *Robotics and Computer-Integrated Manufacturing* 53, 108–121 (2018). DOI [10.1016/j.rcim.2018.02.005](https://doi.org/10.1016/j.rcim.2018.02.005).

**Open accepted manuscript:** [Strathprints PDF](https://strathprints.strath.ac.uk/63903/1/Cocuzza_Yan_RCIM_2018_First_engineering_framework_for_the_out_of_plane_robotic.pdf). **Evidence: full manuscript, §§2–3.4 and the introduction’s description of validation.** The manuscript carries an earlier title, “Robotic shaping of a thin rheological material over a moulding object – Modeling and experimental validation”; the final title above is verified against Crossref.

**Actual material and model:** fondant icing, not bread dough. A Kelvin–Voigt spring/dashpot pair in series with a second dashpot forms a three-element material. A catenary-like mass–spring–damper discretization describes thin-sheet deformation. The paper calls the series-dashpot behavior non-recoverable/viscoplastic; mechanically, the reported unit does not contain a yield threshold. This is an excellent example of why “plasticity” needs an explicit definition.

**Measurements:** Instron 3342 with a sensitive load cell; horizontal support to avoid unwanted sag in the material-identification test; fondant specimens 0.20×0.05 m, thickness 4 mm. Ten tensile specimens were pulled at 1 mm/s. The average force curve was used to identify spring \(k_1\) and dashpots \(c_1,c_2\). These are specimen-level force/displacement coefficients; they are not automatically continuum \(E\) and \(\eta\).

The third parameter could not be reliably determined from the asymptotic tensile force because fondant fractured before reaching it. The authors therefore used known-force loading/unloading and long-time residual deformation to estimate the series dashpot, averaging ten specimens. This is directly useful experiment-design precedent: if the original motion does not expose a parameter, add a different test rather than trust optimizer precision.

**Validation:** the article distinguishes identification/fit of the material unit from verification of the complete sheet model. Two physical deformation cases were considered: gravity-only sag and deformation by a shaping tool. The validated model was then used to compare tool motions and velocities in simulation. Do not claim the article demonstrates closed-loop multi-tool robotic control or metric RGB-D inverse identification.

**Implication:** a new TaichiDough paper can build on rather than ignore this history of rheological identification and tool-conditioned prediction. Its distinct aspects could be volumetric 3-D continuum simulation, partial depth sensing, two moving tools, and uncertainty/identifiability under incomplete observations. Those aspects need quantified evidence.

### R10. Petit et al. — RGB-D pizza-chef tracking

**Citation:** Antoine Petit, Vincenzo Lippiello, Giuseppe Andrea Fontanelli, and Bruno Siciliano. “Tracking elastic deformable objects with an RGB-D sensor for a pizza chef robot.” *Robotics and Autonomous Systems* 88, 187–201 (2017). DOI [10.1016/j.robot.2016.08.023](https://doi.org/10.1016/j.robot.2016.08.023).

**Primary record:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0921889016305395); [institutional record](https://www.iris.unina.it/handle/11588/649348). **Evidence:** exact metadata checked; methodological scope from primary-page discovery excerpts, full text not acquired in this branch.

It combines RGB-D observation, rigid registration, and non-rigid FEM fitting to track elastic deformation. This is an important perception precedent, not evidence that viscosity, yield, or unseen physical dynamics were identified. The distinction is central to TaichiDough: repeated visual state correction and a one-time parameter fit followed by free prediction answer different questions.

## 8. Experiments that can distinguish the target responses

These are proposed experiments derived from constitutive mechanics and the literature above, **not already demonstrated TaichiDough results**. Numerical values for speeds, forces, and duration should be chosen after a pilot that resolves the response above camera noise and below rupture/uncontrolled slip.

| Experiment | What is controlled and recorded | Main information | Essential limitation |
|---|---|---|---|
| Small-amplitude indentation or compression, followed by full release | Known tool displacement; full depth motion including exposed regions; force if available | Reversible deformation, geometry/contact checks; force–displacement helps modulus | A displacement-controlled shape alone may not fix the absolute stiffness scale |
| Same path at several speeds | Identical nominal displacement and contact geometry; different time parameterization | Rate-dependent response and viscous/elastic timescale trade-offs | Changing friction, inertia, slip, and specimen history can also alter rate response |
| Fixed-displacement hold | Tool held still after loading; depth and optionally force over time | Force decay identifies stress relaxation if the material law supports it | A visually stationary specimen can still relax stress: camera-only shape may reveal little or nothing |
| Known-force creep followed by unloading | Constant force (controlled or independently measured), deformation vs time, extended recovery | Compliance, retardation time, steady flow and residual deformation | A position-controlled robot without force feedback is not a constant-force test |
| Load–unload–recover at increasing peak deformation | Several amplitudes; repeat after sufficient recovery; track persistent geometry | Onset of non-recoverable response and whether a yield-type model is useful | Finite residual strain is not by itself a yield stress; slow recovery and damage must be tested |
| Repeated cycles and long rest | Identical paths on matched fresh and preconditioned specimens | History dependence, work softening/hardening, permanent change | Samples are changed by testing; order needs randomization or fresh replicates |
| Gravity sag or slump with known geometry and independently measured mass | Release tools; depth over time; known gravity, mass and dimensions | Adds a known body-force scale and a contact-light test | Gravity may be too weak at the observed timescale; hidden volume and floor support remain important |
| Controlled slide and withdrawal | Fixed normal indentation or measured normal force; varied sliding/retraction speed; known tool material | Slip/adhesion separated from compressive bulk response | One fitted velocity attenuation is not an identified Coulomb coefficient or adhesion law |
| Withheld manipulation | New path, speed, geometry or two-tool sequence; no fitting/resetting after initialization | Whether calibration has practical predictive value | Later frames of the calibration episode are a forecast test, not an independent experiment |

For actual food dough, every trial should record flour/recipe, hydration, salt/other ingredients, mixing duration, rest duration, preparation temperature, sample mass, initial volume, time since preparation, handling history, tool surface and cleaning/moisture condition. “Same dough” without this information is not a reproducible material condition. Split at the physical specimen/batch level when claiming material transfer; do not inflate sample size by counting adjacent video frames as independent.

## 9. What cannot automatically be recovered from depth and trajectories

The following is an analytical identifiability argument, not a novelty claim or a result attributed to the above papers.

### Absolute mechanical scale can be unobservable

For a homogeneous quasistatic body with prescribed displacement boundaries, traction-free remaining boundaries, negligible body forces, and fixed dimensionless constitutive ratios, multiplying all stress-scale parameters by the same positive constant can leave the predicted displacement unchanged. Equilibrium is \(\nabla\cdot\sigma=0\); if \(\sigma\) is replaced by \(a\sigma\), the same displacement still satisfies it. Reaction forces scale by \(a\), but a depth camera does not measure them.

A one-dimensional Kelvin–Voigt example makes the issue explicit: \(\sigma=E\varepsilon+\eta\dot\varepsilon\). If strain is prescribed by the robot and only strain is observed, the force response contains the information about \(E\) and \(\eta\). In a spatially deforming specimen, visible motion can add useful information, but it does not remove the scale ambiguity in every regime. Scaling \(E\) and \(\eta\) together preserves \(\eta/E\); similarly, scaling a yield stress with \(E\) preserves a normalized yield level in suitable models.

**How the ambiguity can be reduced:** independently measured force; known mass/density with resolvable inertial motion; known gravity and a deformation that responds measurably to it; independent rheometry; or fixing one parameter from an external measurement. Known timing and gravity do not guarantee adequate sensitivity—test it. If density is also free, stress and density scaling can create further compensation in dynamics.

### Viscoelasticity, viscosity and plasticity need different evidence

- A Kelvin–Voigt-type viscous stress addition allows creep/retarded recovery, but it is not equivalent to a generalized Maxwell or standard-linear-solid relaxation spectrum. At strictly held homogeneous strain, the Kelvin–Voigt viscous term vanishes after the loading ramp; it does not produce a sustained exponential decay of elastic stress. Check what the implemented model can express before choosing a relaxation test.
- A dimensionless deformation-gradient clamp is not automatically a measured yield stress. State the map from simulator controls to constitutive behavior.
- A series dashpot leaves unrecovered strain without a threshold. A rate-independent plastic element creates a different load–unload relationship. Slow finite-time viscoelastic recovery can resemble both over a short video.
- Contact damping, floor slip, tool adhesion, hidden-volume error, and camera/tool calibration can be traded against bulk viscosity or stiffness in a visual loss. Fix or characterize them independently where possible, and otherwise report joint sensitivity.

Consequently, first test recovery of known synthetic parameters under the actual observation operator, including visibility, timing, and noise. Then examine profile losses and parameter-pair sensitivity on real data. A numerically sharp optimum does not overcome a wrong constitutive model or incorrect observation geometry.

## 10. Scientific novelty routes supported by this review

A strong framework paper can state and test the following hypotheses without claiming a new MPM solver or a universal dough law:

1. **Excitation-dependent identifiability:** multiple short motion primitives designed to separate reversible, rate-dependent, and residual response yield more stable parameter estimates and better withheld-motion prediction than one long arbitrary tool trajectory.
2. **Partial-observation sufficiency:** single-view depth can support useful prediction for stated operations, even when some absolute material parameters are weakly identifiable; quantify what adding a force trace, independent mass, or an extra view changes.
3. **Model-selection rather than only parameter selection:** compare elastic, elastic-plus-viscous, and elastic/viscous/plastic variants at comparable evaluation budgets. A more complex model is justified only by unseen-motion performance, not lower training loss.
4. **Interface–bulk separation:** estimating/checking tool contact before fitting bulk parameters reduces compensation and improves transfer across tool orientations or surfaces. Show an interaction between interface errors and inferred parameters rather than merely adding an SDF option.
5. **Food-specific repeatability:** determine whether identified effective parameters are stable across independently prepared specimens of a controlled recipe, and how they vary with hydration/resting history. This directly connects robotics calibration to food rheology.
6. **A benchmark connecting measurements to prediction:** archive controlled preparations, depth recordings, calibrated tools/trajectories, mass and optional force traces, and fixed train/validation/test trials. Compare transparent forward sweeps against at least default/unfitted models and a budget-matched non-grid search where appropriate.

These are plausible contributions, not established absences in the complete literature. The novelty assessment should be conditional: the paper’s contribution would be a measured answer or a released resource, rather than the combination’s description alone.

## 11. Search log

WebSearch queries executed in this branch (12):

1. `Fabbri Cevoli 2015 Estimation Semolina Dough Rheological Parameters inversion finite elements model pdf`
2. `MIRANDA RELAPP viscoelastic material characterization computer vision robotics dough 2025`
3. `wheat flour dough nonlinear viscoelastic rheology large deformation relaxation recovery constitutive model review`
4. `Tracking elastic deformable objects RGB D sensor pizza chef robot Petit 2017 material parameters`
5. `"Wheat Flour Quality Assessment by Fundamental Non-Linear" "PMC"`
6. `dough robot rheology inverse viscosity elasticity plasticity depth force relaxation creep camera identification`
7. `"First engineering framework" "rheological" dough model identification`
8. `dough "inverse" "viscoplastic" "identification" robot`
9. `"Linear to Non-linear Rheology of Wheat Flour Dough" pdf`
10. `"Modeling the viscoelastic behavior of wheat flour dough prepared from a wide range of formulations" authors`
11. `dough adhesion friction wall slip rheometry constitutive characterization robots dough nonlinear`
12. `"viscoelastic" "dough" "elastoplastic" material model relaxation`

Follow-up metadata checks used Crossref exact DOIs and exact-title queries for rolling and adhesion. Full-text follow-up used the publisher Fabbri PDF, Strathprints accepted manuscript for Cocuzza–Yan, and Europe PMC XML for MIRANDA/RELAPP and Yazar. The Ghorbel–Launay primary abstract was retrieved through Europe PMC. Several blocked publisher pages yielded metadata only; the relevant evidence entries say so explicitly. No repository source or existing reports were edited.
