# Overleaf review comments

Project: `askara_icra2026`  
Overleaf project ID: `6a96b07076bd7fc6a0f4485a`  
Captured: 2026-09-14  
Reviewer shown by Overleaf: `maric.bruno`

The Review overview contained **17 unresolved comments** and **10 tracked text changes** across three files. The revision target is an ICRA-format, six-page, conservative system paper. The existing title, section order, method, reported experiment, and figures should remain recognizable; changes should clarify and correct the current paper rather than recast it. Numerical claims may be retained only when supported by stored, reproducible artifacts. The five-parameter `E`, `ν`, `η`, `σ_min`, `σ_max` run remains as a preliminary result rather than final material calibration. The revision must not discuss whether a program or operator stopped the run; any convergence statement must be based on the saved optimization history itself.

Exact browser data and source-offset context are preserved in:

- `comments_browser_raw.json`
- `overleaf_review_items.json`

Status labels below compare each comment with the downloaded project version at the time of capture. They do not change or resolve anything on Overleaf.

## `content/01_into_related.tex`

### 1. Clarify the description of the intended application

- **Location:** line 15, near “future bimanual deformable object manipulation practices”
- **Original comment:** `to je?`
- **Meaning:** What exactly does “practices” mean here?
- **Status:** **Open.** The phrase remains vague and unidiomatic.
- **Local response:** Replace it with a precise statement that the present paper studies observation-conditioned simulation and identification for future deformable-object manipulation.

### 2. Do not restrict the broader system to spatulas

- **Location:** line 15, near the present two-tool setup
- **Original comment:** `ne bi se ogranicio samo na spatulu. Bit ce u buducnosti i silikonska ruka`
- **Meaning:** Do not define the broader system only around spatulas; a silicone hand may be used later.
- **Status:** **Partly addressed.** The paper distinguishes the current two-spatula experiment from future work, but several broad claims still say “spatulas” where “tools or end effectors” is more accurate.
- **Local response:** State that the recorded experiment uses two spatulas, while the pipeline is intended for registered tools or end effectors more generally.

### 3. Rewrite or remove the identifiability paragraph

- **Location:** line 18
- **Original comment:** `cijeli ovaj paragraf mi je meh...`
- **Meaning:** The reviewer dislikes the entire paragraph.
- **Status:** **Handled by removal, but its useful qualification should not disappear.** The paragraph is now commented out. Its main caution—parameters are conditional on the selected model and observation setup—is already stated in the abstract, Method, Results, and Conclusion.
- **Local response:** Keep the long paragraph removed. Retain one concise sentence in the Introduction explaining that the paper estimates model-dependent effective parameters rather than universal dough constants.

### 4. Soften the emphasis on future robot experiments

- **Location:** line 18, near the sentence saying robot execution and synthetic-data learning are future work
- **Original comment:** `ovo treba ublaziti nekako. Reci da je naglasak na ovom prethodno sad, a robot nije u fokusu.`
- **Meaning:** Soften this statement; emphasize the current identification work and clarify that robot execution is not the focus.
- **Status:** **Mostly addressed.** The abstract and pipeline caption now identify robot execution as downstream work, but the Introduction can state the paper’s present scope more directly.
- **Local response:** Describe the completed contribution first. Mention future robot-controlled data collection only once, without presenting it as a reported experiment.

### 5. Add an attractive introductory figure

- **Location:** before the pipeline figure
- **Original comment:** `prije ove slike treba neku INTRO sliku atraktivnu staviti. Mozda prikaz simulatora.`
- **Meaning:** Add an attractive introductory figure, possibly a simulator view, before the pipeline diagram.
- **Status:** **Addressed.** Figure 1 now presents the reconstructed dough and two simulation tools before the pipeline diagram.
- **Local response:** Keep Figure 1, improve its caption, and correct the label typo `fig:simiulation`.

### 6. Incomplete contribution list

- **Location:** immediately after the contribution list and before the pipeline figure
- **Original comment:** `?`
- **Status:** **Open.** The second and final bullet ends with “and,” but no third contribution follows.
- **Local response:** Provide a complete three-item contribution list covering metric reconstruction/tool replay, differentiable partial-view identification, and an honest experimental result/limitations contribution.

### 7. Broaden Related Work

- **Location:** transition into Related Work
- **Original comment:** `Rekao bi da related work u ovoj formi obuhvaca samo ono sto si ti koristio u ovom radu. To nije poanta related worka. Related work treba sagledati siru sliku. Reci nesto o tom kako ljudi rade s deformabilnim objektima, sto su oni, sto su izazovi, koji su pristupi dostupni u State of the art. Onda mozes uci u neke specificne stvari (subsectioni kako si stavio) ali i tu bi trebao dati pregled sta sve postoji i zasto si odabrao ovo sto si odabrao.`
- **Meaning:** Related Work should survey the broader deformable-object-manipulation field, its object classes, challenges, and available approaches before narrowing to the methods used here; it should also explain why the selected approach was chosen.
- **Status:** **Substantially addressed.** The current version now covers deformable-object manipulation, simulation families, differentiable physics, visual real-to-simulation identification, robotic shaping, and dough mechanics. It still needs a sharper comparison between alternatives and the paper’s design choices.
- **Local response:** Keep the expanded structure, reduce self-description inside the survey paragraphs, and add concise rationale for single-view metric depth, MLS-MPM, measured tool trajectories, and partial-view fitting.

## `content/02_method.tex`

### 8. Add a short Method overview before the inverse problem

- **Location:** line 4, opening sentence
- **Original comment:** `Malo mi je ovo previse in medias res...`
- **Meaning:** The section begins too abruptly.
- **Status:** **Open.** It still starts immediately with mathematical notation.
- **Local response:** Add a short paragraph describing the measurement, reconstruction, replay, simulation, rendering, and optimization sequence before introducing symbols.

### 9. Describe the experimental setup before calibration details

- **Location:** line 29, near the AprilTag/mocap calibration sentence
- **Original comment:** `ovo mi je isto iz neba pa u rebra. Odakle, kako, gdje je smjesteno? Mozda prije ovoga treba neki mali setup opisati.`
- **Meaning:** The calibration description lacks setup context: what components are used and where they are placed.
- **Status:** **Partly addressed, but in the wrong order.** A Data Collection subsection now names the RealSense D435 and OptiTrack system, but it appears after the simulation and optimization subsections.
- **Local response:** Move and expand “Experimental Setup and Data Collection” to the beginning of Method, then describe camera-to-mocap, table, and marker-to-tool calibration.

### 10. Cite or define the contact model

- **Location:** line 50, near “unilateral Coulomb approximation”
- **Original comment:** `referencirati?`
- **Meaning:** Should this contact/friction formulation be cited?
- **Status:** **Open.** No citation or derivation is supplied.
- **Local response:** Either cite the implementation’s source if one is already supported by the bibliography/repository, or explicitly label it as the simulator’s implemented contact rule and define it sufficiently without implying a standard exact formulation.

### 11. Define the SVD factors

- **Location:** line 89, `F_p=U\Sigma V^T`
- **Original comment:** `sta su U i V?`
- **Meaning:** What are `U` and `V`?
- **Status:** **Open.** They are not defined.
- **Local response:** State that `U` and `V` contain the left and right singular vectors of `F_p`.

### 12. Define the singular values and stretch limits

- **Location:** line 90, near `\sigma_{\min}` and `\sigma_{\max}`
- **Original comment:** `a sta su sigme?`
- **Meaning:** What are the sigma quantities?
- **Status:** **Partly addressed.** The text says they are dimensionless principal-stretch bounds rather than yield stresses, but it does not explicitly define `\Sigma` or each singular value.
- **Local response:** Define `\Sigma=\operatorname{diag}(\sigma_1,\sigma_2,\sigma_3)` and distinguish the singular values from the lower/upper clipping bounds.

### 13. Explain the treatment of unknown pixels

- **Location:** line 108, “Unknown pixels are not treated as empty space.”
- **Original comment:** `are or are not? ako nisu kako su onda tretirani?`
- **Meaning:** Confirm whether unknown pixels are treated as empty; if not, explain how they enter the objective.
- **Status:** **Open.** The statement gives no operational definition.
- **Local response:** State exactly which observed pixels contribute to each loss term and that invalid/unobserved pixels are masked out rather than penalized as free space.

### 14. Rename the subsection

- **Location:** line 138, `Data Collection`
- **Original comment:** `Mozda preimenovati u Method`
- **Meaning:** Consider a more method-oriented subsection name.
- **Status:** **Open.** The heading remains `Data Collection`.
- **Local response:** Rename it to `Experimental Setup and Data Collection` and place it before reconstruction and simulation.

### 15. Explain the figure/setup in the prose

- **Location:** line 138, at the Data Collection subsection
- **Original comment:** `napisati u tekstu sto je`
- **Meaning:** Explain in the text what is shown or what the setup consists of.
- **Status:** **Partly addressed.** The caption and two paragraphs identify the operator, camera, OptiTrack system, tools, and robot-based marker calibration, but the prose does not connect these elements into a complete acquisition sequence.
- **Local response:** Add a compact setup paragraph and refer explicitly to the data-collection figure.

## `content/03_results.tex`

### 16. State the contribution to the state of the art honestly

- **Location:** beginning of Results and Discussion
- **Original comment:** `Mozemo li s rezultatima reci da su oni doprinijeli razvoju SOTA(state of the art), odnodno da su pomaknuli nekako postojeci SOTA?`
- **Meaning:** Can the results support a claim that the work advances the state of the art?
- **Status:** **Open question; the current evidence does not support a benchmark-leading claim.** The reported optimization is interrupted, uses one sequence, and is not compared quantitatively with competing methods.
- **Local response:** Claim a demonstrated end-to-end single-view pipeline and report the measured objective reduction and failure modes. Do not claim state-of-the-art performance. Present the contribution as a carefully validated system integration and diagnostic result that identifies requirements for reliable future parameter identification.

### 17. Replace the vague figure-introduction sentence

- **Location:** line 18, “show the camera-matched sequence and differentiable optimization history”
- **Original comment:** `To je sta?`
- **Meaning:** What specifically are these plots/results?
- **Status:** **Open.** The sentence remains too generic.
- **Local response:** State what sequence is compared, which interval was optimized, what each row/curve represents, and what conclusion can and cannot be drawn.

## Tracked changes visible in the Review panel

The Review overview also showed **10 tracked changes**: one in `01_into_related.tex` and nine in `02_method.tex`. The downloaded source already contains their resulting text, including:

- addition of “system (Mocap)”;
- replacement of “We” with “Let us” at the Method opening;
- insertion of the inverse-problem equation;
- insertion of several comment markers around an older equation;
- the phrase “given with:” and two added colons.

These should be edited for final prose quality rather than accepted blindly. The live Overleaf project has not been modified.
