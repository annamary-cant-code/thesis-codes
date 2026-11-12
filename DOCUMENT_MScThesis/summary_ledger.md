# Summary ledger: chain map of the airbag design workflow

Source of truth: the thesis `.tex` files only (read in full: `report.tex`, frontmatter,
Chapters 1-6, `chap4/step0`-`step4`, Appendices A-G, pseudocode).
Values are copied exactly as written in the thesis. Paths are relative to the repo root.

## 0. Rules found in the template / repo

- `tudelft-report.cls`: `book` class, 10 pt, oneside, A4, `hscale=0.75, vscale=0.8`.
  No rule on Summary/Abstract length, language or placement.
- `report.tex:175`: the Summary is `\input{frontmatter/summary}`, after the title page
  and preface, before the ToC. `frontmatter/summary.tex` uses `\chapter*{Summary}` +
  `\addcontentsline{toc}{chapter}{Summary}`.
- Language: `% LTeX: language=en-GB` in every chapter, `babel` with `british` → British
  spelling (characterisation, optimisation, modelling, phial, metre).
- README: generic TU Delft template docs only. No faculty instructions found anywhere in
  `CODES/` (searched for guideline/instruction/regulation/manual/abstract/summary/template).
- So the only constraints come from the prompt; nothing overrides it.

## 1. Chain map

The thesis defines the workflow as the *inner design loop* (Ch. 3, Sec. "Inner design
loop", `3_DesignWorkflow_METHODOLOGY.tex:149-179`): Steps 0-4 are the thesis
deliverable; Steps 5-7 are left to future developers. Section 4.6 "System design
hypothesis" is not a numbered step but turns Step 0 and Step 4 outputs into
recommendations, so it is listed as the closing row.

| Step | Purpose | Inputs and where each comes from | Method | Output (exact value + unit) | Used by / for | File, section |
|---|---|---|---|---|---|---|
| **0 Preliminary mission analysis** | Impact conditions of the falling payload | Route A (over Po and Stura, ~11 km, ~15 min) and Route B (straight line): prior research (Cavallo, Bertelli). Cruise ~75 m, 17 m/s: Bertelli ConOps. Failure scenario (UAV failure + parachute failure, ballistic fall from 75 m): assumption. Separation conditions as Gaussians, h 75±5 m, u_x,i 17±6 m/s, u_y,i 0±0.5 m/s, C_D 1.0±0.2: assumption built on ConOps. Wind 15 m/s, σ = 0, random direction: assumption ("conservative"). Payload M = 1.0 kg, S = 0.04 m²: assumption. ρ = 1.225 kg/m³, ISA. OSM map of Turin: data. Fall model: Angelini | OSM map → 500×500 terrain grid; B-spline trajectory sampled every 10 m; Monte Carlo, 5000 samples per trajectory point (1099 points Route A, 658 Route B); 3-phase ballistic descent with drag and wind drift | Terminal velocity Γ ≈ 20.0 m/s. Impact by terrain: Route A 59.2 % water, 14.4 % green; Route B 32.2 % buildings, 28.5 % roads, 33.6 % ground. Ranges: u_x 0-10 m/s, u_y 16-27 m/s, flight 4-6.5 s. **Design values (98.5th percentile): u_x0 = 7.5 m/s, u_y0 = 25 m/s.** t_fall 4-6.5 s (text) / 4-6 s (table) | u_x0, u_y0 → Step 3 (pulse area ‖u0‖ = 26.1 m/s) and Step 4 (oracle input; u_n0 ≈ 26.1 m/s at θ ≈ 16.7°). t_fall → not used in Steps 1-4; used in Sec. 4.6 (inflation within ~4 s) and Ch. 5. Terrain shares → nobody downstream | `chap4/step0_MissionAnalysis.tex`, Sec. 4.1 (tables `tab:uncertain_params`, `tab:step0-summary-data`, l. 165, 170, 192, 209-211) |
| **1 Axial compression test (experimental)** | Measure phial fragility (failure load) | 76 saline 5 ml phials, 70 mm × 15 mm, wall assumed 0.4-0.5 mm, from Molinette (Batch A 36, Batch B 40). Assumption: their glass equals the Akineton phial glass (Type I borosilicate, ISO 9187-1). MTS QTest/10, 10 kN cell, 1/2/3 mm/min. PETG adaptor | Compression to failure; failure modes classified A-E + hybrids; two-parameter Weibull fit per batch; post-hoc stiffness test of the PETG adaptor | Batch A favours mode A (53 % of Batch A), Batch B favours mode E. Batch B stiffer, stronger, more scattered → **only Batch A kept**. **F_fail = 98.7 N** (Batch A, Weibull 5th percentile, β = 2.44, α = 332.7 N, n = 36). Mode A stiffness 318-1506 N/mm; mode AD 259-1321 N/mm (14 % of Batch A). K_PETG = 2037 N/mm (fit 20-316 N) | F_fail → Step 2 calibration target for FT. Stiffness ranges + K_PETG → Step 2 stiffness check. Failure modes A, AD → Step 2 qualitative check; Step 3 failure-mode comparison | `chap4/step1_CompressionTest.tex`, Sec. 4.2 (`tab:step1-summary-data`, l. 107, 122-125) |
| **2 Axial compression test (FEM)** | Calibrated glass model of the phial | F_fail 98.7 N, stiffness ranges, K_PETG (Step 1). Geometry: CAD from calliper measurements (assumption-laden). Shell thickness 0.4 mm. E = 70 GPa, PR = 0.23: soda-lime literature values used for borosilicate. FC = 1 GPa (arbitrary), FTSCL = 1.2, other MAT_GLASS parameters from literature/manual. PETG adaptor not modelled | LS-DYNA explicit, shells, `MAT_GLASS` (MAT_280), Rankine criterion; modal analysis (f1 = 8.179 kHz, T1 ≈ 122 µs → 5 ms ramp); imperfection by 1° tilt; FT tuned until breaking load within 2 % of target; series-stiffness correction with PETG | **FT = 60 MPa → breaking load 96.9 N** (mode A) vs 98.7 N. Reproduces mode A (perfect geometry) and mode AD (tilted). K_eq point by point 1251 N/mm (mode A, inside 318-1506 N/mm) and 1154 N/mm (mode AD, ~10 % above Batch-A-only range 259-1049 N/mm). Model alone ≈ 3.7× / ≈ 2.8× stiffer than median. Deliverable: calibrated `MAT_GLASS` card | Calibrated card → Step 3 phial material. Crack flag HISV#1 → Step 3 failure detection (stricter criterion there) | `chap4/step2_CompressionFEM.tex`, Sec. 4.3 (l. 192, 290, 330, 363, 386; `lst:mat-glass`, `tab:step2-stiffness-summary`) |
| **3 Sled test (FEM)** | Allowable payload acceleration | Calibrated card (Step 2). ‖u0‖ = 26.1 m/s (Step 0). Payload assembly: inspired by Sivieri but **hypothetical**: 5 mm ABS shell, PU foam sponge (curve after Croop 2009), PP mask, one phial; materials from literature, sponge and mask cannot fail. Trapezoidal pulse, RISE_FRACTION = 0.2 (assumption). Mass scaling Δt = 2×10⁻⁵ ms | LS-DYNA explicit; acceleration pulse on two rigid rings of the bay; equal-area pulse family; direction chosen from a qualitative 600 g comparison in X, Y', Z' (Y', Z' rotated 35°); amplitude sweep refined from 100 g to 5 g; failure = HISV#1 ≥ 1 anywhere | **a_allow = 225 g** (no damage at 225 g, damage at 230 g, 5 g step). Critical direction Y' (Ch. 5: "perpendicular to the phial axis (the most critical case)"). Failure mode: dome breaking extending into the body, partly comparable to mode A | a_allow → Step 4 (a_lim = a_allow/SF ≈ 1839 m/s², i.e. 187.5 g). Critical direction, failure mode → nobody downstream | `chap4/step3_SledTestFEM.tex`, Sec. 4.4 (l. 37, 207, 248, `tab:step3-summary-data`) |
| **4 Design variables optimisation** | Lightest feasible inflated airbag | M = 1 kg (assumption; see gaps), u_x0 = 7.5 m/s, u_y0 = 25 m/s (Step 0), a_allow = 225 g (Step 3). SF = 1.2, η_seam = 0.70, SF_burst = 1.50, PEN = 10⁴: assumptions. Fabric data: rounded literature values (aramid extrapolated, least certain). Gas constants. T0 = 288 K (ISA), d_or = 0. Bounds: D0 from D0,min to 1.5 m, L0/D0 0.5-4.0, P0 P_amb to 1.4×10⁵ Pa, φ_vent 0-0.10, n_ply 1-4. Caps u_res,max = 0.5/1.0/1.5/2.0 m/s | Python oracle (Zhou et al. 1-DoF model extended to 2-DoF, sphere geometry, fabric porosity venting, seam-reduced burst check; termination by burst, bottoming, rebound, settling) inside SciPy differential evolution; 120 categorical combinations (2 shapes × 3 fabrics × 5 gases × 4 caps); 600 runs; MAXITER = 150; winner re-simulated at full fidelity | Same winner for all caps: **cylinder, silicone-coated Nylon 6,6, helium, 1 ply, L0/D0 = 0.50** (lower bound). D0 = 0.774 / 0.671 / 0.534 / 0.522 m for 0.5 / 1.0 / 1.5 / 2.0 m/s. P0 = 129.6 / 138.5 / 140.0 / 140.0 kPa (upper bound at 1.5, 2.0). m_tot = 0.379 / 0.282 / 0.176 / 0.167 kg ("below 400 g"). Peak 187.5 g (187.6 g at 2.0) = on a_lim. Burst margin 25.3 / 25.1 / 32.1 / 32.9 %. Outcome: settled at 0.5, bottomed out at 1.0-2.0 | Gas choice + P0 range → Sec. 4.6 and Ch. 5 recommendations (stored-gas He inflator; P0 129.6-140.0 kPa absolute). Nothing else downstream in this thesis; meant for Steps 5-7 | `chap4/step4_Optimisation.tex`, Sec. 4.5 (`tab:airbag-design-variables-selected`, `tab:step4-winners-per-cap`, l. 289-303) |
| *(4.6) System design hypothesis* | Turn outputs into recommendations for the next design phase | t_fall (Step 0), He choice (Step 4), literature on folding and seams | Qualitative | Helium must be stored as pressurised gas in small cylinders (feasibility and cylinder mass still to check). Activation + full inflation within the shortest flight time, about 4 s, including failure detection. Folding: zig-zag or rolling families. Seam pattern left for future work. Idea: control fall attitude so the bag covers only the impacting side | Steps 5-7 (future work); Ch. 5 sub-question 3 | `4_TestingDesignWorkflow_RESULTS.tex:20-40`; `5_Conclusions.tex:25-26` |
| *Steps 5-7* | Inflated configuration for real geometry; folding; deployment and impact tests | Need receptacle geometry and UAV hook-up, not available | Not carried out | **No output** | — | `3_DesignWorkflow_METHODOLOGY.tex:176-179` |

## 2. Outputs that no later step uses

1. Step 0 impact-by-terrain shares (Route A 59.2 % water, 14.4 % green; Route B 32.2 %
   buildings, 28.5 % roads, 33.6 % ground) and the impact heatmaps. The thesis says so:
   "These results do not necessarily contribute to the airbag design workflow"
   (`step0`, l. 167).
2. Step 0 time of flight (4-6.5 s): not used by Steps 1-4 ("will not be considered for
   the scopes of this thesis", l. 170), but it *is* used afterwards in Sec. 4.6 and Ch. 5.
3. Route B as a whole: impact conditions are "comparable" for both routes, and the design
   values are not tied to one route.
4. Step 1 Batch B data (discarded after the Weibull comparison).
5. Step 2 stiffness verification (K_eq 1251 / 1154 N/mm): a credibility check, not passed on.
6. Step 3 critical direction Y' and failure mode: Step 4 does not use them (its single
   impact axis n is at θ ≈ 16.7° with no link to the payload's orientation).
7. Step 4 φ_vent, A_vent,0, landing speeds, t_stop, outcomes, burst margins: reported, not
   passed on (no Step 5 in this thesis). Appendix E per-gas transients: same.

## 3. Inputs that do not come from an earlier step (limit validity)

Step 0
- Payload mass M = 1.0 kg, frontal area S = 0.04 m², C_D 1.0 ± 0.2.
- Wind fixed at 15 m/s with σ = 0 and random direction (Turin average is 1.8 m/s, 2013
  peak 28.6 m/s).
- Separation conditions (75 ± 5 m, 17 ± 6 m/s, 0 ± 0.5 m/s) built on the ConOps.
- Failure scenario: parachute fails, free ballistic fall.
- Choice of the 98.5th percentile as design value; the two velocity percentiles "do not
  necessarily correspond to the same impact event" (l. 192).

Step 1
- Saline phials stand in for Akineton phials (same glass assumed).
- Wall thickness only assumed (0.4-0.5 mm).

Step 2
- E = 70 GPa and PR = 0.23 from soda-lime glass; FC = 1 GPa arbitrary; FTSCL = 1.2.
- Geometry from calliper measurements; shell thickness 0.4 mm.
- FT = 60 MPa is "a calibrated model parameter rather than ... the actual tensile
  strength" (l. 200).

Step 3
- The whole payload assembly is hypothetical: ABS shell instead of Sivieri's PETG
  sandwich, PU sponge curve from literature, PP mask, one phial, sponge and mask cannot
  fail. The thesis states a_allow "depends strongly on the sponge" (`step4`, l. 303).
- Pulse shape (trapezoid, RISE_FRACTION 0.2).
- Loading direction from a qualitative comparison at 600 g, not a full sweep.

Step 4
- SF = 1.2, η_seam = 0.70, SF_burst = 1.50.
- Fabric properties are rounded literature values; aramid is extrapolated.
- Search bounds and the u_res,max caps.
- T0 = 288 K; no orifice.
- Model limits: rigid ground, fixed impact axis, crude coupling of horizontal and vertical
  velocity, bag flattens to zero thickness, no fabric heating, no helium permeation, no
  seam leakage.
- Not validated against tests.

## 4. Places where a step's output and the value used downstream differ

1. **Time of flight**: Step 0 text says 4-6.5 s (l. 170), and Sec. 4.6 and Ch. 5 also say
   4-6.5 s, but the Step 0 deliverable table says "4-6 s" (l. 211).
2. **Time of flight used / not used**: Step 0 says it "will not be considered for the
   scopes of this thesis" and is "not required by the following steps", but Sec. 4.6 and
   Ch. 5 use it for the inflation-time recommendation. (Consistent only if Sec. 4.6 counts
   as outside the workflow.)
3. **Payload mass**: Step 4 says 1 kg was "numerically estimated in Step 3" (l. 6). Step 3
   gives no mass value anywhere; the only source of the 1.0 kg is the Step 0 assumption.
4. **Target vs calibrated failure load**: 98.7 N target vs 96.9 N obtained with FT = 60 MPa
   (mode A), inside the stated 2 % tolerance. Mode AD "points to a different FT value"
   when taken alone (l. 363).
5. **Failure criterion changes between Step 2 and Step 3**: Step 2 takes failure at the
   force peak and ignores mesh-induced cracks; Step 3 counts any HISV#1 ≥ 1. The thesis
   calls the Step 3 allowable conservative for this reason (`step3`, l. 200).
6. **Loading direction**: Steps 1-2 calibrate under axial compression. Step 3 takes the
   allowable along Y', which Ch. 5 calls "perpendicular to the phial axis". Step 4 applies
   that allowable along an impact axis at θ ≈ 16.7° from vertical without a defined
   payload orientation. (The attitude-control idea in Sec. 4.6 is the only bridge.)
7. **Allowable → limit**: 225 g becomes a_lim = 225/1.2 = 187.5 g (≈ 1839 m/s²) in Step 4.
   Two winners (1.5 and 2.0 m/s) exceed a_lim by 0.017 % and 0.027 % and are feasible only
   within the 0.1 % re-simulation tolerance.
8. **Geometric imperfection**: Step 2 main text says option 2 (1° tilt) was chosen instead
   of the eigenmode perturbation (l. 192). Appendix C, row V8 (final model) lists "mode 18
   at 0.08 mm thickness perturbation via `PERTURBATION_NODE` + 1° tilt". (The current
   `frontmatter/summary.tex` says "geometric imperfections taken from its vibration
   modes", which matches Appendix C but not the main text.)
9. **Ch. 3 description of Step 4 vs Step 4 itself**: Ch. 3 says the code "optimises ...
   against critical criteria, including total mass, bottoming-out speed, and rebound".
   In Step 4, mass is the objective; bottoming speed and rebound are constraints/termination
   checks.
10. **Ch. 3 description of Step 2 output**: Ch. 3 names it the "brittle failure threshold
    (FT_fail)"; Step 2 reports FT = 60 MPa and the calibrated card, and never uses FT_fail.

## 5. Requirements stated at the start, and whether the end answers them

| Requirement (where stated) | Answered? (where) |
|---|---|
| Objective 1: propose an application-specific design workflow within the research line (Ch. 1, l. 13) | Yes: inner design loop, Ch. 3; Ch. 5 main question |
| Objective 2: test the workflow experimentally and through simulation (Ch. 1, l. 14) | Yes for Steps 0-4 (Ch. 4); Steps 5-7 not tested |
| Workflow "yields an optimised airbag-type device ... which brings the impact accelerations within the allowable limits of the payload container and of the medical payload" (Ch. 1, l. 11) | Partly: preliminary inflated sizing against the phial allowable only. No allowable is derived for the container (the ABS bay has a failure criterion, but no result is reported for it) |
| Commissioner req. 1: identify and characterise mechanical allowables of *typical* medical payloads based on general packaging requirements (Ch. 2, l. 327) | Partly: one phial type, Batch A only, one hypothetical assembly; a_allow = 225 g. Packaging requirements are not derived; Step 4 / Ch. 5 recommend defining them |
| Commissioner req. 2: informed conceptual design of an airbag able to deploy during emergency release and preserve the payload (Ch. 2, l. 328) | Partly: inflated configuration sized (Step 4); deployment only as recommendations (Sec. 4.6) |
| Main RQ: how can a multi-phase simulation workflow be developed ... (Ch. 2) | Answered (Ch. 5; Ch. 6 "Main question (workflow): answered") |
| SQ1: allowable mechanical limits and how they define the design envelope | Answered: 225 g + u_x0 7.5 m/s, u_y0 25 m/s define the envelope (Ch. 5). No stress allowable: FT is a calibrated parameter |
| SQ2: optimise inflated configuration while balancing requirements | Answered at pre-design level (Ch. 5, Ch. 6) |
| SQ3: recommendations for a compact deployment system | Recommendation level only; moved to future work (Ch. 5, Ch. 6) |
| R-D objective: ensure integrity and containment, prevent dispersion, support certification (Ch. 2 box) | Integrity is checked only through phial cracking in simulation. Support for certification is argued (Step 4, l. 301 "may be worth ... in view of certification") but not demonstrated |
| Literature requirements, Table 2.2 (`tab:requirements`) | Limit peak acceleration: yes (constraint). Reduce touchdown velocity: yes (u_res,max cap). Cope with horizontal velocity: partly (2-DoF, crude coupling). Different landing surfaces: no (rigid ground). Avoid fabric rupture: yes (burst check). Prevent bottoming-out: no; three of four winners bottom out, and only the landing-speed cap applies. Limit rebound: yes (rebound discards); rollover: no. Deploy within available time: no (only the ~4 s recommendation). Retain pressure: no (He permeation, seam leakage not modelled). Stowage, integration, temperature range, reuse, redundancy: no. Minimise mass: yes (objective) |
| Safety-first principle, item 4: integrate several devices (airbag + parachute) at the start (Ch. 3 box) | No; airbag + parachute is future work (Ch. 5, pathway 4) |

## 6. Noticed in passing (not resolved, not edited)

- `step0`, l. 192: "u_x=7.5\,m/" (unit missing an "s").
- `step4`, l. 133: the bottoming check reads "too high compared to the manually set
  threshold, i.e. u_land ≤ u_res,max, the combination is discarded" (inequality is the
  acceptance condition, not the rejection one).
- Ch. 3, l. 150: "divided into seven steps", while the figure and text number Steps 0-7.
