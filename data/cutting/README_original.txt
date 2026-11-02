================================================================================
  README — Dataset and Analysis Scripts
================================================================================

Title:   Surface Roughness Prediction and Tool Wear Analysis in Steel Turning
         Using Machine Learning and Bayesian Optimisation

Author:  M. Gombár
         University of West Bohemia, Faculty of Mechanical Engineering
         Pilsen, Czech Republic

Journal: Journal of Manufacturing Processes (Elsevier)
Date:    2026

================================================================================
  CONTENTS
================================================================================

DATA FILES
----------
  DATA_Final.xlsx         Primary dataset — all experimental measurements
                          Sheets:
                            1_Info            Experiment metadata and conditions
                            2_Rz_TimeSeries   424 time-series Rz measurements
                                              (6 cutting speeds, cols: obs | vc |
                                               tau | Rz | Ra)
                            3_VB_TimeSeries   42 flank wear VB measurements
                                              (cols: obs | vc | tau | VB | VB_pct)
                            4_Wide_Rz_VB      Wide format for plotting
                            5_DOE_RzRa        40 DOE measurements (Rz + Ra)
                            6_MATLAB_Import   Clean numeric matrix with import
                                              commands

MATLAB SCRIPTS
--------------
  S0_LoadData.m           Load and preprocess all data from DATA_Final.xlsx
                          Output: DATA.mat

  S1_DOE_Analysis.m       Full DOE analysis
                          - ANOVA with effect size (eta2, Cohen f)
                          - Effect estimates (main effects + interactions)
                          - 2FI regression model (coded variables)
                          - Power-law model (physical variables)
                          - Bootstrap 95% CI on power-law exponents (B=1000)
                          - Residual diagnostics
                          Output: DOE_Results.mat, figures/Fig1-Fig5

  S2_DOE_ML.m             Machine learning model comparison
                          - GPR (kernel selection: SE, Matern52, Matern32, ARD-SE)
                          - ANN (architecture search)
                          - SVR (hyperparameter grid search)
                          - LOO-CV validation for all models
                          - Metrics: R2, RMSE, MAE, MAPE
                          Output: ML_Results.mat, figures/Fig6-Fig8

  S3_DOE_Optimization.m   Optimisation of cutting conditions
                          - Bayesian optimisation (Expected Improvement Plus, 60 iter.)
                          - Genetic algorithm (population 200, 300 generations)
                          - Pareto front (25^3 grid, non-dominated sorting)
                          - Harrington desirability function
                          Output: OPT_Results.mat, figures/Fig9-Fig11

  S4_Rz_TimeSeries.m      Time-series analysis of surface roughness
                          - Nonlinear regression Rz = A*exp(b*tau) per vc (nlinfit)
                          - 95% CI on parameters (nlpredci)
                          - Parameter trends A(vc) and |b|(vc)
                          - Unified model Rz = f(tau, vc)
                          Output: TS_Results.mat, figures/Fig12-Fig14

  S5_VB_Analysis.m        Tool wear analysis
                          - Power-law fit VB = C*tau^m per vc
                          - Taylor tool-life equation T = CT*vc^(-nT)
                          - Bootstrap 95% CI on Taylor exponent (B=2000)
                          Output: VB_Results.mat, figures/Fig15-Fig16

  S6_Rz_VB_Correlation.m  Rz-VB correlation analysis
                          - PCHIP interpolation of Rz at VB time points
                          - Pearson + Spearman correlation (per vc and pooled)
                          - Partial correlation (controlling for vc)
                          - Regression models Rz = f(VB, vc): 4 model comparison
                          Output: CORR_Results.mat, figures/Fig17-Fig18

  S7_Figures.m            Publication-quality composite figures
                          - FigA: DOE summary
                          - FigB: ML summary
                          - FigC: Time dynamics (dual-axis Rz + VB)
                          - FigD: Key finding (Rz-VB correlation)
                          Output: figures/FigA-FigD

  S_DOE_Cube.m            2^3 DOE cube visualisation for Materials & Methods
                          Output: figures/Fig_DOE_Cube

  S_ExperimentalSetup.m   Experimental setup schematic for Materials & Methods
                          Output: figures/Fig_ExperimentalSetup

================================================================================
  EXPERIMENTAL CONDITIONS
================================================================================

Workpiece material : C45 (EN 10083-2, AISI 1045), normalised condition
                     Rm = 620-850 MPa, Re = 340-490 MPa, 179-229 HB
                     Cylinder: diameter 60 mm, length 300 mm

Machine            : Centre lathe MUS 25 (SU 50)
Cutting tool       : KNUX 190408-EL P20 (ISO 1832)
Tool holder        : MWLNR
Nose radius        : r_epsilon = 0.80 mm
Edge angle         : kappa_r = 75 deg
Coolant            : None (dry machining)

DOE conditions:
  Cutting speed vc  : 8.792 / 351.680 m/min  (low / high)
  Feed rate f       : 0.10  / 0.50 mm         (low / high)
  Depth of cut ap   : 0.10  / 3.00 mm         (low / high)
  Replications      : m = 5 per run, N = 40 total

Wear/time-series conditions:
  Feed rate f       : 0.10 mm (fixed)
  Depth of cut ap   : 0.20 mm (fixed)
  Cutting speeds    : 6, 25, 70, 110, 140, 240 m/min
  VB criterion      : 0.30 mm (ISO 3685)

Surface roughness measurement:
  Instrument        : Mitutoyo Surftest SJ-301
  Standard          : ISO 4287
  Evaluation length : lm = 12.5 mm
  Cut-off           : lambda_c = 2.5 mm
  Filter            : Gaussian

================================================================================
  SOFTWARE REQUIREMENTS
================================================================================

MATLAB version     : R2025b

Required Toolboxes :
  - Statistics and Machine Learning Toolbox
    (fitrgp, fitrsvm, fitrnet, bayesopt, nlinfit, nlpredci,
     nlparci, corr, lillietest, jbtest)
  - Global Optimization Toolbox
    (ga)
  - Optimization Toolbox
    (optimoptions, optimizableVariable)

No additional third-party packages required.

================================================================================
  USAGE
================================================================================

1. Place DATA_Final.xlsx and all S*.m scripts in the same folder.
2. Set MATLAB working directory to that folder.
3. Run scripts in order:

   >> run('S0_LoadData.m')        % generates DATA.mat
   >> run('S1_DOE_Analysis.m')    % generates DOE_Results.mat
   >> run('S2_DOE_ML.m')          % generates ML_Results.mat  (~5 min)
   >> run('S3_DOE_Optimization.m')% generates OPT_Results.mat (~5 min)
   >> run('S4_Rz_TimeSeries.m')   % generates TS_Results.mat
   >> run('S5_VB_Analysis.m')     % generates VB_Results.mat
   >> run('S6_Rz_VB_Correlation.m')% generates CORR_Results.mat
   >> run('S7_Figures.m')         % generates composite figures

   All figures are saved as PDF (vector) and PNG (300 DPI)
   in a subfolder named 'figures/' next to the scripts.

   Total runtime: approximately 15-20 minutes (dominated by S2 LOO-CV
   and S3 Bayesian optimisation).

================================================================================
  NOTES
================================================================================

- vc = 25 m/min is excluded from Taylor tool-life analysis (R2 = -0.34,
  built-up edge regime; see Section 3.5 of the article).

- vc = 140 m/min is excluded from Taylor analysis because the experiment
  ended at VB = 0.15 mm (50% of VBcrit = 0.30 mm, ISO 3685).

- The unified Rz time-series model (S4) achieves R2 = 0.516 due to the
  absence of a monotonic A(vc) trend. Per-vc individual models (Table 5
  of the article) are recommended for practical prediction.

- Random seed is fixed (rng(42)) in S2 and S3 for reproducibility.

================================================================================
  LICENSE
================================================================================

This dataset and code are released under the Creative Commons
Attribution 4.0 International License (CC BY 4.0).
https://creativecommons.org/licenses/by/4.0/

You are free to share and adapt the material for any purpose, provided
appropriate credit is given to the original publication.

================================================================================
  CITATION
================================================================================

If you use this dataset or code, please cite:

  Gombar M. Surface Roughness Prediction and Tool Wear Analysis in Steel
  Turning Using Machine Learning and Bayesian Optimisation.
  Journal of Manufacturing Processes. [year];[vol]:[pages].
  doi:[DOI]

  Dataset: Gombar M. (2026). Dataset and analysis scripts for surface
  roughness and tool wear study in C45 steel turning [Data set].
  Mendeley Data. doi:[10.17632/yz37pvtztx.1]

================================================================================
