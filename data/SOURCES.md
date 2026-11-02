# Data sources

| Folder | Dataset | Licence / access | Used for |
|---|---|---|---|
| `cutting/` | M. Gombár (University of West Bohemia), *Dataset and MATLAB analysis scripts for surface roughness and tool wear study in C45 steel turning*, Mendeley Data, 2026. DOI [10.17632/yz37pvtztx.1](https://doi.org/10.17632/yz37pvtztx.1). File `DATA_Final.xlsx`, unchanged (renamed). | CC BY 4.0 | Stage 1: roughness model, Taylor tool life, measurement scatter |
| `scheduling/` | S. Jun, *PMSP_SDST_TWT* benchmark (unrelated parallel machines, sequence-dependent setup times, total weighted tardiness), GitHub [sjun-dgu/PMSP_SDST_TWT](https://github.com/sjun-dgu/PMSP_SDST_TWT), 2025. Instances converted from JSON to one compressed `.npz` per set (values unchanged); `Best_Solutions_EO.xlsx` unchanged. | MIT (see `scheduling/LICENSE`) | Stage 2: benchmark and this week's order book |
| `maintenance/` | Kharazian, Lindgren, Magnússon, Steinert, Andersson Reyna, *SCANIA Component X dataset*, Swedish National Data Service, DOI [10.5878/jvb5-d390](https://doi.org/10.5878/jvb5-d390); article in Scientific Data (2025), DOI 10.1038/s41597-025-04802-6. File `train_tte.csv`. | Open access via SND ("data are freely accessible"); no explicit redistribution licence, so the file is **not** in the repository. Download `train_tte.csv` and save it as `maintenance/scania_component_x_train_tte.csv`. | Stage 3: component lifetimes with censoring |
| `inventory/` | Monthly sales of car spare parts, Jan 1998 - Mar 2002, from R. J. Hyndman, A. B. Koehler, J. K. Ord, R. D. Snyder, *Forecasting with Exponential Smoothing* (Springer, 2008), R package `expsmooth`, dataset `carparts`. Exported to CSV, values unchanged. | Distributed with the GPL R package `expsmooth` | Stage 4: spare-part demand |

All costs (machine rate, insert price, repair and breakdown costs, part values) are **assumptions**, because none of
the datasets contains prices. They are listed in the code (`plant.py`, `maintenance.COSTS`, `inventory.COSTS`).
