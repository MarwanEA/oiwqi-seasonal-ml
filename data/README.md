# Input data

The original data are not distributed with this repository; they are available from the corresponding author of the associated study on reasonable request.

To run the pipeline on your own data, put one CSV table per hydrological phase in a folder and pass it with
`--data-dir`. With the default configuration the files are:

| Season key | File |
|------------|------|
| `dry` | `dry.csv` |
| `first_flush` | `first_flush.csv` |
| `dilution` | `dilution.csv` |

Season keys, file names and column names can be changed in a configuration file (`data.seasons`,
`data.files`, `data.station_column`, `data.target`, `data.ec_raw_column`).
Files are read as UTF-8 (a byte-order mark is accepted) and column names are stripped of spaces.

## Columns

| Column | Meaning | Unit |
|--------|---------|------|
| `Station` | Well identifier (text) | |
| `NO3` | Nitrate | mg/L |
| `pH` | pH | |
| `EC` (or `CE`) | Electrical conductivity; a column named `CE` is renamed to `EC` | µS/cm |
| `TDS` | Total dissolved solids | mg/L |
| `HCO3` | Bicarbonate | mg/L |
| `Cl` | Chloride | mg/L |
| `Mg` | Magnesium | mg/L |
| `Ca` | Calcium | mg/L |
| `Na` | Sodium | mg/L |
| `K` | Potassium | mg/L |
| `SO4` | Sulphate | mg/L |
| `SAR` | Sodium adsorption ratio, Na / sqrt((Ca + Mg) / 2), ions in meq/L | |
| `Na%` | Percent sodium, (Na + K) / (Ca + Mg + Na + K) × 100, meq/L | % |
| `MAR` | Magnesium adsorption ratio, Mg / (Ca + Mg) × 100, meq/L | % |
| `PI` | Permeability index as implemented, (Na + HCO3) / (Ca + Mg + Na + K) × 100, meq/L | % |
| `RSBC` | Residual sodium bicarbonate as implemented, HCO3 − (Ca + Mg), meq/L | meq/L |
| `SSP` | Soluble sodium percentage, Na / (Ca + Mg + Na + K) × 100, meq/L | % |
| `O_IWQI` | Target index; `src/oiwqi_ml/index.py` recomputes it from the columns above | |

Other columns (for example a category label) are ignored. Step 1 reports missing columns, missing values,
duplicate stations and non-numeric values as errors, and values outside broad plausible ranges as warnings.

## Example row (fake values)

```
Station,NO3,pH,CE,TDS,HCO3,Cl,Mg,Ca,Na,K,SO4,SAR,Na%,MAR,PI,RSBC,SSP,O_IWQI
S01,12.5,7.6,950,608,310,95,38,82,60,6.2,75,1.37,27.72,43.32,77.0,-2.14,26.13,18.79
```

`python scripts/run_all.py --synthetic` writes complete synthetic tables in this format to `data/synthetic/`.
