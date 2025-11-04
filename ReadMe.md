# AIPI 510 Project 2
## Statistics

### Students
Dominic Tanzillo
Roshan Gill

## Overview

This project investigates whether perceived sex (male vs female) predicts backpack carrying behavior (left, right, both, or none) among Duke University freshmen exiting the C1 bus.
The dataset is simple but demonstrates how to structure an observational behavioral study into a formal Chi-Square Test of Independence, including a power analysis and post-hoc residual testing.

Repository Structure
├── data/
│   ├── dominic_obs.csv
│   ├── roshan_obs.csv
│   └── backpack_counts.xlsx      # merged dataset output
│
├── src/
│   └── power_stats.py            # helper function for power analysis
│
├── Freshman_Backpacks.ipynb      # main analysis notebook
├── bag_observation_matrix.csv    # final cleaned contingency table
└── README.md

### Packages

`pip install pandas numpy matplotlib seaborn scipy`

### Reproduction Steps
Step 1: Run Power Analysis

File: `src/power_stats.py`

The function `n_for_power(w, df=3, alpha=0.05, power=0.8)` calculates the minimum sample size needed for a desired power level using a binary search over the non-central chi-square distribution.

Example:

```
from src.power_stats import n_for_power
n_for_power(0.2, df=3, alpha=0.05, power=0.8)
```

Step 2: Load and Combine Observations

In the notebook (`Freshman_Backpacks.ipynb`):

```
import pandas as pd

data_dir = "data"
files = ["dominic_obs.csv", "roshan_obs.csv"]

dfs = []
for f in files:
    df = pd.read_csv(f"data/{f}", index_col=0)
    df.columns = ["Female", "Male"]
    dfs.append(df)

combined = sum(dfs)
```

Step 3: Run Chi-Square Test

```
from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(combined)
print(f"Chi-Squared statistic: {chi2:.2f}, df = {dof}, p = {p:.3e}")
```

Step 4: Compute and Visualize Residuals

```
import numpy as np, seaborn as sns, matplotlib.pyplot as plt

residuals = (combined - expected) / np.sqrt(expected)
sns.heatmap(residuals, annot=True, cmap="RdBu_r", center=0)
plt.title("Standardized Residuals")
plt.show()
```

Step 5: Compute P-Values per Cell
```
from scipy.stats import norm

pvals = 2 * (1 - norm.cdf(np.abs(residuals)))
pvals = pd.DataFrame(pvals, index=residuals.index, columns=residuals.columns)
sns.heatmap(pvals.round(3), annot=True, cmap="RdBu_r", center=0)
plt.title("Cell-Specific P-Values")
plt.show()
```

Step 6: Optional to Collapse Left/Right into “One Shoulder”
```
one_shoulder = (
    combined.rename(index={"L": "One", "R": "One"})
    .groupby(level=0)
    .sum()
)
chi2_contingency(one_shoulder)
```
## Results Summary

Sample size: 602 observations
Chi-Square statistic: 28.59
Degrees of freedom: 3

p-value: $2.72 \times 10^{-6}$

## Interpretation:

- Females were significantly more likely to wear backpacks on both shoulders.
- Males were significantly more likely to carry no backpack.
- No significant differences were found for one-shoulder (left/right) carrying.

## Limitations

- Perceived sex coded visually may not match actual gender identity.
- Weather (light drizzle) likely affected behavior.
- Observations limited to a single day and bus route (C1).

## References

- U.S. News (2024). Duke University Student Life Report.
- Stanford Medicine (2017). How Men’s and Women’s Brains Differ.
- Anishnama (2023). Understanding the Chi-Square Test .