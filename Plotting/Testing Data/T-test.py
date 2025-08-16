import pandas as pd
from scipy.stats import ttest_rel

# Load your results
phcapam_df1 = pd.read_excel("PHCAPAM_TestSample1_BestModel_results.xlsx")
capam_df1   = pd.read_excel("CAPAM_TestSample1_BestModel_results.xlsx")

# Metrics to test
metrics = ["Reward", "EnergySupplied", "VoltageViolation"]
print("Test Sample 1")
# Run paired t-tests
for metric in metrics:
    t_stat1, p_val1 = ttest_rel(phcapam_df1[metric], capam_df1[metric], nan_policy='omit')
    print(f"Metric: {metric}")
    print(f"  PHCAPAM Mean: {phcapam_df1[metric].mean():.4f}")
    print(f"  CAPAM Mean:   {capam_df1[metric].mean():.4f}")
    print(f"  t-statistic:  {t_stat1:.4f}")
    print(f"  p-value:      {p_val1:.4e}")
    print()


#----------- Test Sample 2 --------------------------
# Load your results
phcapam_df2 = pd.read_excel("PHCAPAM_TestSample2_BestModel_results.xlsx")
capam_df2   = pd.read_excel("CAPAM_TestSample2_BestModel_results.xlsx")

# Metrics to test
metrics = ["Reward", "EnergySupplied", "VoltageViolation"]
print("Test Sample 2")
# Run paired t-tests
for metric in metrics:
    t_stat, p_val = ttest_rel(phcapam_df2[metric], capam_df2[metric], nan_policy='omit')
    print(f"Metric: {metric}")
    print(f"  PHCAPAM Mean: {phcapam_df2[metric].mean():.4f}")
    print(f"  CAPAM Mean:   {capam_df2[metric].mean():.4f}")
    print(f"  t-statistic:  {t_stat:.4f}")
    print(f"  p-value:      {p_val:.4e}")
    print()


#----------- Test Sample 3 --------------------------
# Load your results
phcapam_df3 = pd.read_excel("PHCAPAM_TestSample3_BestModel_results.xlsx")
capam_df3   = pd.read_excel("CAPAM_TestSample3_BestModel_results.xlsx")

# Metrics to test
metrics = ["Reward", "EnergySupplied", "VoltageViolation"]
print("Test Sample 3")
# Run paired t-tests
for metric in metrics:
    t_stat, p_val = ttest_rel(phcapam_df3[metric], capam_df3[metric], nan_policy='omit')
    print(f"Metric: {metric}")
    print(f"  PHCAPAM Mean: {phcapam_df3[metric].mean():.4f}")
    print(f"  CAPAM Mean:   {capam_df3[metric].mean():.4f}")
    print(f"  t-statistic:  {t_stat:.4f}")
    print(f"  p-value:      {p_val:.4e}")
    print()