import sys
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# Load data
file1 = sys.argv[1]
file2 = sys.argv[2]

data1 = pd.read_csv(file1, header=0, sep=",")
data2 = pd.read_csv(file2, header=0, sep=",")

# Combine data
data = pd.concat([data1, data2], axis=1)

# Create contingency table
contingency_table = pd.crosstab(data['model_answer'], data['predictions'])

# Perform McNemar test
result = mcnemar(contingency_table, exact=True)

# Output results
print(f'Statistic: {result.statistic}, p-value: {result.pvalue}')
