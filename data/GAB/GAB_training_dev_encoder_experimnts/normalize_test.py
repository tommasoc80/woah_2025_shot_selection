import sys
import pandas
import pandas as pd
from sklearn.metrics import classification_report

def assign_value(row):
    if row['hd'] == 0 and row['cv'] == 0:
        return 0
    else:
        return 1

inputf = sys.argv[1]

input_df = pd.read_csv(inputf, sep="\t", header=0)
nbr_rows = input_df.shape[0]
input_df = input_df.drop('vo', axis=1)
input_df['gold_hate'] = input_df.apply(assign_value, axis=1)
input_df = input_df.drop('hd', axis=1)
input_df = input_df.drop('cv', axis=1)

input_df.to_csv('ghc_test_20250307.csv', index=False)
