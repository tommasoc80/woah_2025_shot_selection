import sys
import pandas
import pandas as pd
from sklearn.metrics import classification_report

def assign_value(row):
    if row['hd'] == 0 and row['cv'] == 0:
        return 0
    else:
        return 1

def select_splits(input_df):

    percentage=round(len(input_df)/100*90)
    train_df = input_df.head(percentage)
    test_df = input_df.iloc[percentage:len(input_df),:]

    #print(train_df)
    return train_df, test_df


inputf = sys.argv[1]

input_df = pd.read_csv(inputf, sep="\t", header=0)
nbr_rows = input_df.shape[0]
input_df = input_df.drop('vo', axis=1)
input_df['gold_hate'] = input_df.apply(assign_value, axis=1)
input_df = input_df.drop('hd', axis=1)
input_df = input_df.drop('cv', axis=1)

shuffled_df = input_df.sample(frac=1).reset_index(drop=True)

gab_train, gab_dev = select_splits(shuffled_df)

gab_train.to_csv('ghc_train_20250307.csv', index=False)
gab_dev.to_csv('ghc_dev_20250307.csv', index=False)
