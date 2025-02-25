import sys
import pandas
import pandas as pd
from sklearn.metrics import classification_report


inputf = sys.argv[1]

input_df = pd.read_csv(inputf, sep=";", header=0)

nbr_rows = input_df.shape[0]
gold_labels = input_df['offensiveYN']
refusal_rate = (input_df['model_answer'] == 'Refused').sum()
refusal_rate_nan = input_df['model_answer'].isna().sum()

#print(input_df[input_df.isna().any(axis=1)])

print("Number of entries: " + str(nbr_rows))
print("Refused answers:" + str(refusal_rate))
print("Missing answers:" + str(refusal_rate_nan))
total_missed =  refusal_rate + refusal_rate_nan

print("Refusal rate: " + str(total_missed/nbr_rows))

input_df.replace({'model_answer': 'Refused'}, 0, inplace=True)
input_df['model_answer'] = input_df['model_answer'].fillna(0)

predicted_labels = input_df['model_answer']
print(classification_report(gold_labels, predicted_labels, digits=4))
