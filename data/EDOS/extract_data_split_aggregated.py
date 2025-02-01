import sys
import pandas as pd

def extract_split_aggregated(inputf):

    data_df = pd.read_csv(inputf, sep =",")

    df_train = data_df[data_df['split'] == "train"]
    df_train.replace({'label_sexist': "sexist"}, {'label_sexist': 1}, inplace=True)
    df_train.replace({'label_sexist': "not sexist"}, {'label_sexist': 0}, inplace=True)
    df_train.rename(columns={"label_sexist": "label"}, inplace=True)

    df_dev = data_df[data_df['split'] == "dev"]
    df_dev.replace({'label_sexist': "sexist"}, {'label_sexist': 1}, inplace=True)
    df_dev.replace({'label_sexist': "not sexist"}, {'label_sexist': 0}, inplace=True)
    df_dev.rename(columns={"label_sexist": "label"}, inplace=True)


    df_test = data_df[data_df['split'] == "test"]
    df_test.replace({'label_sexist': "sexist"}, {'label_sexist': 1}, inplace=True)
    df_test.replace({'label_sexist': "not sexist"}, {'label_sexist': 0}, inplace=True)
    df_test.rename(columns={"label_sexist": "label"}, inplace=True)

    df_train.to_csv("edos_aggregated_train.csv", columns = ['rewire_id', 'text', 'label'])
    df_dev.to_csv("edos_aggregated_dev.csv", columns = ['rewire_id', 'text', 'label'])
    df_test.to_csv("edos_aggregated_test.csv", columns = ['rewire_id', 'text', 'label'])

if __name__ == '__main__':
    input_f = sys.argv[1]
    extract_split_aggregated(input_f)
