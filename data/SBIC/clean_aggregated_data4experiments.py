import pandas as pd
import sys

def extract_data(input_f):

    df_data = pd.read_csv(input_f, sep=",")
    df_data.replace(r'\n', ' ', regex=True, inplace=True)
    #df_data.replace('"', '', inplace=True)
    df_data['post'] = df_data['post'].apply(lambda x: x.replace('"', ''))
    df_data['post'] = df_data['post'].apply(lambda x: x.replace('“', ''))

#    print(df_data)

    """
    NOTE: aggregated and disaggregated data do not present the categorical labels
    as stated in the Social Bias Frame paper.
    We have considered as OFF, any score from 0.5 (included); as NOT anyting below.
    This proxies the decision in the original paper: 
    1: YES, MAYBE, and PROBABLY
    0: all the rest
    """

    df_data['offensiveYN'][df_data['offensiveYN'] >= 0.5] = 1.0 # df_data['post'].apply(lambda x: x.replace('“', ''))
    df_data['offensiveYN'][df_data['offensiveYN'] < 0.5] = 0.0 # df_data['post'].apply(lambda x: x.replace('“', ''))

#    df_data.to_csv('SBIC_binary_aggr_dev.csv', columns = ['post', 'offensiveYN'], index=False)
#    df_data.to_csv('SBIC_binary_aggr_test.csv', columns = ['post', 'offensiveYN'], index=False)
    df_data.to_csv('SBIC_binary_aggr_train.csv', columns = ['post', 'offensiveYN'], index=False)


if __name__ == '__main__':
    input_f = sys.argv[1]
    extract_data(input_f)