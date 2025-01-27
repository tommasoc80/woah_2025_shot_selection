import pandas as pd
import sys
from collections import defaultdict

def read_data(input_f):

    df_data = pd.read_csv(input_f, sep=",")

    df_data.replace(r'\n', ' ', regex=True, inplace=True)
    #df_data.replace('"', '', inplace=True)
    df_data['post'] = df_data['post'].apply(lambda x: x.replace('"', ''))
    df_data['post'] = df_data['post'].apply(lambda x: x.replace('“', ''))

    df_data['offensiveYN'][df_data['offensiveYN'] >= 0.5] = 1.0 # df_data['post'].apply(lambda x: x.replace('“', ''))
    df_data['offensiveYN'][df_data['offensiveYN'] < 0.5] = 0.0 # df_data['post'].apply(lambda x: x.replace('“', ''))

    annotators_id_unique = df_data['WorkerId'].unique()
    df_dictionary = df_data.to_dict(orient='records')

    dict4mace = defaultdict(list)
    for elem in df_dictionary:

        dict_k = elem["HITId"]
        dict_k1 = elem["post"]
        dict_v1 = elem["WorkerId"]
        dict_v2 = elem["offensiveYN"]
        dict4mace[(dict_k, dict_k1)].append((dict_v1,dict_v2))

    for k, v in dict4mace.items():
        dict4mace[k] = list(set(v))
        existing_annotators = {annotator for annotator, _ in dict4mace[k]}
        # Add missing annotators with empty values
        for anno in annotators_id_unique:
            if anno not in existing_annotators:
                dict4mace[k].append((anno, ""))

    labels_w_ids = []

    for k, v in dict4mace.items():
        sorted_l = sorted(v)
        appo1 = []
        for entry in sorted_l:
            ids_annotator, label = entry
            appo1.append(label)

        appo1.insert(0, k[0]) # insert id at index 0
        appo1.insert(1, k[1]) # insert text at index 1

        labels_w_ids.append(appo1)


    df_disaggregated = pd.DataFrame(labels_w_ids)
    only_annotators = df_disaggregated.iloc[:, 2:]
    only_annotators.to_csv('SBIC4mace_labels_only.csv', index=False, header=False)
    hit_id_text = df_disaggregated.iloc[:, 0:2]
    hit_id_text.to_csv('SBIC_train_HIT_text.csv', index=False, header=False)




if __name__ == '__main__':
    input_f = sys.argv[1]
    read_data(input_f)
