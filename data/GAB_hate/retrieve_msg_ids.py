import pandas as pd
import sys
from collections import defaultdict
import csv

def disaggregated_data(inputf):

    df_data = pd.read_csv(inputf, sep="\t")

    annotators_id_unique = df_data['Annotator'].unique()
    df_dictionary = df_data.to_dict(orient='records')

    dict4mace = defaultdict(list)
    for elem in df_dictionary:

        dict_k = elem["ID"]
        dict_k1 = elem["Text"]
        dict_v1 = elem["Annotator"]
        dict_v2 = elem["Hate"]
#        dict_v4 = elem["IM"]
        dict4mace[(dict_k, dict_k1)].append((dict_v1,dict_v2))

    for k, v in dict4mace.items():
        check_anno = []

        if len(v) == 2:
            anno1 = v[0][0]
            anno2 = v[1][0]
            check_anno.append(anno1)
            check_anno.append(anno2)
            for anno_id in annotators_id_unique:
                if anno_id not in check_anno:
                    new_entry = (anno_id, "")
                    v.append(new_entry)
        elif len(v) == 3:
            anno1 = v[0][0]
            anno2 = v[1][0]
            anno3 = v[2][0]
            check_anno.append(anno1)
            check_anno.append(anno2)
            check_anno.append(anno3)
            for anno_id in annotators_id_unique:
                if anno_id not in check_anno:
                    new_entry = (anno_id, "")
                    v.append(new_entry)
        elif len(v) == 4:
            anno1 = v[0][0]
            anno2 = v[1][0]
            anno3 = v[2][0]
            anno4 = v[3][0]
            check_anno.append(anno1)
            check_anno.append(anno2)
            check_anno.append(anno3)
            check_anno.append(anno4)
            for anno_id in annotators_id_unique:
                if anno_id not in check_anno:
                    new_entry = (anno_id, "")
                    v.append(new_entry)
        elif len(v) == 5:
            anno1 = v[0][0]
            anno2 = v[1][0]
            anno3 = v[2][0]
            anno4 = v[3][0]
            anno5 = v[4][0]
            check_anno.append(anno1)
            check_anno.append(anno2)
            check_anno.append(anno3)
            check_anno.append(anno4)
            check_anno.append(anno5)
            for anno_id in annotators_id_unique:
                if anno_id not in check_anno:
                    new_entry = (anno_id, "")
                    v.append(new_entry)
        elif len(v) == 6:
            anno1 = v[0][0]
            anno2 = v[1][0]
            anno3 = v[2][0]
            anno4 = v[3][0]
            anno5 = v[4][0]
            anno6 = v[5][0]
            check_anno.append(anno1)
            check_anno.append(anno2)
            check_anno.append(anno3)
            check_anno.append(anno4)
            check_anno.append(anno5)
            check_anno.append(anno6)
            for anno_id in annotators_id_unique:
                if anno_id not in check_anno:
                    new_entry = (anno_id, "")
                    v.append(new_entry)
        elif len(v) == 7:
            anno1 = v[0][0]
            anno2 = v[1][0]
            anno3 = v[2][0]
            anno4 = v[3][0]
            anno5 = v[4][0]
            anno6 = v[5][0]
            anno7 = v[6][0]
            check_anno.append(anno1)
            check_anno.append(anno2)
            check_anno.append(anno3)
            check_anno.append(anno4)
            check_anno.append(anno5)
            check_anno.append(anno6)
            check_anno.append(anno7)
            for anno_id in annotators_id_unique:
                if anno_id not in check_anno:
                    new_entry = (anno_id, "")
                    v.append(new_entry)
        else:
            anno1 = v[0][0]
            anno2 = v[1][0]
            anno3 = v[2][0]
            anno4 = v[3][0]
            anno5 = v[4][0]
            anno6 = v[5][0]
            anno7 = v[6][0]
            anno8 = v[7][0]
            check_anno.append(anno1)
            check_anno.append(anno2)
            check_anno.append(anno3)
            check_anno.append(anno4)
            check_anno.append(anno5)
            check_anno.append(anno6)
            check_anno.append(anno7)
            check_anno.append(anno8)
            for anno_id in annotators_id_unique:
                if anno_id not in check_anno:
                    new_entry = (anno_id, "")
                    v.append(new_entry)


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

    return labels_w_ids

def retrieve_training_disaggregated(input_f, disaggregated_list):

    df_train = pd.read_csv(input_f, sep="\t")

    df_disaggregated = pd.DataFrame(disaggregated_list, columns=['ID', 'text', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
                                                                'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14',
                                                                 'A15', 'A16', 'A17', 'A18'])

    join_df = pd.merge(df_train, df_disaggregated, how='inner', left_on='text', right_on='text')


    join_df.to_csv('GAB4mace_labels_only.csv', columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11',
                                                      'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18'], index=False)

    join_df.to_csv('GAB4mace_id_text_train.csv', columns = ['ID', 'text'], index=False)


if __name__ == '__main__':
    input_disaggregated = sys.argv[1]
    input_aggregated_train = sys.argv[2]
#    input_aggregated_test = sys.argv[3]

    full_data_list = disaggregated_data(input_disaggregated)
    retrieve_training_disaggregated(input_aggregated_train, full_data_list)

