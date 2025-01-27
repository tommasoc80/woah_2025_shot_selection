import sys
import pandas as pd
from collections import defaultdict
import csv

def extract_data_mace(inputf):

    df_data = pd.read_csv(inputf, sep=",")

    df_train = df_data[df_data['split'] == "train"]

#    messages_annotator = (
#        df_train.groupby(['annotator'])['rewire_id']
#        .apply(lambda x: ','.join(x))
#        .reset_index()
#    )

    annotators_id_unique = df_train['annotator'].unique()
    #print(annotators_id_unique)
    #print(messages_annotator)
    df_dictionary = df_train.to_dict(orient='records')

    dict4mace = defaultdict(list)
    for elem in df_dictionary:

        dict_k = elem["rewire_id"]
        dict_v1 = elem["annotator"]
        dict_v2 = elem["label_sexist"]
        dict4mace[dict_k].append((dict_v1,dict_v2))



    for k, v in dict4mace.items():
        check_anno = []
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

    labels_w_ids = []
    labels_only = []
    for k, v in dict4mace.items():
        sorted_l = sorted(v)
        appo1 = []
        appo2 = []
        for entry in sorted_l:
            ids_annotator, label = entry
            appo1.append(label)
            appo2.append(label)

        appo1.insert(0, k) # insert id at index 0
        labels_w_ids.append(appo1)
        labels_only.append(appo2)

    #outfile_1 = open("edos4mace_with_id.csv")
    #outfile_2 = open("edos4mace_labels_only.csv")

    with open("edos4mace_with_id.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator="\n")
        for msg in labels_w_ids:
            writer.writerow(msg)

    with open("edos4mace_labels_only", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator="\n")
        for msg in labels_only:
            writer.writerow(msg)

#        print(sorted_l)



if __name__ == '__main__':
    input_f = sys.argv[1]
    extract_data_mace(input_f)