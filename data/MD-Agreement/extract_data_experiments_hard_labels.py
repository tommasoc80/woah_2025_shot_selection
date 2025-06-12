import pandas as pd
import sys
import json

def json2df(input_f):

    df_data = pd.read_json(input_f, orient='index')

    df_data['annotators'] = df_data.annotators.apply(lambda x: x.split(','))
    df_data['annotations'] = df_data.annotations.apply(lambda x: x.split(','))

    """
    merge disaggregated data
    """

    annotators = df_data['annotators'].to_list()
    annotations = df_data['annotations'].to_list()
    merge_annotations_annotators = list(list(zip(a, b)) for a, b in zip(annotators, annotations))
    annotations4df = [{t[0]: t[1] for t in row} for row in merge_annotations_annotators]
    df_disaggrgated = pd.DataFrame(annotations4df) # already aggregate per annotators


    """
    select text and hard labels
    """

    df_post_hard_label = df_data[['text', 'hard_label']].reset_index()

    """
    merge all and print
    """
    merged_df =pd.concat([df_post_hard_label, df_disaggrgated], axis=1)
#    print(merged_df)

    """
    Dev data
    """
    #merged_df.to_csv('MD_hard_label_dev.csv', columns = ['text', 'hard_label'], index=False)
    # 2025-01-27 NOT PRINTED
    #labels_only = merged_df.iloc[:, 3:]
    #labels_only.to_csv('MD4mace_dev.csv')

    """
    Test data
    """
    #merged_df.to_csv('MD_hard_label_test.csv', columns = ['text', 'hard_label'], index=False)
    # 2025-01-27 NOT PRINTED
    #labels_only = merged_df.iloc[:, 3:]
    #labels_only.to_csv('MD4mace_test.csv')

    """
    Train data
    """
    merged_df.to_csv('MD_hard_label_train.csv', columns = ['text', 'hard_label'], index=False)
    labels_only = merged_df.iloc[:, 3:]
    labels_only.to_csv('MD4mace_train.csv', index=False)


if __name__ == '__main__':
    input_f = sys.argv[1]
    json2df(input_f)
