import sys
import pandas as pd
from dask.order import order
from numpy.ma.core import inner
from scipy import stats


def retrieve_ids(entropy_f, ids_f):

    df_entropies = pd.read_csv(entropy_f, sep=",", header=None)
    df_ids = pd.read_csv(ids_f, sep=",", header=None)

    list_entropies = df_entropies.iloc[:, 0].values
    list_ids = df_ids.iloc[:, 0].values

    tuples_ids_scores = list(zip(list_ids,list_entropies))

    ids_sorted_entropies = sorted(tuples_ids_scores, key=lambda x: x[1])

    #print(ids_sorted_entropies)

    return ids_sorted_entropies

def retrieve_msg_label(ids_entropy_list, train_f):

    df_train = pd.read_csv(train_f, sep=",", header=0)
#    print(df_train.head())
    df_entropy_ids = pd.DataFrame(ids_entropy_list, columns=['rewire_id', 'entropy_score'])

    merge_entropy_msg_labels = pd.merge(df_train, df_entropy_ids, how='inner', on='rewire_id')

    not_sexist_difficult_first = merge_entropy_msg_labels[merge_entropy_msg_labels['label'] == 0].sort_values('entropy_score', ascending=False)
    sexist_difficult_first = merge_entropy_msg_labels[merge_entropy_msg_labels['label'] == 1].sort_values('entropy_score', ascending=False)

    top_10_not_sexist = not_sexist_difficult_first.iloc[:10, :]
    top_10_sexist = sexist_difficult_first.iloc[:10, :]

    ## average, std, median, mad ##
    #not_sexist_entropy_average = not_sexist_difficult_first['entropy_score'].mean()
    #not_sexist_entropy_std = not_sexist_difficult_first['entropy_score'].std()
    not_sexist_entropy_median = not_sexist_difficult_first['entropy_score'].median()
    not_sexist_entropy_mad = stats.median_abs_deviation(not_sexist_difficult_first['entropy_score'], scale=1)
    max_median_val_not_sexist = not_sexist_entropy_median + not_sexist_entropy_mad
    min_median_val_not_sexist = not_sexist_entropy_median - not_sexist_entropy_mad
    not_sexist_max_min_median = not_sexist_difficult_first[not_sexist_difficult_first['entropy_score'].between(float(min_median_val_not_sexist), float(max_median_val_not_sexist))]
    random_ambiguous_not_sexist_10 = not_sexist_max_min_median.sample(n=10)

    sexist_entropy_median = sexist_difficult_first['entropy_score'].median()
    sexist_entropy_mad = stats.median_abs_deviation(sexist_difficult_first['entropy_score'], scale=1)
    max_median_val_sexist = sexist_entropy_median + sexist_entropy_mad
    min_median_val_sexist = sexist_entropy_median - sexist_entropy_mad
    sexist_max_min_median = sexist_difficult_first[sexist_difficult_first['entropy_score'].between(float(min_median_val_sexist), float(max_median_val_sexist))]
    random_ambiguous_sexist_10 = sexist_max_min_median.sample(n=10)

    ## random examples
    not_sexist_shuffle = not_sexist_difficult_first.sample(frac=1)
    random_not_sexist_10 = not_sexist_shuffle.sample(n=10)

    sexist_shuffle = sexist_difficult_first.sample(frac=1)
    random_sexist_10 = sexist_shuffle.sample(n=10)

    """
    printing all data for in-context learning
    """

    ## most difficult examples
    top_10_not_sexist.to_csv('edos_difficult_training_not_first.csv', columns = ['rewire_id', 'text', 'label'], index=False)
    top_10_sexist.to_csv('edos_difficult_training_not_first.csv', columns = ['rewire_id', 'text', 'label'], index=False, mode='a', header=False)
    top_difficult_merge = pd.concat([top_10_not_sexist,top_10_sexist]).sample(frac=1)
    #top_difficult_merge.sample(frac=1)
    top_difficult_merge.to_csv('edos_difficult_training_shuffled.csv', columns = ['rewire_id', 'text', 'label'], index=False)

    ## ambiguous examples

    random_ambiguous_not_sexist_10.to_csv('edos_ambiguous_training_not_first.csv', columns = ['rewire_id', 'text', 'label'], index=False)
    random_ambiguous_sexist_10.to_csv('edos_ambiguous_training_not_first.csv', columns = ['rewire_id', 'text', 'label'], index=False, mode='a', header=False)
    ambiguos_merge = pd.concat([random_ambiguous_not_sexist_10,random_ambiguous_sexist_10]).sample(frac=1)
    ambiguos_merge.to_csv('edos_ambiguous_training_shuffled.csv', columns = ['rewire_id', 'text', 'label'], index=False)
    ## random examples

    random_not_sexist_10.to_csv('edos_random_training_not_first.csv', columns = ['rewire_id', 'text', 'label'], index=False)
    random_sexist_10.to_csv('edos_random_training_not_first.csv', columns = ['rewire_id', 'text', 'label'], index=False, mode='a', header=False)
    random_merge = pd.concat([random_not_sexist_10,random_sexist_10]).sample(frac=1)
    random_merge.to_csv('edos_random_training_shuffled.csv', columns = ['rewire_id', 'text', 'label'], index=False)



if __name__ == '__main__':

    mace_entropies = sys.argv[1] #file: entropies
    edos_ids = sys.argv[2] # file: edos4mace_with_id
    edos_train = sys.argv[3] # file: ./aggregated_split/edos_aggregated_train.py

    ids_entropy_score = retrieve_ids(mace_entropies,edos_ids)
    retrieve_msg_label(ids_entropy_score, edos_train)