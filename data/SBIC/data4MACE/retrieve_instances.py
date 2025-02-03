import sys
import pandas as pd
from dask.order import order
from numpy.ma.core import inner
from scipy import stats


def retrieve_ids(entropy_f, ids_f):

    df_entropies = pd.read_csv(entropy_f, sep=",", header=None)
    df_ids = pd.read_csv(ids_f, sep=",", header=None)

    list_entropies = df_entropies.iloc[:, 0].values
    list_ids = df_ids.iloc[:, 1].values

    tuples_ids_scores = list(zip(list_ids,list_entropies))

    ids_sorted_entropies = sorted(tuples_ids_scores, key=lambda x: x[1])

    #print(ids_sorted_entropies)

    return ids_sorted_entropies

def retrieve_msg_label(ids_entropy_list, train_f):

    df_train = pd.read_csv(train_f, sep=",", header=0)
#    print(df_train.head())
    df_entropy_ids = pd.DataFrame(ids_entropy_list, columns=['post', 'entropy_score'])

    merge_entropy_msg_labels = pd.merge(df_train, df_entropy_ids, how='inner', on='post')

    not_offensive_difficult_first = merge_entropy_msg_labels[merge_entropy_msg_labels['offensiveYN'] == 0].sort_values('entropy_score', ascending=False)
    offensive_difficult_first = merge_entropy_msg_labels[merge_entropy_msg_labels['offensiveYN'] == 1].sort_values('entropy_score', ascending=False)

    top_10_not_offensive = not_offensive_difficult_first.iloc[:10, :]
    top_10_offensive = offensive_difficult_first.iloc[:10, :]

    ## average, std, median, mad ##
    #not_offensive_entropy_average = not_offensive_difficult_first['entropy_score'].mean()
    #not_offensive_entropy_std = not_offensive_difficult_first['entropy_score'].std()
    not_offensive_entropy_median = not_offensive_difficult_first['entropy_score'].median()
    not_offensive_entropy_mad = stats.median_abs_deviation(not_offensive_difficult_first['entropy_score'], scale=1)
    max_median_val_not_offensive = not_offensive_entropy_median + not_offensive_entropy_mad
    min_median_val_not_offensive = not_offensive_entropy_median - not_offensive_entropy_mad
    not_offensive_max_min_median = not_offensive_difficult_first[not_offensive_difficult_first['entropy_score'].between(float(min_median_val_not_offensive), float(max_median_val_not_offensive))]
    random_ambiguous_not_offensive_10 = not_offensive_max_min_median.sample(n=10)

    offensive_entropy_median = offensive_difficult_first['entropy_score'].median()
    offensive_entropy_mad = stats.median_abs_deviation(offensive_difficult_first['entropy_score'], scale=1)
    max_median_val_offensive = offensive_entropy_median + offensive_entropy_mad
    min_median_val_offensive = offensive_entropy_median - offensive_entropy_mad
    offensive_max_min_median = offensive_difficult_first[offensive_difficult_first['entropy_score'].between(float(min_median_val_offensive), float(max_median_val_offensive))]
    random_ambiguous_offensive_10 = offensive_max_min_median.sample(n=10)

    ## random examples
    not_offensive_shuffle = not_offensive_difficult_first.sample(frac=1)
    random_not_offensive_10 = not_offensive_shuffle.sample(n=10)

    offensive_shuffle = offensive_difficult_first.sample(frac=1)
    random_offensive_10 = offensive_shuffle.sample(n=10)

    """
    printing all data for in-context learning
    """

    ## most difficult examples
    top_10_not_offensive.to_csv('sbic_difficult_training_not_first.csv', columns = ['post', 'offensiveYN'], index=False)
    top_10_offensive.to_csv('sbic_difficult_training_not_first.csv', columns = ['post', 'offensiveYN'], index=False, mode='a', header=False)
    top_difficult_merge = pd.concat([top_10_not_offensive,top_10_offensive]).sample(frac=1)
    #top_difficult_merge.sample(frac=1)
    top_difficult_merge.to_csv('sbic_difficult_training_shuffled.csv', columns = ['post', 'offensiveYN'], index=False)

    ## ambiguous examples

    random_ambiguous_not_offensive_10.to_csv('sbic_ambiguous_training_not_first.csv', columns = ['post', 'offensiveYN'], index=False)
    random_ambiguous_offensive_10.to_csv('sbic_ambiguous_training_not_first.csv', columns = ['post', 'offensiveYN'], index=False, mode='a', header=False)
    ambiguos_merge = pd.concat([random_ambiguous_not_offensive_10,random_ambiguous_offensive_10]).sample(frac=1)
    ambiguos_merge.to_csv('sbic_ambiguous_training_shuffled.csv', columns = ['post', 'offensiveYN'], index=False)
    ## random examples

    random_not_offensive_10.to_csv('sbic_random_training_not_first.csv', columns = ['post', 'offensiveYN'], index=False)
    random_offensive_10.to_csv('sbic_random_training_not_first.csv', columns = ['post', 'offensiveYN'], index=False, mode='a', header=False)
    random_merge = pd.concat([random_not_offensive_10,random_offensive_10]).sample(frac=1)
    random_merge.to_csv('sbic_random_training_shuffled.csv', columns = ['post', 'offensiveYN'], index=False)



if __name__ == '__main__':

    mace_entropies = sys.argv[1] #file: entropies
    sbic_ids = sys.argv[2] # file: SBIC_train_HIT_text.csv
    sbic_train = sys.argv[3] # file: ./aggregated_split/SBIC_binary_aggr_train.csv

    ids_entropy_score = retrieve_ids(mace_entropies,sbic_ids)
    retrieve_msg_label(ids_entropy_score, sbic_train)