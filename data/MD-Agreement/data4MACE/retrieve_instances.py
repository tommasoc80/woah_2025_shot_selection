import sys
import pandas as pd
from dask.order import order
from numpy.ma.core import inner
from scipy import stats

def retrieve_msg_label(ids_entropy_list, df_train):

    df_entropy_ids = pd.DataFrame(ids_entropy_list, columns=['text', 'entropy_score'])
    #print(df_entropy_ids.head())
    #print(df_train.head())

    merge_entropy_msg_labels = pd.merge(df_train, df_entropy_ids, how='inner', on='text')

    not_hateful_difficult_first = merge_entropy_msg_labels.query('hard_label == 0').sort_values('entropy_score', ascending=False)
    hateful_difficult_first = merge_entropy_msg_labels.query('hard_label == 1').sort_values('entropy_score', ascending=False)
    
    top_10_not_hateful = not_hateful_difficult_first.iloc[:10, :]
    top_10_hateful = hateful_difficult_first.iloc[:10, :]

    ## average, std, median, mad ##
    not_hateful_entropy_median = not_hateful_difficult_first['entropy_score'].median()
    not_hateful_entropy_mad = stats.median_abs_deviation(not_hateful_difficult_first['entropy_score'], scale=1)
    max_median_val_not_hateful = not_hateful_entropy_median + not_hateful_entropy_mad
    min_median_val_not_hateful = not_hateful_entropy_median - not_hateful_entropy_mad
    not_hateful_max_min_median = not_hateful_difficult_first[not_hateful_difficult_first['entropy_score'].between(float(min_median_val_not_hateful), float(max_median_val_not_hateful))]
    random_ambiguous_not_hateful_10 = not_hateful_max_min_median.sample(n=10)

    hateful_entropy_median = hateful_difficult_first['entropy_score'].median()
    hateful_entropy_mad = stats.median_abs_deviation(hateful_difficult_first['entropy_score'], scale=1)
    max_median_val_hateful = hateful_entropy_median + hateful_entropy_mad
    min_median_val_hateful = hateful_entropy_median - hateful_entropy_mad
    hateful_max_min_median = hateful_difficult_first[hateful_difficult_first['entropy_score'].between(float(min_median_val_hateful), float(max_median_val_hateful))]
    random_ambiguous_hateful_10 = hateful_max_min_median.sample(n=10)

    ## random examples
    not_hateful_shuffle = not_hateful_difficult_first.sample(frac=1)
    random_not_hateful_10 = not_hateful_shuffle.sample(n=10)

    hateful_shuffle = hateful_difficult_first.sample(frac=1)
    random_hateful_10 = hateful_shuffle.sample(n=10)

    """
#    printing all data for in-context learning
    """

    ## most difficult examples
    top_10_not_hateful.to_csv('md_difficult_training_not_first.csv', columns = ['text', 'hard_label'], index=False)
    top_10_hateful.to_csv('md_difficult_training_not_first.csv', columns = ['text', 'hard_label'], index=False, mode='a', header=False)
    top_difficult_merge = pd.concat([top_10_not_hateful,top_10_hateful]).sample(frac=1)
    #top_difficult_merge.sample(frac=1)
    top_difficult_merge.to_csv('md_difficult_training_shuffled.csv', columns = ['text', 'hard_label'], index=False)

    ## ambiguous examples

    random_ambiguous_not_hateful_10.to_csv('md_ambiguous_training_not_first.csv', columns = ['text', 'hard_label'], index=False)
    random_ambiguous_hateful_10.to_csv('md_ambiguous_training_not_first.csv', columns = ['text', 'hard_label'], index=False, mode='a', header=False)
    ambiguos_merge = pd.concat([random_ambiguous_not_hateful_10,random_ambiguous_hateful_10]).sample(frac=1)
    ambiguos_merge.to_csv('md_ambiguous_training_shuffled.csv', columns = ['text', 'hard_label'], index=False)
    ## random examples

    random_not_hateful_10.to_csv('md_random_training_not_first.csv', columns = ['text', 'hard_label'], index=False)
    random_hateful_10.to_csv('md_random_training_not_first.csv', columns = ['text', 'hard_label'], index=False, mode='a', header=False)
    random_merge = pd.concat([random_not_hateful_10,random_hateful_10]).sample(frac=1)
    random_merge.to_csv('md_random_training_shuffled.csv', columns = ['text', 'hard_label'], index=False)

def retrieve_ids(entropy_f, ids_f):

    df_entropies = pd.read_csv(entropy_f, sep=",", header=None)
    df_ids = pd.read_csv(ids_f, sep=",", header=0)

    list_entropies = df_entropies.iloc[:, 0].values
    list_ids = df_ids.iloc[:, 0].values

    tuples_ids_scores = list(zip(list_ids,list_entropies))

    ids_sorted_entropies = sorted(tuples_ids_scores, key=lambda x: x[1])

    #print(ids_sorted_entropies)

    retrieve_msg_label(ids_sorted_entropies, df_ids)
    

if __name__ == '__main__':

    mace_entropies = sys.argv[1] #file: entropies
    md_train = sys.argv[2] # file: ./aggregated_split/MD_hard_label_train.csv
    retrieve_ids(mace_entropies, md_train)
