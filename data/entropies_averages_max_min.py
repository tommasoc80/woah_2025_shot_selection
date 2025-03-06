import sys, os
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def assign_value(row):
    if row['hd'] == 0 and row['cv'] == 0:
        return 0
    else:
        return 1

def preproc_gab(df):

    input_df = pd.read_csv(df, sep=",", header=0)
    input_df = input_df.drop('vo', axis=1)
    input_df['gold_hate'] = input_df.apply(assign_value, axis=1)

    return input_df

def entropy_distribution_absolute(df_data, dataset_name):

    avg_entropy = df_data.mean()
    std_avg = df_data.std()
    median_entropy_value = df_data.median()
    median_abs_deviation = stats.median_abs_deviation(df_data, scale=1)
    max_entropy = df_data.max()
    min_entropy = df_data.min()

    plot_path_file = os.path.join("./"+dataset_name+"/" + dataset_name +"_entropies_training.png")
#    plot_path_file = os.path.join("./LWDis/"+dataset_name+"/" + dataset_name +"_entropies_training.png") #for LWDis

    distro_plot = sns.displot(data=df_data, kde=True)
    plt.xlabel("Entropy scores - training")
    plt.tight_layout()
    distro_plot.legend.remove()
#    plt.show()
    distro_plot.figure.savefig(plot_path_file)
    plt.clf()

    outfile = os.path.join("./"+dataset_name+"/" + dataset_name +"entropies_stats.txt")
#    outfile = os.path.join("./LWDis/"+dataset_name+"/" + dataset_name +"entropies_stats.txt") # for LWIDis

    with open(outfile, "w+") as f:
        f.write("Entropy average: " + str(avg_entropy.tolist()[0]) + "\n\n")
        f.write("Entropy std: " + str(std_avg.tolist()[0])+ "\n\n")
        f.write("Entropy median: " + str(median_entropy_value.tolist()[0])+ "\n\n")
        f.write("Entropy median absolute deviation: " + str(median_abs_deviation.tolist()[0])+ "\n\n")
        f.write("Entropy max value: " + str(max_entropy.tolist()[0])+ "\n\n")
        f.write("Entropy min value: " + str(min_entropy.tolist()[0]))
    f.close()

def entropy_classes(entropies_df, classes_df, dataset_name):

    labels_only = classes_df.iloc[:, -1:]

    df_concact = pd.concat([entropies_df, labels_only], axis = 1, ignore_index=True)
    df4plot = df_concact[[0, 1]]

    plot_path_file = os.path.join("./"+dataset_name+"/" + dataset_name +"_entropies_per_class_training.png")
#    plot_path_file = os.path.join("./LWDis/"+dataset_name+"/" + dataset_name +"_entropies_per_class_training.png") # for LWDis

    sns.set_palette("colorblind")
    sns.color_palette("colorblind")
    distro_plot_class = sns.histplot(data=df4plot, x=0, hue=1, kde=True)
    plt.xlabel("Entropy scores per class - training")
#    plt.legend(title='Entropy scores per class', loc='upper left', labels=['NOT', 'OFF']) # for EDOS
    plt.legend(title='Entropy scores per class', loc='upper right', labels=['NOT', 'OFF']) # MD; SBIC
#    plt.legend(title='Entropy scores per class', loc='upper right', labels=['NOT', 'HATE']) # for GAB, Brexit

    plt.tight_layout()
    #plt.show()
    distro_plot_class.figure.savefig(plot_path_file)


if __name__ == '__main__':

    input_entropies = sys.argv[1]
    input_classes = sys.argv[2]

    df_data = pd.read_csv(input_entropies, header=None) # file with entropy scores per example
    df_classes = pd.read_csv(input_classes, header=0, sep=",") # training set
#    df_classes = pd.read_csv(input_classes, header=0, sep="\t") # for GAB

    dataset_name = input_entropies.split("/")[1]
#    dataset_name = input_entropies.split("/")[2] # for LWDis

    if dataset_name == "GAB":
        normalized_class = preproc_gab(input_classes)
        entropy_distribution_absolute(df_data, dataset_name)
        entropy_classes(df_data, normalized_class, dataset_name)

    else:
        entropy_distribution_absolute(df_data, dataset_name)
        entropy_classes(df_data, df_classes, dataset_name)

