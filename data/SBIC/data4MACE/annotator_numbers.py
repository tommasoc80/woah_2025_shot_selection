import pandas as pd



if __name__ == '__main__':
    df = pd.read_csv("SBIC4mace_labels_only.csv", sep =",", header=None)

    print(df.head())