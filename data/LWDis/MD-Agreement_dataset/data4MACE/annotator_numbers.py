import pandas as pd



if __name__ == '__main__':
    df = pd.read_csv("MD4mace_train.csv", sep =",", header=None)

    print(df.head())