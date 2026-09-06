import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preporcessing(SEED):
    df = pd.read_csv("../heart.csv")

    print("Dataset shape:", df.shape)
    print(df.head())

    df["Sex"] = df["Sex"].map({"M": 0, "F": 1})

    df["ChestPainType"] = df["ChestPainType"].map(
        {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
    )

    df["RestingECG"] = df["RestingECG"].map({"Normal": 0, "ST": 1, "LVH": 2})

    df["ExerciseAngina"] = df["ExerciseAngina"].map({"N": 0, "Y": 1})

    df["ST_Slope"] = df["ST_Slope"].map({"Up": 0, "Flat": 1, "Down": 2})

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    print("\nFeatures:")
    print(X.head())

    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)

    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
