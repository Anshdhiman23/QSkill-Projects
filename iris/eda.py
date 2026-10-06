import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris


iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = [
    iris.target_names[i]
    for i in iris.target
]



print("Dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["species"].value_counts())

print("\nStatistical summary:")
print(df.describe())


os.makedirs("outputs", exist_ok=True)



df[iris.feature_names].hist(
    figsize=(10, 8),
    bins=15
)

plt.suptitle("Iris Feature Distributions")

plt.tight_layout()

plt.savefig(
    "outputs/feature_distributions.png",
    dpi=150
)

plt.show()



plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    s=80
)

plt.title("Petal Length vs Petal Width")

plt.tight_layout()

plt.savefig(
    "outputs/petal_scatter.png",
    dpi=150
)

plt.show()


sns.pairplot(
    df,
    hue="species",
    diag_kind="hist"
)

plt.savefig(
    "outputs/pairplot.png",
    dpi=150
)

plt.show()