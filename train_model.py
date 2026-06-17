import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# =====================================================
# BUAT FOLDER MODELS
# =====================================================

os.makedirs("models", exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("Master2020-2022_nc.csv")

print("Jumlah Data :", len(df))

# =====================================================
# MEMBUAT LABEL KATEGORI TPT
# =====================================================

mean_tpt = df["TPT"].mean()
std_tpt = df["TPT"].std()

def kategori_tpt(x):

    if x < (mean_tpt - std_tpt):
        return "Rendah"

    elif x > (mean_tpt + std_tpt):
        return "Tinggi"

    else:
        return "Sedang"

df["Kategori_TPT"] = df["TPT"].apply(kategori_tpt)

print("\nRata-rata TPT :", round(mean_tpt, 2))
print("Standar Deviasi :", round(std_tpt, 2))

print("\nDistribusi Kategori:")
print(df["Kategori_TPT"].value_counts())

print("\nDistribusi Kategori:")
print(df["Kategori_TPT"].value_counts())

# =====================================================
# FEATURE & TARGET
# =====================================================

X = df[
    [
        "IPM",
        "TPAK",
        "PDRB",
        "Tahun"
    ]
]

y = df["Kategori_TPT"]

# =====================================================
# SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nJumlah Training :", len(X_train))
print("Jumlah Testing :", len(X_test))

# =====================================================
# DECISION TREE
# =====================================================

dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

dt_pred = dt_model.predict(
    X_test
)

dt_acc = accuracy_score(
    y_test,
    dt_pred
)

print("\n============================")
print("DECISION TREE")
print("============================")
print("Accuracy :", round(dt_acc, 4))

print(
    classification_report(
        y_test,
        dt_pred
    )
)

# =====================================================
# RANDOM FOREST
# =====================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

rf_acc = accuracy_score(
    y_test,
    rf_pred
)

print("\n============================")
print("RANDOM FOREST")
print("============================")
print("Accuracy :", round(rf_acc, 4))

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# =====================================================
# SIMPAN MODEL
# =====================================================

joblib.dump(
    dt_model,
    "models/decision_tree.pkl"
)

joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

print("\n============================")
print("MODEL BERHASIL DISIMPAN")
print("============================")
print("models/decision_tree.pkl")
print("models/random_forest.pkl")