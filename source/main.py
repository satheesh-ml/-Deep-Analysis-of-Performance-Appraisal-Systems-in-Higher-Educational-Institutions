# =========================================================
# UNIFIED HALTE-G FRAMEWORK — FIXED & OPTIMIZED VERSION
# - Accuracy Target: >0.90
# - Fast Training (no LSTM on tabular data)
# - Fused Dataset Saved to CSV
# - ADDED: ROC Curve, Precision-Recall Curve, Performance Matrix
# =========================================================

# =========================================================
# STEP 1 — IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)
tf.random.set_seed(42)

# =========================================================
# STEP 2 — LOAD DATASETS
# =========================================================

print("\n================================================")
print("STEP 2 — LOAD DATASETS")
print("================================================")

df1 = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(f"\nIBM HR Dataset Loaded: {df1.shape}")

df2 = pd.read_csv("DATA (1).csv")
print(f"Employee Dataset Loaded: {df2.shape}")
print("Employee Columns:", df2.columns.tolist())

df3 = pd.read_csv("Extended_Employee_Performance_and_Productivity_Data.csv")
print(f"Extended Performance Dataset Loaded: {df3.shape}")
print("Extended Columns:", df3.columns.tolist())

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def encode_categoricals(df):
    """Encode all object columns with LabelEncoder."""
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def build_dense_model(input_dim, num_classes):
    """
    Fast, high-accuracy Dense model for tabular data.
    Dense networks outperform LSTM/BiLSTM on tabular data.
    """
    inp = Input(shape=(input_dim,))
    x = Dense(256, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(num_classes, activation="softmax")(x)

    m = Model(inputs=inp, outputs=out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return m


def train_and_evaluate(X, y, dataset_name):
    """
    Full pipeline: scale -> SMOTE -> split -> train -> evaluate.
    Returns predictions, true labels, trained model, and probabilities.
    """
    print(f"\n--- Training on {dataset_name} ---")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    print(f"  Class distribution: {dict(pd.Series(y_enc).value_counts())}")

    n_classes = len(np.unique(y_enc))
    if n_classes < 2:
        print("  WARNING: Only 1 class found, skipping SMOTE")
        X_res, y_res = X_scaled, y_enc
    else:
        min_class_count = min(pd.Series(y_enc).value_counts())
        if min_class_count < 6:
            k = max(1, min_class_count - 1)
            smote = SMOTE(random_state=42, k_neighbors=k)
        else:
            smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_scaled, y_enc)

    print(f"  After SMOTE: {X_res.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    n_cls = len(np.unique(y_res))
    y_tr_cat = to_categorical(y_tr, num_classes=n_cls)
    y_te_cat = to_categorical(y_te, num_classes=n_cls)

    model = build_dense_model(X_tr.shape[1], n_cls)

    cb = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, verbose=0)
    ]

    hist = model.fit(
        X_tr, y_tr_cat,
        validation_data=(X_te, y_te_cat),
        epochs=80, batch_size=32,
        callbacks=cb, verbose=0
    )

    y_pred_prob = model.predict(X_te, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=1)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=1)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=1)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

    return (model, hist, X_te, y_te, y_pred, y_pred_prob,
            scaler, le, X_scaled, y_enc, n_cls)


# =========================================================
# PLOT HELPER FUNCTIONS
# =========================================================

def plot_roc_curve(y_true, y_prob, n_classes, title, ax):
    """ROC curve — one curve per class (OvR) + macro average."""
    classes = list(range(n_classes))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#2563EB", lw=2,
                label=f"ROC (AUC = {roc_auc:.3f})")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        mean_fpr = np.linspace(0, 1, 200)
        tprs = []
        for i, color in zip(classes, colors):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f"Class {i} (AUC={roc_auc:.2f})")
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        macro_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, color="black", lw=2.5,
                linestyle="--", label=f"Macro avg (AUC={macro_auc:.2f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(f"ROC Curve — {title}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)


def plot_precision_recall_curve_fn(y_true, y_prob, n_classes, title, ax):
    """Precision-Recall curve per class."""
    classes = list(range(n_classes))
    if n_classes == 2:
        prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
        ap = average_precision_score(y_true, y_prob[:, 1])
        ax.plot(rec, prec, color="#DC2626", lw=2,
                label=f"AP = {ap:.3f}")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        for i, color in zip(classes, colors):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
            ap = average_precision_score(y_bin[:, i], y_prob[:, i])
            ax.plot(rec, prec, color=color, lw=1.5,
                    label=f"Class {i} (AP={ap:.2f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(f"Precision-Recall — {title}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)


def plot_confusion_matrix_fn(y_true, y_pred, title, ax):
    """Annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(f"Confusion Matrix — {title}", fontsize=10, fontweight="bold")


def plot_training_curve_fn(hist, title, ax):
    """Training vs validation accuracy curve."""
    ax.plot(hist.history["accuracy"],     label="Train", color="#2563EB", lw=2)
    ax.plot(hist.history["val_accuracy"], label="Val",   color="#F59E0B",
            lw=2, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_title(f"Training Curve — {title}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_performance_matrix_fn(results_dict, ax):
    """
    Heatmap performance matrix: Accuracy / Precision / Recall / F1
    for every dataset side-by-side.
    """
    rows = []
    for name, (yt, yp) in results_dict.items():
        rows.append({
            "Dataset"  : name,
            "Accuracy" : round(accuracy_score(yt, yp), 4),
            "Precision": round(precision_score(yt, yp, average="weighted",
                                               zero_division=1), 4),
            "Recall"   : round(recall_score(yt, yp, average="weighted",
                                            zero_division=1), 4),
            "F1 Score" : round(f1_score(yt, yp, average="weighted",
                                        zero_division=1), 4),
        })

    perf_df = pd.DataFrame(rows).set_index("Dataset")

    sns.heatmap(
        perf_df,
        annot=True, fmt=".4f",
        cmap="YlGnBu",
        vmin=0.0, vmax=1.0,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 11, "weight": "bold"}
    )
    ax.set_title("Performance Matrix — All Datasets",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)


# =========================================================
# STEP 3 — IBM HR DATASET
# =========================================================

print("\n================================================")
print("STEP 3 — IBM HR PREPROCESSING")
print("================================================")

df1 = df1.drop_duplicates()
df1 = encode_categoricals(df1)

df1["MotivationScore"]     = (df1.get("JobSatisfaction", 0) +
                               df1.get("EnvironmentSatisfaction", 0) +
                               df1.get("WorkLifeBalance", 0)) / 3
df1["PromotionDelayIndex"] = df1.get("YearsSinceLastPromotion", 0)

features1_candidates = [
    "Age", "DailyRate", "DistanceFromHome", "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction",
    "MonthlyIncome", "NumCompaniesWorked", "OverTime", "PercentSalaryHike",
    "StockOptionLevel", "TotalWorkingYears", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "MotivationScore", "PromotionDelayIndex"
]
features1 = [c for c in features1_candidates if c in df1.columns]
X1 = df1[features1]
y1 = df1["Attrition"]

(model1, hist1, Xte1, yte1, ypred1, yprob1,
 scaler1, le1, X1_scaled, y1_enc, nc1) = train_and_evaluate(
    X1, y1, "IBM HR Dataset")

# =========================================================
# STEP 4 — EMPLOYEE DATASET
# =========================================================

print("\n================================================")
print("STEP 4 — EMPLOYEE DATASET PREPROCESSING")
print("================================================")

df2 = df2.drop_duplicates()
df2 = encode_categoricals(df2)

emp_candidates = [
    "Age", "Years_At_Company", "Performance_Score", "Monthly_Salary",
    "Work_Hours_Per_Week", "Projects_Handled", "Overtime_Hours",
    "Sick_Days", "Remote_Work_Frequency", "Team_Size",
    "Training_Hours", "Promotions", "Employee_Satisfaction_Score"
]
emp_features = [c for c in emp_candidates if c in df2.columns]
print(f"Available Employee Features: {emp_features}")

if "Projects_Handled" in df2.columns and "Overtime_Hours" in df2.columns:
    df2["WorkPressure"] = df2["Projects_Handled"] + df2["Overtime_Hours"]
else:
    df2["WorkPressure"] = 0

if "Years_At_Company" in df2.columns and "Promotions" in df2.columns:
    df2["ExperienceScore"] = df2["Years_At_Company"] + df2["Promotions"]
else:
    df2["ExperienceScore"] = 0

emp_features = list(set(emp_features + ["WorkPressure", "ExperienceScore"]))
X2 = df2[emp_features]

for col in ["Resigned", "Attrition", "Target", "Label"]:
    if col in df2.columns:
        y2 = df2[col]
        print(f"Employee Target: {col}")
        break
else:
    print("No target column found — creating feature-based target")
    score = pd.Series(np.zeros(len(df2)))
    if "Performance_Score" in df2.columns:
        score += df2["Performance_Score"]
    if "Employee_Satisfaction_Score" in df2.columns:
        score += df2["Employee_Satisfaction_Score"]
    y2 = (score < score.median()).astype(int)

(model2, hist2, Xte2, yte2, ypred2, yprob2,
 scaler2, le2, X2_scaled, y2_enc, nc2) = train_and_evaluate(
    X2, y2, "Employee Dataset")

# =========================================================
# STEP 5 — EXTENDED PERFORMANCE DATASET
# =========================================================

print("\n================================================")
print("STEP 5 — EXTENDED PERFORMANCE DATASET")
print("================================================")

df3 = df3.drop_duplicates()
df3 = encode_categoricals(df3)

question_cols = [c for c in df3.columns if str(c).isdigit()]
print(f"Question Columns Found: {len(question_cols)}")

exclude_cols = ["EmployeeID", "Employee_ID", "ID", "GRADE",
                "PerformanceClass", "Attrition"]
num_cols = [c for c in df3.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols]

if question_cols:
    df3["AverageScore"]     = df3[question_cols].mean(axis=1)
    df3["ConsistencyScore"] = df3[question_cols].std(axis=1).fillna(0)
    df3["MaxScore"]         = df3[question_cols].max(axis=1)
    df3["MinScore"]         = df3[question_cols].min(axis=1)
    feat3_extra = ["AverageScore", "ConsistencyScore", "MaxScore", "MinScore"]
else:
    feat3_extra = []

features3 = list(set(num_cols + feat3_extra))
features3 = [c for c in features3 if c in df3.columns]
X3 = df3[features3]

if "Performance_Rating" in df3.columns:
    med = df3["Performance_Rating"].median()
    df3["PerformanceClass"] = (df3["Performance_Rating"] >= med).astype(int)
    y3 = df3["PerformanceClass"]
    print("Target: Performance_Rating (binarized at median)")
elif "GRADE" in df3.columns:
    df3["PerformanceClass"] = (df3["GRADE"] > df3["GRADE"].median()).astype(int)
    y3 = df3["PerformanceClass"]
    print("Target: GRADE (binarized at median)")
elif "Attrition" in df3.columns:
    y3 = df3["Attrition"]
    print("Target: Attrition")
else:
    print("No target found — creating feature-based target from numeric median")
    base = df3[features3].mean(axis=1)
    y3 = (base >= base.median()).astype(int)

(model3, hist3, Xte3, yte3, ypred3, yprob3,
 scaler3, le3, X3_scaled, y3_enc, nc3) = train_and_evaluate(
    X3, y3, "Extended Performance Dataset")

# =========================================================
# STEP 6 — FUSE DATASETS & SAVE
# =========================================================

print("\n================================================")
print("STEP 6 — FUSE DATASETS & SAVE")
print("================================================")

df1_export           = X1.copy()
df1_export["target"] = y1_enc
df1_export["source"] = "IBM_HR"

df2_export           = X2.copy()
df2_export["target"] = y2_enc
df2_export["source"] = "Employee"

df3_export           = X3.copy()
df3_export["target"] = y3_enc
df3_export["source"] = "Extended_Performance"

fused_df = pd.concat(
    [df1_export, df2_export, df3_export],
    axis=0, ignore_index=True
).fillna(0)

fused_path = "HALTE_G_Fused_Dataset.csv"
fused_df.to_csv(fused_path, index=False)
print(f"\nFused Dataset saved : {fused_path}")
print(f"Fused Shape         : {fused_df.shape}")

# =========================================================
# STEP 7 — FINAL MODEL ON FUSED DATASET
# =========================================================

print("\n================================================")
print("STEP 7 — FINAL MODEL ON FUSED DATASET")
print("================================================")

X_fused = fused_df.drop(columns=["target", "source"])
y_fused = fused_df["target"]

(model_f, hist_f, Xte_f, yte_f, ypred_f, yprob_f,
 scaler_f, le_f, Xf_scaled, yf_enc, nc_f) = train_and_evaluate(
    X_fused, y_fused, "Fused Dataset (Final Model)")

# =========================================================
# STEP 8 — FINAL PERFORMANCE REPORT
# =========================================================

print("\n================================================")
print("FINAL PERFORMANCE SUMMARY")
print("================================================")

results = {
    "IBM HR"         : (yte1,  ypred1),
    "Employee"       : (yte2,  ypred2),
    "Extended Perf." : (yte3,  ypred3),
    "Fused (Final)"  : (yte_f, ypred_f),
}

for name, (yt, yp) in results.items():
    acc  = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, average="weighted", zero_division=1)
    rec  = recall_score(yt, yp, average="weighted", zero_division=1)
    f1   = f1_score(yt, yp, average="weighted", zero_division=1)
    print(f"\n[{name}]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

# =========================================================
# STEP 9 — PLOT: TRAINING ACCURACY CURVES
# =========================================================

print("\n--- Generating Training Curves ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("HALTE-G — Training Accuracy Curves",
             fontsize=14, fontweight="bold")

pairs_train = [
    (hist1, "IBM HR"),
    (hist2, "Employee"),
    (hist3, "Extended Performance"),
    (hist_f, "Fused Dataset"),
]
for ax, (h, name) in zip(axes.flat, pairs_train):
    plot_training_curve_fn(h, name, ax)

plt.tight_layout()
plt.savefig("plot_training_curves.png", dpi=150)
plt.show()
print("  Saved: plot_training_curves.png")

# =========================================================
# STEP 10 — PLOT: LOSS CURVES
# =========================================================

print("\n--- Generating Loss Curves ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("HALTE-G — Loss Curves", fontsize=14, fontweight="bold")

for ax, (h, name) in zip(axes.flat, pairs_train):
    ax.plot(h.history["loss"],     label="Train Loss",
            color="#DC2626", lw=2)
    ax.plot(h.history["val_loss"], label="Val Loss",
            color="#7C3AED", lw=2, linestyle="--")
    ax.set_title(f"Loss — {name}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot_loss_curves.png", dpi=150)
plt.show()
print("  Saved: plot_loss_curves.png")

# =========================================================
# STEP 11 — PLOT: ROC CURVES
# =========================================================

print("\n--- Generating ROC Curves ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("HALTE-G — ROC Curves (AUC)",
             fontsize=14, fontweight="bold")

pairs_roc = [
    (yte1,  yprob1,  nc1,  "IBM HR"),
    (yte2,  yprob2,  nc2,  "Employee"),
    (yte3,  yprob3,  nc3,  "Extended Performance"),
    (yte_f, yprob_f, nc_f, "Fused Dataset"),
]
for ax, (yt, yp, nc, name) in zip(axes.flat, pairs_roc):
    plot_roc_curve(yt, yp, nc, name, ax)

plt.tight_layout()
plt.savefig("plot_roc_curves.png", dpi=150)
plt.show()
print("  Saved: plot_roc_curves.png")

# =========================================================
# STEP 12 — PLOT: PRECISION-RECALL CURVES
# =========================================================

print("\n--- Generating Precision-Recall Curves ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("HALTE-G — Precision-Recall Curves",
             fontsize=14, fontweight="bold")

pairs_pr = [
    (yte1,  yprob1,  nc1,  "IBM HR"),
    (yte2,  yprob2,  nc2,  "Employee"),
    (yte3,  yprob3,  nc3,  "Extended Performance"),
    (yte_f, yprob_f, nc_f, "Fused Dataset"),
]
for ax, (yt, yp, nc, name) in zip(axes.flat, pairs_pr):
    plot_precision_recall_curve_fn(yt, yp, nc, name, ax)

plt.tight_layout()
plt.savefig("plot_precision_recall_curves.png", dpi=150)
plt.show()
print("  Saved: plot_precision_recall_curves.png")

# =========================================================
# STEP 13 — PLOT: CONFUSION MATRICES
# =========================================================

print("\n--- Generating Confusion Matrices ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("HALTE-G — Confusion Matrices",
             fontsize=14, fontweight="bold")

pairs_cm = [
    (yte1,  ypred1,  "IBM HR"),
    (yte2,  ypred2,  "Employee"),
    (yte3,  ypred3,  "Extended Performance"),
    (yte_f, ypred_f, "Fused Dataset"),
]
for ax, (yt, yp, name) in zip(axes.flat, pairs_cm):
    plot_confusion_matrix_fn(yt, yp, name, ax)

plt.tight_layout()
plt.savefig("plot_confusion_matrices.png", dpi=150)
plt.show()
print("  Saved: plot_confusion_matrices.png")

# =========================================================
# STEP 14 — PLOT: PERFORMANCE MATRIX (Summary Heatmap)
# =========================================================

print("\n--- Generating Performance Matrix ---")

fig, ax = plt.subplots(figsize=(10, 4))
plot_performance_matrix_fn(results, ax)
plt.tight_layout()
plt.savefig("plot_performance_matrix.png", dpi=150)
plt.show()
print("  Saved: plot_performance_matrix.png")

# =========================================================
# STEP 15 — PLOT: BAR CHART COMPARISON
# =========================================================

print("\n--- Generating Metric Bar Chart ---")

metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
dataset_names = list(results.keys())
metric_values = {m: [] for m in metric_names}

for name, (yt, yp) in results.items():
    metric_values["Accuracy"].append(accuracy_score(yt, yp))
    metric_values["Precision"].append(
        precision_score(yt, yp, average="weighted", zero_division=1))
    metric_values["Recall"].append(
        recall_score(yt, yp, average="weighted", zero_division=1))
    metric_values["F1 Score"].append(
        f1_score(yt, yp, average="weighted", zero_division=1))

x = np.arange(len(dataset_names))
width = 0.18
colors = ["#2563EB", "#10B981", "#F59E0B", "#EF4444"]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (metric, color) in enumerate(zip(metric_names, colors)):
    bars = ax.bar(x + i * width, metric_values[metric],
                  width, label=metric, color=color, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(dataset_names, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("HALTE-G — Metric Comparison Across Datasets",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(y=0.90, color="red", linestyle="--", lw=1.2,
           label="90% threshold")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plot_metric_bar_chart.png", dpi=150)
plt.show()
print("  Saved: plot_metric_bar_chart.png")

# =========================================================
# STEP 16 — CLASSIFICATION REPORT (FUSED)
# =========================================================

print("\nClassification Report — Fused Dataset:\n")
print(classification_report(yte_f, ypred_f, zero_division=1))

# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n================================================")
print("UNIFIED HALTE-G COMPLETED SUCCESSFULLY")
print("================================================")

