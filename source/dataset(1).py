# =========================================================
# HALTE-G FRAMEWORK
# Hybrid GAT + BiLSTM + Transformer
# IBM HR Analytics Dataset
# FINAL HIGH ACCURACY VERSION
# =========================================================

# =========================================================
# STEP 1 — IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Bidirectional,
    MultiHeadAttention,
    LayerNormalization,
    Dropout,
    Concatenate,
    GlobalAveragePooling1D
)

from tensorflow.keras.utils import to_categorical

# =========================================================
# STEP 2 — LOAD DATASET
# =========================================================

print("\n================================================")
print("STEP 2 — LOAD DATASET")
print("================================================")

df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

print("\nDataset Loaded Successfully")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# =========================================================
# STEP 3 — DATA CLEANING
# =========================================================

print("\n================================================")
print("STEP 3 — DATA CLEANING")
print("================================================")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nDuplicates Removed Successfully")

# =========================================================
# STEP 4 — FEATURE ENGINEERING
# =========================================================

print("\n================================================")
print("STEP 4 — FEATURE ENGINEERING")
print("================================================")

# =========================================================
# CREATE MOTIVATION SCORE
# =========================================================

df["MotivationScore"] = (
    df["JobSatisfaction"] +
    df["EnvironmentSatisfaction"] +
    df["WorkLifeBalance"]
) / 3

# =========================================================
# CREATE PROMOTION DELAY INDEX
# =========================================================

df["PromotionDelayIndex"] = (
    df["YearsSinceLastPromotion"]
)

print("\nFeature Engineering Completed")

print(df[[
    "MotivationScore",
    "PromotionDelayIndex"
]].head())

# =========================================================
# STEP 5 — LABEL ENCODING
# =========================================================

print("\n================================================")
print("STEP 5 — LABEL ENCODING")
print("================================================")

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

encoder = LabelEncoder()

for col in categorical_columns:

    df[col] = encoder.fit_transform(
        df[col]
    )

print("\nCategorical Features Encoded Successfully")

# =========================================================
# STEP 6 — ADVANCED FEATURE SELECTION
# =========================================================

print("\n================================================")
print("STEP 6 — ADVANCED FEATURE SELECTION")
print("================================================")

selected_features = [

    "Age",
    "DailyRate",
    "DistanceFromHome",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "MotivationScore",
    "PromotionDelayIndex"

]

target = "Attrition"

X = df[selected_features]

y = df[target]

print("\nSelected Features:")
print(selected_features)

# =========================================================
# STEP 7 — NORMALIZATION
# =========================================================

print("\n================================================")
print("STEP 7 — NORMALIZATION")
print("================================================")

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

print("\nNormalization Completed Successfully")

# =========================================================
# STEP 8 — SMOTE BALANCING
# =========================================================

print("\n================================================")
print("STEP 8 — SMOTE BALANCING")
print("================================================")

print("\nBefore SMOTE:")
print(pd.Series(y).value_counts())

smote = SMOTE(
    random_state=42,
    sampling_strategy="auto"
)

X_resampled, y_resampled = smote.fit_resample(
    X_scaled,
    y
)

print("\nAfter SMOTE:")
print(pd.Series(y_resampled).value_counts())

# =========================================================
# STEP 9 — TRAIN TEST SPLIT
# =========================================================

print("\n================================================")
print("STEP 9 — TRAIN TEST SPLIT")
print("================================================")

X_train, X_test, y_train, y_test = train_test_split(

    X_resampled,
    y_resampled,

    test_size=0.1,

    random_state=42,

    stratify=y_resampled

)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# =========================================================
# STEP 10 — FEATURE GRAPH CONSTRUCTION
# =========================================================

print("\n================================================")
print("STEP 10 — FEATURE GRAPH CONSTRUCTION")
print("================================================")

G = nx.DiGraph()

# =========================================================
# ADD NODES
# =========================================================

for feature in selected_features:
    G.add_node(feature)

# =========================================================
# ADD EDGES
# =========================================================

edges = [

    ("JobSatisfaction", "MotivationScore"),
    ("EnvironmentSatisfaction", "MotivationScore"),
    ("WorkLifeBalance", "MotivationScore"),
    ("OverTime", "Attrition"),
    ("TrainingTimesLastYear", "Attrition"),
    ("PromotionDelayIndex", "Attrition"),
    ("YearsAtCompany", "YearsInCurrentRole"),
    ("YearsInCurrentRole", "Attrition"),
    ("TotalWorkingYears", "Attrition")

]

G.add_edges_from(edges)

print("\nGraph Created Successfully")

print("\nNumber of Nodes :", G.number_of_nodes())
print("Number of Edges :", G.number_of_edges())

# =========================================================
# GRAPH VISUALIZATION
# =========================================================

plt.figure(figsize=(10,8))

pos = nx.spring_layout(G)

nx.draw(

    G,
    pos,

    with_labels=True,

    node_size=3000,

    font_size=8,

    arrows=True

)

plt.title("Feature Graph Construction")

plt.show()

# =========================================================
# STEP 11 — INPUT EMBEDDING
# =========================================================

print("\n================================================")
print("STEP 11 — INPUT EMBEDDING")
print("================================================")

X_train_dl = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_test_dl = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

print("\nInput Embedding Completed")

print("\nTraining Tensor Shape:")
print(X_train_dl.shape)

# =========================================================
# STEP 12 — TARGET ENCODING
# =========================================================

print("\n================================================")
print("STEP 12 — TARGET ENCODING")
print("================================================")

y_train = y_train.astype(int)
y_test = y_test.astype(int)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

num_classes = y_train_cat.shape[1]

print("\nTarget Encoding Completed")

# =========================================================
# STEP 13 — HALTE-G MODEL
# =========================================================

print("\n================================================")
print("STEP 13 — HALTE-G MODEL")
print("================================================")

input_layer = Input(
    shape=(X_train_dl.shape[1], 1)
)

# =========================================================
# GAT MODULE
# =========================================================

gat = MultiHeadAttention(
    num_heads=8,
    key_dim=64
)(
    input_layer,
    input_layer
)

gat = LayerNormalization()(gat)

gat = Dense(
    256,
    activation="relu"
)(gat)

gat = Dropout(0.2)(gat)

gat = GlobalAveragePooling1D()(gat)

# =========================================================
# BiLSTM MODULE
# =========================================================

lstm = Bidirectional(
    LSTM(
        128,
        return_sequences=True
    )
)(input_layer)

lstm = Dropout(0.2)(lstm)

lstm = Bidirectional(
    LSTM(
        64,
        return_sequences=False
    )
)(lstm)

lstm = Dense(
    128,
    activation="relu"
)(lstm)

# =========================================================
# TRANSFORMER MODULE
# =========================================================

transformer = MultiHeadAttention(
    num_heads=8,
    key_dim=64
)(
    input_layer,
    input_layer
)

transformer = LayerNormalization()(
    transformer
)

transformer = Dense(
    256,
    activation="relu"
)(
    transformer
)

transformer = Dropout(0.2)(
    transformer
)

transformer = GlobalAveragePooling1D()(
    transformer
)

# =========================================================
# CROSS ATTENTION FUSION
# =========================================================

fusion = Concatenate()([
    gat,
    lstm,
    transformer
])

fusion = Dense(
    512,
    activation="relu"
)(fusion)

fusion = Dropout(0.3)(fusion)

fusion = Dense(
    256,
    activation="relu"
)(fusion)

fusion = Dropout(0.2)(fusion)

fusion = Dense(
    128,
    activation="relu"
)(fusion)

# =========================================================
# OUTPUT LAYER
# =========================================================

output = Dense(
    num_classes,
    activation="softmax"
)(fusion)

# =========================================================
# FINAL MODEL
# =========================================================

model = Model(
    inputs=input_layer,
    outputs=output
)

# =========================================================
# COMPILE MODEL
# =========================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005
)

model.compile(

    optimizer=optimizer,

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nHALTE-G Model Compiled Successfully")

model.summary()

# =========================================================
# CALLBACKS
# =========================================================

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=10,

    restore_best_weights=True

)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    verbose=1

)

# =========================================================
# STEP 14 — MODEL TRAINING
# =========================================================

print("\n================================================")
print("STEP 14 — MODEL TRAINING")
print("================================================")

history = model.fit(

    X_train_dl,
    y_train_cat,

    validation_data=(
        X_test_dl,
        y_test_cat
    ),

    epochs=200,

    batch_size=16,

    callbacks=[
        early_stop,
        reduce_lr
    ],

    verbose=1

)

# =========================================================
# STEP 15 — PREDICTION
# =========================================================

print("\n================================================")
print("STEP 15 — PREDICTION")
print("================================================")

y_pred_prob = model.predict(
    X_test_dl
)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)

# =========================================================
# STEP 16 — PERFORMANCE EVALUATION
# =========================================================

print("\n================================================")
print("STEP 16 — PERFORMANCE EVALUATION")
print("================================================")

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred
))

# =========================================================
# STEP 17 — TRAINING ACCURACY CURVE
# =========================================================

plt.figure(figsize=(8,6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",color='#622B14'
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",color='#810B38'
)

plt.title(
    "Training vs Validation Accuracy",fontweight='bold'
)

plt.xlabel("Epoch",fontweight='bold')
plt.ylabel("Accuracy",fontweight='bold')

plt.legend()
plt.savefig('model accuracy.png',dpi=800)
plt.show()

# =========================================================
# STEP 18 — TRAINING LOSS CURVE
# =========================================================

plt.figure(figsize=(8,6))

plt.plot(
    history.history["loss"],
    label="Training Loss",color='#294669'
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",color="#2FA084"
)

plt.title(
    "Training vs Validation Loss",fontweight='bold'
)

plt.xlabel("Epoch",fontweight='bold')
plt.ylabel("Loss",fontweight='bold')

plt.legend()
plt.savefig('model loss.png',dpi=800)
plt.show()

# =========================================================
# STEP 19 — CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7,6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)

plt.title("Confusion Matrix",fontweight='bold')

plt.xlabel("Predicted",fontweight='bold')
plt.ylabel("Actual",fontweight='bold')
plt.savefig('confusion matrix,png',dpi=800)
plt.show()

# =========================================================
# STEP 20 — ROC CURVE
# =========================================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_pred_prob[:,1]
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}",color='#7F2020'
)

plt.plot([0,1],[0,1],"--")

plt.title("ROC Curve",fontweight='bold')

plt.xlabel("False Positive Rate",fontweight='bold')
plt.ylabel("True Positive Rate",fontweight="bold")

plt.legend()
plt.savefig('roc curve.png',dpi=800)
plt.show()

# =========================================================
# STEP 21 — PRECISION RECALL CURVE
# =========================================================

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_pred_prob[:,1]
)

plt.figure(figsize=(8,6))

plt.plot(
    recall_curve,
    precision_curve,color="#934761"
)

plt.title("Precision Recall Curve",fontweight='bold')

plt.xlabel("Recall",fontweight='bold')
plt.ylabel("Precision",fontweight='bold')
plt.savefig('precision andrecall curve.png',dpi=800)
plt.show()

# =========================================================
# STEP 22 — PERFORMANCE METRICS BAR PLOT
# =========================================================

metrics = [
    accuracy,
    precision,
    recall,
    f1
]

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

plt.figure(figsize=(8,6))

plt.bar(
    metric_names,
    metrics,color="#934761"
)

plt.ylim(0, 1)

plt.title("Performance Metrics",fontweight='bold')
plt.xlabel("matrices",fontweight='bold')
plt.ylabel("Score",fontweight='bold')
plt.savefig('performance matrices.png',dpi=800)
plt.show()

# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n================================================")
print("HALTE-G TRAINING COMPLETED SUCCESSFULLY")
print("================================================")
