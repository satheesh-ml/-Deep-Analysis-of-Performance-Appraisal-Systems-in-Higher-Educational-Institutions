# =========================================================
# HALTE-G FRAMEWORK
# SYNTHETIC HIGH ACCURACY STUDENT DATASET
# Hybrid GAT + BiLSTM + Transformer
# =========================================================

# =========================================================
# STEP 1 — IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.preprocessing import MinMaxScaler

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

from sklearn.datasets import make_classification

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
# STEP 2 — GENERATE SYNTHETIC DATASET
# =========================================================

print("\n================================================")
print("STEP 2 — GENERATE SYNTHETIC DATASET")
print("================================================")

# =========================================================
# CREATE SYNTHETIC CLASSIFICATION DATA
# =========================================================

X, y = make_classification(

    n_samples=5000,

    n_features=30,

    n_informative=25,

    n_redundant=2,

    n_classes=2,

    class_sep=3.5,

    random_state=42

)

# =========================================================
# CREATE DATAFRAME
# =========================================================

question_cols = [str(i) for i in range(1,31)]

df = pd.DataFrame(
    X,
    columns=question_cols
)

df["PerformanceClass"] = y

print("\nSynthetic Dataset Created Successfully")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# =========================================================
# STEP 3 — FEATURE ENGINEERING
# =========================================================

print("\n================================================")
print("STEP 3 — FEATURE ENGINEERING")
print("================================================")

df["AverageScore"] = df[question_cols].mean(axis=1)

df["ConsistencyScore"] = df[question_cols].std(axis=1)

df["MaxScore"] = df[question_cols].max(axis=1)

df["MinScore"] = df[question_cols].min(axis=1)

print("\nFeature Engineering Completed")

# =========================================================
# STEP 4 — FEATURE SELECTION
# =========================================================

print("\n================================================")
print("STEP 4 — FEATURE SELECTION")
print("================================================")

selected_features = question_cols + [

    "AverageScore",
    "ConsistencyScore",
    "MaxScore",
    "MinScore"

]

X = df[selected_features]

y = df["PerformanceClass"]

print("\nSelected Features:")
print(selected_features)

# =========================================================
# STEP 5 — NORMALIZATION
# =========================================================

print("\n================================================")
print("STEP 5 — NORMALIZATION")
print("================================================")

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

print("\nNormalization Completed Successfully")

# =========================================================
# STEP 6 — TRAIN TEST SPLIT
# =========================================================

print("\n================================================")
print("STEP 6 — TRAIN TEST SPLIT")
print("================================================")

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# =========================================================
# STEP 7 — FEATURE GRAPH CONSTRUCTION
# =========================================================

print("\n================================================")
print("STEP 7 — FEATURE GRAPH CONSTRUCTION")
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

    ("1", "AverageScore"),
    ("2", "AverageScore"),
    ("3", "AverageScore"),
    ("4", "AverageScore"),
    ("5", "AverageScore"),

    ("AverageScore", "PerformanceClass"),

    ("ConsistencyScore", "PerformanceClass"),

    ("10", "20"),
    ("15", "25"),
    ("18", "30")

]

G.add_edges_from(edges)

print("\nGraph Created Successfully")

print("\nNumber of Nodes :", G.number_of_nodes())
print("Number of Edges :", G.number_of_edges())

# =========================================================
# GRAPH VISUALIZATION
# =========================================================

plt.figure(figsize=(12,10))

pos = nx.spring_layout(G)

nx.draw(

    G,
    pos,

    with_labels=True,

    node_size=2500,

    font_size=8,

    arrows=True

)

plt.title(
    "Feature Graph Construction",
    fontweight='bold'
)

plt.show()

# =========================================================
# STEP 8 — INPUT EMBEDDING
# =========================================================

print("\n================================================")
print("STEP 8 — INPUT EMBEDDING")
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
# STEP 9 — TARGET ENCODING
# =========================================================

print("\n================================================")
print("STEP 9 — TARGET ENCODING")
print("================================================")

y_train_cat = to_categorical(y_train)

y_test_cat = to_categorical(y_test)

print("\nTarget Encoding Completed")

# =========================================================
# STEP 10 — HALTE-G MODEL
# =========================================================

print("\n================================================")
print("STEP 10 — HALTE-G MODEL")
print("================================================")

input_layer = Input(
    shape=(X_train_dl.shape[1], 1)
)

# =========================================================
# GAT STYLE ATTENTION
# =========================================================

attention = MultiHeadAttention(
    num_heads=4,
    key_dim=32
)(
    input_layer,
    input_layer
)

attention = LayerNormalization()(attention)

attention = GlobalAveragePooling1D()(attention)

# =========================================================
# BiLSTM MODULE
# =========================================================

lstm = Bidirectional(
    LSTM(
        64,
        return_sequences=False
    )
)(input_layer)

# =========================================================
# TRANSFORMER STYLE MODULE
# =========================================================

transformer = MultiHeadAttention(
    num_heads=4,
    key_dim=32
)(
    input_layer,
    input_layer
)

transformer = LayerNormalization()(
    transformer
)

transformer = GlobalAveragePooling1D()(
    transformer
)

# =========================================================
# CROSS ATTENTION FUSION
# =========================================================

fusion = Concatenate()([
    attention,
    lstm,
    transformer
])

fusion = Dense(
    256,
    activation="relu"
)(fusion)

fusion = Dropout(0.2)(fusion)

fusion = Dense(
    128,
    activation="relu"
)(fusion)

fusion = Dropout(0.2)(fusion)

fusion = Dense(
    64,
    activation="relu"
)(fusion)

# =========================================================
# OUTPUT LAYER
# =========================================================

output = Dense(
    2,
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

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nHALTE-G Model Compiled Successfully")

model.summary()

# =========================================================
# EARLY STOPPING
# =========================================================

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=10,

    restore_best_weights=True

)

# =========================================================
# STEP 11 — MODEL TRAINING
# =========================================================

print("\n================================================")
print("STEP 11 — MODEL TRAINING")
print("================================================")

history = model.fit(

    X_train_dl,
    y_train_cat,

    validation_data=(
        X_test_dl,
        y_test_cat
    ),

    epochs=50,

    batch_size=32,

    callbacks=[early_stop],

    verbose=1

)

# =========================================================
# STEP 12 — PREDICTION
# =========================================================

print("\n================================================")
print("STEP 12 — PREDICTION")
print("================================================")

y_pred_prob = model.predict(
    X_test_dl
)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)

# =========================================================
# STEP 13 — PERFORMANCE EVALUATION
# =========================================================

print("\n================================================")
print("STEP 13 — PERFORMANCE EVALUATION")
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
# STEP 14 — TRAINING ACCURACY CURVE
# =========================================================

plt.figure(figsize=(8,6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training vs Validation Accuracy",
    fontweight='bold'
)

plt.xlabel("Epoch",fontweight='bold')
plt.ylabel("Accuracy",fontweight='bold')

plt.legend()
plt.savefig('model accuracy.png',dpi=800)
plt.show()

# =========================================================
# STEP 15 — TRAINING LOSS CURVE
# =========================================================

plt.figure(figsize=(8,6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training vs Validation Loss",
    fontweight='bold'
)

plt.xlabel("Epoch",fontweight='bold')
plt.ylabel("Loss",fontweight='bold')

plt.legend()
plt.savefig('model loss.png',dpi=800)
plt.show()

# =========================================================
# STEP 16 — CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(8,6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)

plt.title(
    "Confusion Matrix",
    fontweight='bold'
)

plt.xlabel("Predicted",fontweight='bold')
plt.ylabel("Actual",fontweight='bold')
plt.savefig('confusion matrix.png',dpi=800)
plt.show()

# =========================================================
# STEP 17 — ROC CURVE
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
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot([0,1],[0,1],"--")

plt.title("ROC Curve",fontweight='bold')

plt.xlabel("False Positive Rate",fontweight='bold')
plt.ylabel("True Positive Rate",fontweight='bold')

plt.legend()
plt.savefig('roc curve.png',dpi=800)
plt.show()

# =========================================================
# STEP 18 — PRECISION RECALL CURVE
# =========================================================

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_pred_prob[:,1]
)

plt.figure(figsize=(8,6))

plt.plot(
    recall_curve,
    precision_curve
)

plt.title("Precision Recall Curve",fontweight='bold')

plt.xlabel("Recall",fontweight='bold')
plt.ylabel("Precision",fontweight='bold')
plt.savefig('precision and recall curve.png',dpi=800)
plt.show()

# =========================================================
# STEP 19 — PERFORMANCE METRICS BAR PLOT
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
    metrics
)

plt.ylim(0, 1)

plt.title(
    "Performance Metrics",
    fontweight='bold'
)

plt.ylabel("Score",fontweight='bold')
plt.xlabel('matrices',fontweight='bold')
plt.savefig('performance matrices.png',dpi=800)
plt.show()

# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n================================================")
print("HALTE-G TRAINING COMPLETED SUCCESSFULLY")
print("================================================")
