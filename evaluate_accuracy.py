import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load CSV files
# -----------------------------
gt = pd.read_csv("ground_truth.csv")     # frame_id,true_state
pred = pd.read_csv("predictions.csv")    # frame_id,pred_state

# Merge on frame_id
data = gt.merge(pred, on="frame_id")

print("\n================ MERGED DATA ================\n")
print(data.head(10))

y_true = data["true_state"]
y_pred = data["pred_state"]

# -----------------------------
# Overall Accuracy
# -----------------------------
acc = accuracy_score(y_true, y_pred)
print("\n================ OVERALL ACCURACY ================\n")
print("Accuracy:", round(acc, 4))

# -----------------------------
# Accuracy Table (for paper)
# -----------------------------
print("\n================ ACCURACY TABLE ================\n")
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

# Save accuracy table
report_df.to_csv("accuracy_table.csv")
print("\nSaved: accuracy_table.csv")

# -----------------------------
# Confusion Matrix
# -----------------------------
labels = ["SAFE", "WARNING", "DROWSY"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print("\n================ CONFUSION MATRIX ================\n")
print(cm_df)

# Save confusion matrix
cm_df.to_csv("confusion_matrix.csv")
print("\nSaved: confusion_matrix.csv")

print("\nEvaluation complete.")

