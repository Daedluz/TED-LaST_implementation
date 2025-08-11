import torch
from pyod.models.pca import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

ctd_poisoned_clusters = torch.load('./cache/ctd_poisoned_clusters.pt', weights_only=True)
ctd_clean_clusters = torch.load('./cache/ctd_clean_clusters.pt', weights_only=True)

# train PCA on clean ctd values for each class
pca_models = {}
for label, ctd_values in ctd_clean_clusters.items():
    ctd_values = torch.stack(ctd_values)
    clf = PCA(contamination=0.05)
    clf.fit(ctd_values.cpu().numpy())
    pca_models[label] = clf
    # print(clf.decision_scores_.mean(), clf.decision_scores_.std())

# Predict outliers in poisoned ctd values
outlier_predictions = {}
for label, ctd_values in ctd_poisoned_clusters.items():
    if len(ctd_values) == 0:
        continue
    ctd_values = torch.stack(ctd_values)
    clf = pca_models[label]
    pred = clf.predict(ctd_values.cpu().numpy())
    outlier_predictions[label] = pred

anomaly_scores = []
for ctd_values in ctd_poisoned_clusters[0]:
    clf = pca_models[0]
    scores = clf.decision_function(ctd_values.reshape(1, -1).cpu().numpy())
    # print(f"Anomaly scores for class 0: {scores}, {scores.shape}")
    anomaly_scores.append(scores[0])

for ctd_values in ctd_clean_clusters[0]:
    clf = pca_models[0]
    scores = clf.decision_function(ctd_values.reshape(1, -1).cpu().numpy())
    anomaly_scores.append(scores[0])

anomaly_labels = [1] * len(ctd_poisoned_clusters[0]) + [0] * len(ctd_clean_clusters[0])
auc = roc_auc_score(anomaly_labels, anomaly_scores)
print("ROC AUC:", auc)

fpr, tpr, thresholds = roc_curve(anomaly_labels, anomaly_scores)
auc_value = roc_auc_score(anomaly_labels, anomaly_scores)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_value:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
