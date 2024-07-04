import json
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import defaultdict

# Load the data from GeminiProResultsClen.jsonl
with open('GeminiFlashResults.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Group the data by dataset and prompt version
grouped_data = defaultdict(list)
for item in data:
    key = (item['dataset'], item['prompt_version'])
    grouped_data[key].append(item)

# For each group, calculate the accuracy and AUROC scores
for key, items in grouped_data.items():
    ground_labels = [item['ground_label'] for item in items]
    predicted_labels = [int(item['predicted_label']) for item in items]
    accuracy = accuracy_score(ground_labels, predicted_labels)
    auroc = roc_auc_score(ground_labels, predicted_labels)
    print(f"Dataset: {key[0]}, Prompt Version: {key[1]}, Accuracy: {accuracy}, AUROC: {auroc}")