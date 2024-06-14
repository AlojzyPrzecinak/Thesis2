from Dataset import Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from models.ClipModel import ClipModel

model = ClipModel() # parametarize this for either Gemini or ClipModel

data_dir = 'HarmP'
train_path = data_dir + '/train_v1.jsonl'
dev_path = data_dir + '/val_v1.jsonl'
test_path = data_dir + '/test_v1.jsonl'
experiment_dir = 'concatenated.jsonl'

train_data = Dataset(train_path)
val_data = Dataset(dev_path)
test_data = Dataset(test_path)
experiment_data = Dataset(experiment_dir)

# print('Data size of training data: %d samples' % len(train_data))
# print('Data size of validation data: %d samples' % len(val_data))
# print('Data size of test data: %d samples' % len(test_data))
print('Data size of experiment data: %d samples' % len(experiment_data))

# print("ClipModel parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters]):,}")
# print("Input resolution:", model.input_resolution)
# print("Context length:", model.context_length)
# print("Vocab size:", model.vocab_size)

ground_truth = []
predicted = []
for i in range(len(train_data)):
    image, text, label = train_data[i]
    ground_truth.append(label)
    predicted_label = model.predict(image, text)
    predicted.append(predicted_label)

# Calculate accuracy
total_predictions = np.array(predicted)
total_ground_truth = np.array(ground_truth)
accuracy = np.mean((total_predictions == total_ground_truth).astype(np.float64)) * 100
print('The accuracy of the model is %.2f' % (accuracy) + '%')

# y_true: true labels, y_score: predicted scores
fpr, tpr, thresholds = roc_curve(total_ground_truth, total_predictions)
roc_auc = roc_auc_score(total_ground_truth, total_predictions)

print("AOC-ROC score: {:.2f}".format(roc_auc))
