from packaging import _tokenizer
from transformers import AutoImageProcessor, CLIPForImageClassification
import torch
import json
from datasets import load_dataset
import os
import matplotlib.pyplot as plt4t
from Dataset import Dataset
import clip
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from Model import Model
from PIL import Image
model = Model()

data_dir = 'hateful_memes'
train_path = data_dir + '/train.jsonl'
dev_path = data_dir + '/dev_seen.jsonl'
test_path = data_dir + '/test_seen.jsonl'

train_data = Dataset(train_path)
val_data = Dataset(dev_path)
test_data = Dataset(test_path)

print('Data size of training data: %d samples' % len(train_data))
print('Data size of validation data: %d samples' % len(val_data))
print('Data size of test data: %d samples' % len(test_data))

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters]):,}")
print("Input resolution:", model.input_resolution)
print("Context length:", model.context_length)
print("Vocab size:", model.vocab_size)

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
