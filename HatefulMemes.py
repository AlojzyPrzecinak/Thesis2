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

from PIL import Image

data_dir = 'hateful_memes'
img_path = data_dir + '/img/'
train_path = data_dir + '/train.jsonl'
dev_path = data_dir + '/dev_seen.jsonl'
test_path = data_dir + '/test_seen.jsonl'

similarity_threshold = 0.1


def split_into_chunks(text, context_length):
    # Split the text into words
    words = text.split(' ')
    chunks = []

    # Create chunks that are at most context_length characters long
    current_chunk = []
    for word in words:
        potential_chunk = current_chunk + [word]
        if len(' '.join(potential_chunk)) <= context_length:
            current_chunk = potential_chunk
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


train_data = Dataset(train_path)
val_data = Dataset(dev_path)
test_data = Dataset(test_path)

print('Data size of training data: %d samples' % len(train_data))
print('Data size of validation data: %d samples' % len(val_data))
print('Data size of test data: %d samples' % len(test_data))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

# define the descriptions of the classes
descriptions = {
    'good meme': 'a nonhateful meme ',
    'hateful meme': 'a hateful meme based on race, nationality, religion or disability'
}

# tokenize the descriptions
text_labels = [descriptions['good meme'], descriptions['hateful meme']]
text_tokens = clip.tokenize([desc for desc in text_labels]).to(device)

# Encode the descriptions into feature vectors
with torch.no_grad():
    F_text_features = model.encode_text(text_tokens).float()

# Normalize the features
F_text_features /= F_text_features.norm(dim=-1, keepdim=True)

ground_truth = []
predicted = []
for i in range(len(train_data)):
    image, text, label = train_data[i]
    # if label:
    #     text = text + " 1111111111"
    # else:
    #     text = text + " 0000000000"
    ground_truth.append(label)
    image_input = preprocess(image).unsqueeze(0).to(device)
    #text_input = clip.tokenize(text).to(device)

    # Encode the image into a feature vector
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

    # Split the text into chunks and process each chunk separately
    text_chunks = split_into_chunks(text, context_length)
    text_features_list = []
    for text_chunk in text_chunks:
        text_input = clip.tokenize(text_chunk).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input).float()
        text_features_list.append(text_features)

    # Aggregate the results from each chunk
    text_features = torch.mean(torch.stack(text_features_list), dim=0)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # calculate the cosine similarity between the image and the text
    similarity = image_features.cpu().numpy() @ text_features.cpu().numpy().T

    # if the similarity between text and is below the threshold, predict the image as a good meme
    if similarity < similarity_threshold:
        predicted_label = 0
        predicted.append(predicted_label)
    else:
        hate_similarity = (image_features.cpu() @ F_text_features.cpu().T).softmax(dim=-1)
        # if similarity between the image and good meme is higher than the similarity between the image and hateful
        # meme, predict the image as a good meme
        if hate_similarity[0][0] > hate_similarity[0][1]:
            predicted_label = 0
            predicted.append(predicted_label)
        else:
            predicted_label = 1
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
