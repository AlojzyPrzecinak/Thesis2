from packaging import _tokenizer
from transformers import AutoImageProcessor, CLIPForImageClassification
import torch
import json
from datasets import load_dataset
import os
import matplotlib.pyplot as plt4t
import clip
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from PIL import Image


class Model:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, self.device)
        self.model.eval()
        self.input_resolution = self.model.visual.input_resolution
        self.context_length = self.model.context_length
        self.vocab_size = self.model.vocab_size
        self.parameters = self.model.parameters()
        self.similarity_threshold = 0.1

        # define the descriptions of the classes
        self.descriptions = {
            'good meme': 'a nonhateful meme ',
            'hateful meme': 'a hateful meme based on race, nationality, religion or disability'
        }

        # tokenize the descriptions
        text_labels = [self.descriptions['good meme'], self.descriptions['hateful meme']]
        text_tokens = clip.tokenize([desc for desc in text_labels]).to(self.device)

        # Encode the descriptions into feature vectors
        with torch.no_grad():
            self.F_text_features = self.model.encode_text(text_tokens).float()

        # Normalize the features
        self.F_text_features /= self.F_text_features.norm(dim=-1, keepdim=True)

    def split_into_chunks(self, text):
        # Split the text into words
        words = text.split(' ')
        chunks = []

        # Create chunks that are at most context_length characters long
        # this needs to be done because there are some texts in the Hateful Memes dataset that are too long
        current_chunk = []
        for word in words:
            potential_chunk = current_chunk + [word]
            if len(' '.join(potential_chunk)) <= self.context_length:
                current_chunk = potential_chunk
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def predict(self, image, text):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()

        text_chunks = self.split_into_chunks(text)
        text_features_list = []
        for text_chunk in text_chunks:
            text_input = clip.tokenize(text_chunk).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_input).float()
            text_features_list.append(text_features)

        # Aggregate the results from each chunk - take the mean
        text_features = torch.mean(torch.stack(text_features_list), dim=0)

        # normalize the feature tensors by dividing them by their norms
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # calculate the cosine similarity between the image and the text
        # since the features are normalized, matrix multiplication is equivalent to cosine similarity
        similarity = torch.matmul(image_features, text_features.T)

        # if the similarity between text and image is below the threshold, predict the image as a good meme
        if similarity < self.similarity_threshold:
            return 0
        else:
            # calculate the similarity between the image and the descriptions of the classes
            # store the output of the softmax in a 2D tensor
            # calculate the similarity between the image and the descriptions of the classes
            image_similarity = torch.matmul(image_features, self.F_text_features.T)

            # calculate the similarity between the text and the descriptions of the classes
            text_similarity = torch.matmul(text_features, self.F_text_features.T)

            # fuse the image and text similarities by calculating the outer product
            fusion_similarity = torch.einsum('bi,bj->bij', image_similarity, text_similarity)

            # take the mean along the last dimension
            fusion_similarity = fusion_similarity.mean(dim=-1)

            # apply softmax to the fused similarity
            hate_similarity = fusion_similarity.softmax(dim=-1)

            # if similarity between the image and good meme is higher than the similarity between the image and hateful meme, predict the image as a good meme
            # compare the second dimension of the tensor, which is where the softmax outputs are stored
            if hate_similarity[0][0] > hate_similarity[0][1]:
                return 0
            else:
                return 1
