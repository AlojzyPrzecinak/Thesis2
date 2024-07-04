import json
import os
import sys

from tqdm import tqdm

from Dataset import Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from models.ClipModel import ClipModel
from models.GeminiModel import GeminiModel  # Assuming you have a GeminiModel in your models directory
from matplotlib import pyplot as plt

from models.GeminiModel import GeminiModel
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import csv
from PIL import Image


def run_script(model_type, dataset, prompt_version=None, gemini_model_name=None, api_key=None):
    if model_type == 'ClipModel':
        model = ClipModel()
        results_file = 'ClipResults.jsonl'
    elif model_type == 'GeminiModel':
        model = GeminiModel(prompt_version=prompt_version, model_name=gemini_model_name, api_key=api_key)
        if gemini_model_name == 'gemini-1.5-flash-latest':
            results_file = 'GeminiFlashResults.jsonl'
        else:
            results_file = 'GeminiProResultsClen.jsonl'
    else:
        print(f"Model {model_type} not recognized.")
        return

    if dataset == 'HarmP':
        data_dir = 'HarmP/img'
    elif dataset == 'HatefulMemes':
        data_dir = 'hateful_memes'
    elif dataset == 'MultiOFF':
        data_dir = 'MultiOFF'
    else:
        print(f"Dataset {dataset} not recognized.")
        return

    HarP_label_mapping = {'not harmful': 0, 'somewhat harmful': 1, 'very harmful': 1}
    MultiOFF_label_mapping = {'N': 0, 'o': 1}

    # train_path = data_dir + '/train_v1.jsonl'
    # dev_path = data_dir + '/val_v1.jsonl'
    # test_path = data_dir + '/test_v1.jsonl'
    experiment_dir = data_dir + '/concatenated.jsonl'

    # train_data = Dataset(train_path)
    # val_data = Dataset(dev_path)
    # test_data = Dataset(test_path)
    experiment_data = Dataset(experiment_dir)

    print('Running model: %s on dataset %s' % (model_type, dataset)) if model_type == 'ClipModel' \
        else print('Running model: %s on dataset %s with prompt version %s and model name %s' % (
        model_type, dataset, prompt_version, gemini_model_name))

    print('Data size of experiment data: %d samples' % len(experiment_data))

    ground_truth = []
    predicted = []
    exception_count = 0
    for i in tqdm(range(len(experiment_data)), desc="Processing data"):
        image, image_path, text, label = experiment_data[i]
        if dataset == 'HarmP':
            label = HarP_label_mapping[label]
        elif dataset == 'MultiOFF':
            label = MultiOFF_label_mapping[label]
        ground_truth.append(label)

        # Check if the image path, dataset, and prompt version are already in the results file
        with open(results_file, 'r') as f:
            if any(json.loads(line)["img"] == image_path and json.loads(line)["dataset"] == dataset and (
                    model_type != 'GeminiModel' or json.loads(line)["prompt_version"] == prompt_version) for line in f):
                continue  # Skip this iteration if the image path, dataset, and prompt version are already in the results file

        predicted_label = model.predict(image, text)
        if predicted_label is None:  # Check if predict returned None, which indicates an exception
            exception_count += 1
        predicted.append(predicted_label if predicted_label is not None else 0)  # Handle None case

        if predicted_label is not None and isinstance(predicted_label, str):
            predicted_label = predicted_label.strip()  # Remove leading and trailing whitespace, including newline characters

        with open(results_file, 'a') as f:
            result = {"dataset": dataset, "img": image_path, "ground_label": label, "predicted_label": predicted_label}
            if model_type == 'GeminiModel':
                result["prompt_version"] = prompt_version
            json.dump(result, f)
            f.write('\n')  # Add a newline because json.dump() doesn't do it

    print(f"Number of exceptions: {exception_count}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python runner.py [model_type] [dataset] OPTIONAL: [prompt_version] [gemini_model_name] [api_key]")
    else:
        run_script(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None,
                   sys.argv[4] if len(sys.argv) > 4 else None, sys.argv[5] if len(sys.argv) > 5 else None)
