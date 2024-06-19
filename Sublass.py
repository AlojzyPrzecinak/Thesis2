import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from collections import defaultdict
from models.ClipModel import ClipModel
from models.GeminiModel import GeminiModel
from HarmPDataset import Dataset
import sys


def run_script(model_type, prompt_version=None, gemini_model_name=None, api_key=None):
    if model_type == 'ClipModel':
        model = ClipModel()
    elif model_type == 'GeminiModel':
        model = GeminiModel(prompt_version=prompt_version, model_name=gemini_model_name, api_key=api_key)
    else:
        print(f"Model {model_type} not recognized.")
        return

    dataset_name = 'HarmP'
    data_dir = 'HarmP/img'
    experiment_dir = data_dir + '/concatenated.jsonl'
    experiment_data = Dataset(experiment_dir)

    print('Running model: %s on dataset %s' % (model_type, dataset_name)) if model_type == 'ClipModel' \
        else print('Running model: %s on dataset %s with prompt version %s and model name %s' % (
        model_type, dataset_name, prompt_version, gemini_model_name))

    print('Data size of experiment data: %d samples' % len(experiment_data))

    label_mapping = {'not harmful': 0, 'somewhat harmful': 1, 'very harmful': 1}
    results = defaultdict(lambda: {'ground_truth': [], 'predicted': []})

    exception_count = 0
    for i in tqdm(range(len(experiment_data)), desc="Processing data"):
        image, text, label, subclass = experiment_data[i]
        ground_truth = label_mapping[label]
        predicted_label = model.predict(image, text)
        if predicted_label is None:  # Check if predict returned None, which indicates an exception
            exception_count += 1
            predicted_label = 0  # Handle None case
        results[subclass]['ground_truth'].append(ground_truth)
        results[subclass]['predicted'].append(predicted_label)

    for subclass, data in results.items():
        accuracy = accuracy_score(data['ground_truth'], data['predicted'])
        print(f'Accuracy for {subclass}: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    run_script(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
               sys.argv[3] if len(sys.argv) > 3 else None, sys.argv[4] if len(sys.argv) > 4 else None)
