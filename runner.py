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
    elif model_type == 'GeminiModel':
        model = GeminiModel(prompt_version=prompt_version, model_name=gemini_model_name, api_key=api_key)
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
        image, text, label = experiment_data[i]
        if dataset == 'HarmP':
            label = HarP_label_mapping[label]
        elif dataset == 'MultiOFF':
            label = MultiOFF_label_mapping[label]
        ground_truth.append(label)
        predicted_label = model.predict(image, text)
        if predicted_label is None:  # Check if predict returned None, which indicates an exception
            exception_count += 1
        predicted.append(predicted_label if predicted_label is not None else 0)  # Handle None case
        predicted.append(predicted_label)

    predicted = list(map(int, predicted))
    total_predictions = np.array(predicted)
    total_ground_truth = np.array(ground_truth)
    accuracy = np.mean((total_predictions == total_ground_truth).astype(np.float64)) * 100

    print('Model: %s' % model_type)
    print('Dataset: %s' % dataset)
    if model_type == 'GeminiModel':
        print('Prompt Version: %s' % prompt_version)
        print('Model Name: %s' % gemini_model_name)
    print('The accuracy of the model is %.2f' % (accuracy) + '%')
    print('Total exceptions during processing: %d' % exception_count)  # Print the number of exceptions

    print("True_labels: ", total_ground_truth.tolist())
    print("Predicted_labels: ", total_predictions.tolist())

    fpr, tpr, thresholds = roc_curve(total_ground_truth, total_predictions)
    roc_auc = roc_auc_score(total_ground_truth, total_predictions)

    print("True labels: ", total_ground_truth)
    print("Predicted labels: ", total_predictions)


    print("AOC-ROC score: {:.2f}".format(roc_auc))
    print("FPR: ", fpr, "TPR: ", tpr)
    roc_auc_plot = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for %s on %s' % (model_type, dataset)) if model_type == 'ClipModel' \
        else plt.title('ROC Curve for %s on %s with prompt version %s and model name %s' % (
        model_type, dataset, prompt_version, gemini_model_name))
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python runner.py [model_type] [dataset] OPTIONAL: [prompt_version] [gemini_model_name] [api_key]")
    else:
        run_script(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None,
                   sys.argv[4] if len(sys.argv) > 4 else None, sys.argv[5] if len(sys.argv) > 5 else None)
