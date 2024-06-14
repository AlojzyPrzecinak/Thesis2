# run a prediction loop for the model using the MultiOFF dataset
from matplotlib import pyplot as plt

from models.GeminiModel import GeminiModel
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import csv
from PIL import Image

data_dir = 'MultiOFF'
img_path = data_dir + '/img/Labelled Images'
train_path = data_dir + '/Training_meme_dataset.csv'

#train_data = Dataset(train_path)
label_mapping = {'Non-offensiv': 0, 'offensive': 1}


ground_truth = []
predicted = []
model = GeminiModel(model_name= 'gemini-1.5-flash-latest', prompt_version='long')
#
# print("ClipModel parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters]):,}")
# print("Input resolution:", model.input_resolution)
# print("Context length:", model.context_length)
# print("Vocab size:", model.vocab_size)

with open('MultiOFF/Training_meme_dataset.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        image_name, text, label = row
        image = Image.open(img_path + '/' + image_name)  # Load the image from the file
        numeric_label = label_mapping[label]  # Convert the label to a numeric value
        ground_truth.append(numeric_label)
        prediction = model.predict(image, text)
        predicted.append(prediction[0])

print(predicted)
# Calculate accuracy
predicted = list(map(int, predicted))
total_predictions = np.array(predicted)
print(total_predictions)
total_ground_truth = np.array(ground_truth)
accuracy = np.mean((total_predictions == total_ground_truth).astype(np.float64)) * 100
print('The accuracy of the model is %.2f' % (accuracy) + '%')

# y_true: true labels, y_score: predicted scores
fpr, tpr, thresholds = roc_curve(total_ground_truth, total_predictions)
roc_auc = roc_auc_score(total_ground_truth, total_predictions)

# After calculating accuracy and AOC-ROC score
# with open('results_gem_pro_def.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Ground Truth", "Predicted", "FPR", "TPR"])
#     writer.writerow([str(total_ground_truth.tolist()), str(total_predictions.tolist()), str(fpr.tolist()), str(tpr.tolist())])

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
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
