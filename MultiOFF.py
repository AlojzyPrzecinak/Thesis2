# run a prediction loop for the model using the MultiOFF dataset
from Dataset import Dataset
from Model import Model
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import csv
from PIL import Image

data_dir = 'MultiOFF'
img_path = data_dir + '/img/Labelled Images'
train_path = data_dir + '/Training_meme_dataset.csv'

#train_data = Dataset(train_path)
label_mapping = {'Non-offensiv': 0, 'offensive': 1}


ground_truth = []
predicted = []
model = Model()

with open('MultiOFF/Training_meme_dataset.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        image_name, text, label = row
        image = Image.open(img_path + '/' + image_name)  # Load the image from the file
        numeric_label = label_mapping[label]  # Convert the label to a numeric value
        ground_truth.append(numeric_label)
        prediction = model.predict(image, text)
        predicted.append(prediction)

# Calculate accuracy
total_predictions = np.array(predicted)
total_ground_truth = np.array(ground_truth)
accuracy = np.mean((total_predictions == total_ground_truth).astype(np.float64)) * 100
print('The accuracy of the model is %.2f' % (accuracy) + '%')

# y_true: true labels, y_score: predicted scores
fpr, tpr, thresholds = roc_curve(total_ground_truth, total_predictions)
roc_auc = roc_auc_score(total_ground_truth, total_predictions)

print("AOC-ROC score: {:.2f}".format(roc_auc))
