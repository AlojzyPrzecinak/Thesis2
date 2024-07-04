import json
import random

def calculate_accuracy_per_class(results_file, original_file, target_dataset):
    class_stats = {}
    class_instances = {}

    # Load original data
    original_data = {}
    with open(original_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            img_name = data['img'].split('/')[-1]
            class_name = data['label'][1] if len(data['label']) > 1 else 'unclassified'
            original_data[img_name] = class_name

    # Calculate accuracy
    with open(results_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data['dataset'] == target_dataset:
                img_name = data['img'].split('\\')[-1]
                class_name = original_data.get(img_name, 'unclassified')
                if class_name not in class_stats:
                    class_stats[class_name] = {'correct': 0, 'total': 0}
                    class_instances[class_name] = []
                if data['ground_label'] == data['predicted_label']:
                    class_stats[class_name]['correct'] += 1
                class_stats[class_name]['total'] += 1
                class_instances[class_name].append(data)

    # Calculate and print accuracy for each class
    for class_name, stats in class_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f'Accuracy for class {class_name}: {accuracy}')
        else:
            print(f'No data for class: {class_name}')

    # Select and print 4 random instances for each class
    for class_name, instances in class_instances.items():
        if len(instances) >= 4:
            sampled_instances = random.sample(instances, 4)
            print(f'4 random instances for class {class_name}: {sampled_instances}')
        else:
            print(f'Not enough instances for class: {class_name}')

results_file = 'ClipResults.jsonl'
original_file = 'HarmP/img/concatenated.jsonl'
target_dataset = 'HarmP'
calculate_accuracy_per_class(results_file, original_file, target_dataset)