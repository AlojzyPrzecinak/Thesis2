import json
import os
from PIL import Image
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(l) for l in f]
            self.data_dir = os.path.dirname(data_path)

    def __getitem__(self, index: int):
        # Load images on the fly.
        image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
        text = self.data[index]["text"]
        label = self.data[index]["label"][0]

        return image, text, label

    def load_image_only(self, index: int):
        image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
        return image

    def get_label(self, index: int):
        label = self.data[index]["label"]
        return label

    def get_test_item(self, index: int):
        # Load images on the fly.
        image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
        text = self.data[index]["text"]

        return image, text

    def __len__(self):
        return len(self.data)
