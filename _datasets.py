import csv
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

DATASETS_PATH = Path('./data')
transform = ToTensor()

class BaseOCRDataset(Dataset):
    def __init__(self, dataset_path, transform=transform):
        self.dataset_path = Path(dataset_path)
        self.transform = transform

    def preprocess_image(self, image):
        if self.transform:
            return self.transform(image)
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__ method")

class IDPLPFOD2Dataset(BaseOCRDataset):
    def __init__(self, dataset_path=DATASETS_PATH / "IDPL-PFOD2/IDPL-PFOD2-dataset", split="train", transform=None):
        super().__init__(dataset_path, transform)
        self.split = split
        self.data = self._load_data()

    def _load_data(self):
        split_file = self.dataset_path / f"{self.split}.txt"
        data = []
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                image_name, text = line.strip().split(",", 1)
                image_path = self.dataset_path / self.split / image_name
                data.append({"image_path": image_path, "text": text})
        return data

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert('RGB')
        image = self.preprocess_image(image)
        return image, item["text"]

class ShotorDataset(BaseOCRDataset):
    def __init__(self, dataset_path=DATASETS_PATH / "Shotor", transform=None):
        super().__init__(dataset_path, transform)
        self.csv_file = self.dataset_path / "Shotor_Words.csv"
        self.data = self._load_data()

    def _load_data(self):
        data = []
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = self.dataset_path / "Shotor_Images" / row["image"]
                data.append({"image_path": image_path, "text": row["word"]})
        return data

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert('RGB')
        image = self.preprocess_image(image)
        return image, item["text"]

class WordByWordDataset(BaseOCRDataset):
    def __init__(self, dataset_path=DATASETS_PATH / 'Arshasb/Arshasb_7k', page_id=1, transform=None):
        super().__init__(dataset_path, transform)
        self.page_id = f"{int(page_id):05d}"
        self.data = self._load_data()

    def _load_data(self):
        label_file = self.dataset_path / self.page_id / f"label_{self.page_id}.xlsx"
        image_file = self.dataset_path / self.page_id / f"page_{self.page_id}.png"
        image = Image.open(image_file).convert('RGB')
        data = []
        df = pd.read_excel(label_file)
        for _, row in df.iterrows():
            bbox = [
                tuple(map(int, row['point1'][1:-1].split(','))),
                tuple(map(int, row['point2'][1:-1].split(','))),
                tuple(map(int, row['point3'][1:-1].split(','))),
                tuple(map(int, row['point4'][1:-1].split(',')))
            ]
            cropped_image = image.crop((
                bbox[0][0], bbox[0][1],
                bbox[2][0], bbox[2][1]
            ))
            data.append({"image": cropped_image, "word": row['word']})
        return data

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.preprocess_image(item["image"])
        return image, item["word"]

class LineByLineDataset(BaseOCRDataset):
    def __init__(self, dataset_path=DATASETS_PATH / 'Arshasb/Arshasb_7k', page_id=1, transform=None):
        super().__init__(dataset_path, transform)
        self.page_id = f"{int(page_id):05d}"
        self.data = self._load_data()

    def _load_data(self):
        line_file = self.dataset_path / self.page_id / f"line_{self.page_id}.xlsx"
        text_file = self.dataset_path / self.page_id / f"fulltext_{self.page_id}.txt"
        image_file = self.dataset_path / self.page_id / f"page_{self.page_id}.png"
        image = Image.open(image_file).convert('RGB')
        data = []
        df = pd.read_excel(line_file)
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, row in df.iterrows():
            bbox = [
                tuple(map(int, row['point1'][1:-1].split(','))),
                tuple(map(int, row['point2'][1:-1].split(','))),
                tuple(map(int, row['point3'][1:-1].split(','))),
                tuple(map(int, row['point4'][1:-1].split(',')))
            ]
            cropped_image = image.crop((
                bbox[0][0], bbox[0][1],
                bbox[2][0], bbox[2][1]
            ))
            data.append({"image": cropped_image, "line_text": lines[i].strip()})
        return data

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.preprocess_image(item["image"])
        return image, item["line_text"]


if __name__ == "__main__":
    idpl_dataset = IDPLPFOD2Dataset()
    shotor_dataset = ShotorDataset()
    word_dataset = WordByWordDataset(page_id=7000)
    line_dataset = LineByLineDataset(page_id=7000)

    BS = 4
    basic_collate = lambda x: list(zip(*x))
    idpl_loader = DataLoader(idpl_dataset, batch_size=BS, collate_fn=basic_collate)
    shotor_loader = DataLoader(shotor_dataset, batch_size=BS, collate_fn=basic_collate)
    word_loader = DataLoader(word_dataset, batch_size=BS, collate_fn=basic_collate)
    line_loader = DataLoader(line_dataset, batch_size=BS, collate_fn=basic_collate)

    loaders = [
        idpl_loader,
        shotor_loader,
        word_loader,
        line_loader,
    ]
    for loader in loaders:
        for images, texts in loader:
            print(images, texts)
            break
