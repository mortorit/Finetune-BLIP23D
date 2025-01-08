from torch.utils.data import Dataset
import json
import torch


class ScanQA_fake(Dataset):
    """
    ScanQA dataset with real questions and answers but fake point cloud features.
    """
    def __init__(self, json_path: str, processor):
        self.json_path = json_path
        self.data = self.load_data()
        self.processor = processor

    def load_data(self):
        # Load data from json file
        data = []

        #load json file
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        # Fake point cloud features
        pc_features = torch.randn(1, 256, 1408)

        processed_text = self.processor(None, data["question"], return_tensors="pt")['input_ids']

        return {
            "pc_features": pc_features,
            "question": data["question"],
            "answer": data["answers"][0],
            "tokenized_question": processed_text
        }
