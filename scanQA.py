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

        print(data["question"])
        processed_question = self.processor.tokenizer(data["question"], return_tensors="pt", padding="max_length", max_length=64, truncation=True)['input_ids']
        processed_answer = self.processor.tokenizer(data["answers"][0], return_tensors="pt", padding="max_length", max_length=64, truncation=True)['input_ids']

        return {
            "pc_features": pc_features,
            "text": data["question"],
            "tokenized_answer": processed_answer,
            "tokenized_question": processed_question
        }

def collate_fn(batch, processor):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch