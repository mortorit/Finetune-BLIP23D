from transformers import (
    Blip2Config,
    Blip2ForConditionalGeneration,
)
from transformers import AutoProcessor
from blip3d import Blip3D
import torch
from scanQA import ScanQA_fake
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

blip3d = Blip3D(model)

#print the number of trainable parameters of the qformer
print(sum(p.numel() for p in blip3d.blip_model.qformer.parameters() if p.requires_grad))

scanQA_dataset = ScanQA_fake("ScanQA_v1.0_train.json", processor)

inputs = scanQA_dataset[0]
pc_features_simulated = inputs["pc_features"]
inputs = inputs["tokenized_question"]

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
)

#set all parameters of blip3d to not trainable
for param in blip3d.parameters():
    param.requires_grad = False

blip3d.blip_model.qformer = get_peft_model(blip3d.blip_model.qformer, peft_config)

blip3d.blip_model.qformer.print_trainable_parameters()

#
# with torch.no_grad():
#     y = blip3d(pc_features_simulated,inputs)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip3d.to(device)

# Define optimizer
optimizer = AdamW(blip3d.blip_model.qformer.parameters(), lr=5e-5)

# Loss function
loss_fn = CrossEntropyLoss()

# Training loop
num_epochs = 3
blip3d.train()  # Put model in training mode

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        # Prepare inputs
        pc_features = batch["pc_features"].to(device)
        tokenized_question = batch["tokenized_question"].squeeze(1).to(device)
        answer = batch["answer"]

        # Forward pass
        outputs = blip3d(pc_features, tokenized_question)
        logits = outputs.logits  # Model outputs logits of size (batch_size, seq_len, vocab_size)

        # Shift logits and labels for loss computation
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokenized_question[:, 1:].contiguous()

        # Compute loss
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")

print("Training complete!")


