import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import Trainer
from src.model import ClassificationModel
from src.data import DataModule
import random

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import Trainer
from src.model import ClassificationModel
from src.data import DataModule
import random

print("FINISH IMPORTING")
# Load the trained model checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "/home/ubuntu/DRO/saved_checkpoints/svhn/acc91.88.ckpt"  # Replace with your actual path
model = ClassificationModel.load_from_checkpoint(checkpoint_path)
model.eval()
model.to(device)

data_module = DataModule(
    dataset="svhn",   # Specify the dataset
    root="../vtab-1k/svhn",         # Adjust this path if needed
    size=224,             # Image size (change if necessary)
    batch_size=32,        # Adjust the batch size
    workers=4             # Number of data loader workers
)

# Prepare and set up the data
data_module.prepare_data()
data_module.setup(stage="test")

# Get the test dataloader
test_dataloader = data_module.test_dataloader()
print(len(test_dataloader), "!!!!!!!!!!!!!!!!!!")

device = 'cuda' if torch.cuda.is_available() else 'cpu'



import torch
import matplotlib.pyplot as plt

# List to store confidence scores for each sample
confidence_scores = []

# Set a range of thresholds to test (e.g., from 0 to 1 with 0.01 intervals)
thresholds = torch.arange(0, 1.01, 0.01)

# Assuming `test_dataloader` is your data loader and `model` is your trained model
with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Get model outputs
        outputs = model(inputs)
        avg_predictions = torch.mean(torch.stack(outputs), dim=0)

        # Get softmax scores (probabilities)
        softmax_scores = torch.softmax(avg_predictions, dim=1)

        # Get the highest confidence (max softmax score for each sample)
        max_confidence, _ = torch.max(softmax_scores, dim=1)
        
        # Append the max confidence scores to the list
        confidence_scores.extend(max_confidence.cpu().numpy())

# Convert to a torch tensor for further processing
confidence_scores = torch.tensor(confidence_scores)

# Calculate the percentage of OOD samples (confidence lower than threshold) for each threshold
percent_ood_samples = []

for threshold in thresholds:
    # Calculate the number of samples where the confidence is lower than the threshold
    ood_samples = (confidence_scores < threshold).sum().item()
    
    # Calculate the percentage of OOD samples
    percent_ood = (ood_samples / len(confidence_scores)) * 100
    percent_ood_samples.append(percent_ood)

# Plotting
plt.plot(thresholds.numpy(), percent_ood_samples, label="Percent OOD Samples")
plt.xlabel('Threshold')
plt.ylabel('Percentage of OOD Samples')
plt.title('Threshold vs OOD Samples')
plt.grid(True)
plt.legend()
plt.savefig(f"./images/ood.pdf")
plt.show()
