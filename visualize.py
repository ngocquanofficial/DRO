import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import Trainer
from src.model import ClassificationModel
from src.data import DataModule
import numpy as np
import matplotlib.pyplot as plt
import random
# Load the trained model checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "/home/ubuntu/DRO/checkpoint/cifar100-step-step=1550-val_acc=0.7700.ckpt"  # Replace with your actual path
model = ClassificationModel.load_from_checkpoint(checkpoint_path)
model.eval()
model.to(device)

data_module = DataModule(
    dataset="cifar100",   # Specify the dataset
    root="../vtab-1k/cifar100",         # Adjust this path if needed
    size=224,             # Image size (change if necessary)
    batch_size=32,        # Adjust the batch size
    workers=4             # Number of data loader workers
)

# Prepare and set up the data
data_module.prepare_data()
data_module.setup(stage="test")

# Get the test dataloader
test_dataloader = data_module.test_dataloader()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
misclassified_samples = []
sais = 0
total = 0
# Iterate over the test dataset to collect misclassified samples
with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Get model predictions
        outputs = model(inputs)
        stack_outputs = torch.stack(outputs)
        pred = torch.mean(stack_outputs, dim= 0)
        probabilities = F.softmax(pred, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        
        # Identify misclassified samples
        misclassified_indices = (predicted_labels != labels).nonzero(as_tuple=True)[0]
        total += labels.shape[0]
        sais += len(misclassified_indices)


        for idx in misclassified_indices:
            ane = random.randint(0, 100)
            if ane > 5 :
                continue
            x = inputs[idx].unsqueeze(0).to(device) 
            label = labels[idx]
            label = label.item()
            preds = model(x)
            print("Start with sample ", idx + 1, "LABEL: ", label)


            fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # 2x2 grid of subplots

            for i, pred in enumerate(preds):
                prob = F.softmax(pred, dim=1).squeeze(0).cpu().numpy()

                # Determine the row and column in the subplot grid
                row, col = divmod(i, 2)

                axes[row, col].bar(range(len(prob)), prob, color='skyblue', label='Prediction Probabilities')
                axes[row, col].bar(label, prob[label], color='orange', label=f'True Label (Class {label})')
                axes[row, col].set_xlabel('Class Index')
                axes[row, col].set_ylabel('Probability')
                axes[row, col].set_title(f'Model {i+1} Prediction Probability Histogram')
                axes[row, col].legend()

            plt.tight_layout()
            plt.savefig(f"./images/{idx}-th index_ batch {batch_idx}.png")
            plt.show()


print(sais, total)
print(sais/total)


# # Visualize the top 20 probabilities for each misclassified sample
# for i, sample in enumerate(misclassified_samples):
#     top20_indices = np.argsort(sample["probabilities"])[-20:][::-1]  # Get the top 20 indices
#     top20_probs = sample["probabilities"][top20_indices]
    
#     plt.figure(figsize=(12, 6))
#     plt.bar(range(20), top20_probs, color='skyblue')
#     plt.title(f"Misclassified Sample {i + 1}: True Label = {sample['label']}, Predicted = {sample['predicted']}")
#     plt.xlabel("Class (Top 20)")
#     plt.ylabel("Probability")
#     plt.xticks(range(20), top20_indices, rotation=45)
#     plt.show()
#     print("SHOW")
