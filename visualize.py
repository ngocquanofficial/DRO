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
checkpoint_path = "/home/ubuntu/newDRO/sam_best_val_acc_0.8100000023841858.ckpt"  # Replace with your actual path
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


true_label= 9
predictions_per_model = [[], [], [], []]  # Collect predictions for each model for samples with label 0
print("GO here")
# Iterate over the test dataset to collect predictions for label=0 samples
with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        # Filter samples where label == 0
        zero_label_indices = (labels == true_label).nonzero(as_tuple=True)[0]

        if len(zero_label_indices) > 0:
            # Get model predictions for these filtered samples
            outputs = model(inputs[zero_label_indices])
            for i, output in enumerate(outputs):
                predictions_per_model[i].append(output)


# Calculate the average prediction per model for samples with label=0
avg_predictions_per_model = []
for i, predictions in enumerate(predictions_per_model):
    if len(predictions) > 0:
        # Concatenate the predictions to make a single tensor with shape [N, 100]
        stacked_predictions = torch.cat(predictions, dim=0)  # Concatenate along the batch dimension
        print(stacked_predictions.shape)
        avg_prediction = torch.mean(stacked_predictions, dim=0)
        print(avg_prediction.shape)
        avg_predictions_per_model.append(avg_prediction)




# Plotting histograms for each model based on the average prediction
fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # 2x2 grid of subplots

# Define colors and edge styles
bar_color = '#f0f0f0'  # Very light gray, nearly white
bar_edge_color = 'black'
highlight_color = 'skyblue'
highlight_edge_color = 'black'

for i, avg_prediction in enumerate(avg_predictions_per_model):
    if avg_prediction is not None:
        prob = F.softmax(avg_prediction, dim=-1).cpu().numpy()

        # Determine the row and column in the subplot grid
        row, col = divmod(i, 2)

        # Plot the histogram bars
        bars = axes[row, col].bar(
            range(len(prob)), prob, color=bar_color, edgecolor=bar_edge_color, label='Prediction Probabilities', linewidth= 2
        )

        # Highlight the true label bar
        bars[true_label].set_color(highlight_color)
        bars[true_label].set_edgecolor(highlight_edge_color)

        # Set titles and labels
        axes[row, col].set_xlabel('Class Index')
        axes[row, col].set_ylabel('Probability')
        axes[row, col].set_title(f'Model {i+1} Average Prediction Probability Histogram for Label 0')
        axes[row, col].legend()

plt.tight_layout()
plt.savefig("./images/average_predictions_histogram_update.png")
plt.show()













# # Plotting histograms for each model based on the average prediction
# fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # 2x2 grid of subplots

# for i, avg_prediction in enumerate(avg_predictions_per_model):
#     if avg_prediction is not None:
#         prob = F.softmax(avg_prediction, dim=-1).cpu().numpy()

#         # Determine the row and column in the subplot grid
#         row, col = divmod(i, 2)

#         axes[row, col].bar(range(len(prob)), prob, color='skyblue', label='Prediction Probabilities')
#         axes[row, col].bar(true_label, prob[true_label], color='orange', label=f'True Label (Class {true_label})')
#         axes[row, col].set_xlabel('Class Index')
#         axes[row, col].set_ylabel('Probability')
#         axes[row, col].set_title(f'Model {i+1} Average Prediction Probability Histogram for Label 0')
#         axes[row, col].legend()

# plt.tight_layout()
# plt.savefig("./images/average_predictions_histogram_update.png")
# plt.show()
