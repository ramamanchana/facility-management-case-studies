# Before running the script, ensure you have the necessary libraries installed:

# pip install pandas numpy matplotlib scikit-learn transformers torch

# Explanation
# Sample Data Creation: The script creates a sample dataset and saves it as contracts.csv. This sample data includes contract texts and compliance labels.
# Data Loading and Preprocessing: The script reads the sample data and preprocesses it using the BERT tokenizer.
# Model Definition: A BERT-based model is defined for contract compliance prediction.
# Training and Evaluation: The model is trained and evaluated, with results such as loss, accuracy, and confusion matrix being computed.
# Visualization: The results are visualized using matplotlib, showing the confusion matrix and loss plots.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sample data creation
data = {
    'Contract Text': [
        "This contract outlines the agreement between Vendor A and the client. Vendor A agrees to deliver services...",
        "The contract specifies the terms and conditions for Vendor B to provide maintenance services...",
        "Vendor C will supply the materials as per the contract terms and conditions. Compliance is mandatory...",
        "This contract is for the delivery of goods by Vendor D. All performance standards must be met...",
        "Vendor E agrees to provide consulting services under the terms specified in this contract..."
    ],
    'Compliance': [1, 0, 1, 0, 1]
}

# Create a DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('contracts.csv', index=False)

# Load the data
data = pd.read_csv('contracts.csv')

# Data Preprocessing
class ContractDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        contract_text = self.data.iloc[index]['Contract Text']
        compliance = self.data.iloc[index]['Compliance']
        inputs = self.tokenizer.encode_plus(
            contract_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(compliance, dtype=torch.float)
        }

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 512
batch_size = 8

# Prepare the data
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = ContractDataset(train_data, tokenizer, max_len)
val_dataset = ContractDataset(val_data, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the model
class ContractClassifier(nn.Module):
    def __init__(self):
        super(ContractClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, mask):
        _, pooled_output = self.bert(ids, attention_mask=mask, return_dict=False)
        x = self.fc(pooled_output)
        return self.sigmoid(x)

model = ContractClassifier()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training the model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)
        outputs = model(ids, mask)
        loss = criterion(outputs, targets.unsqueeze(1))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in data_loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets'].to(device)
            outputs = model(ids, mask)
            loss = criterion(outputs, targets.unsqueeze(1))
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    return total_loss / len(data_loader), predictions, true_labels

num_epochs = 3
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, predictions, true_labels = eval_model(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')

# Evaluate the model
predictions = [1 if p > 0.5 else 0 for p in predictions]
accuracy = accuracy_score(true_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predictions)
class_report = classification_report(true_labels, predictions)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Visualizing the results
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

import itertools

plt.figure(figsize=(10, 7))
plot_confusion_matrix(conf_matrix, classes=['Non-compliant', 'Compliant'])
plt.show()

# Plotting the accuracy
epochs = range(1, num_epochs + 1)
train_losses = [train_epoch(model, train_loader, criterion, optimizer, device) for epoch in epochs]
val_losses = [eval_model(model, val_loader, criterion, device)[0] for epoch in epochs]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, 'bo', label='Training loss')
plt.plot(epochs, val_losses, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
