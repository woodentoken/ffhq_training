import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import FFHQDataset
from torch.utils.data import DataLoader
from copy import deepcopy

def split_data(data_set, split_ratios=(0.8, 0.1, 0.1)):
    #pudb.set_trace()
    train_split_index = int(len(data_set) * split_ratios[0])
    validation_split_index = train_split_index + int(len(data_set) * split_ratios[1])
    test_split_index = validation_split_index + int(len(data_set) * split_ratios[2])
    return data_set[:train_split_index], data_set[train_split_index:validation_split_index], data_set[test_split_index:]

def subset_data(data_set, subset_size=0):
    if subset_size == 0:
        subset_size = len(data_set)
        
    return data_set[:subset_size]

# Training loop
def train_model(model, dataloader, num_epochs=1, save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define loss function and optimizer
    numeric_criterion = nn.MSELoss()  # Mean Squared Error
    binary_criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = model.to(device)
    
    sigmoid = nn.Sigmoid()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = sigmoid(model(inputs))
            loss = numeric_criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.data
        
        epoch_loss = running_loss / len(inputs)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
    print('Training complete')

    if save:
        torch.save(model.state_dict(), 'model.pth')
        
    return model

def eval(model, test_loader, criterion=nn.MSELoss()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    sigmoid = nn.Sigmoid()
    running_loss = 0.0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = sigmoid(model(inputs))
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.data
    epoch_loss = running_loss / len(inputs)
    print(f'Loss: {epoch_loss:.4f}')
    print(preds.round(decimals=1))

def main(type_run='sample'):
    input_path = os.getcwd()
    if type_run == 'data':
        dataset_path = '/ffhq_dataset/'
        data_source = input_path + dataset_path + 'processed_image_data_set_128x128.csv'
    else:
        dataset_path = '/sample/'
        data_source = input_path + dataset_path + 'processed_image_data_set_sample.csv'

    data = pd.read_csv(data_source)
    attributes = data.drop(columns=['image_path'])

    train_data, validation_data, test_data = split_data(data, split_ratios =(0.8, 0.1, 0.1));

    transform = transforms.Compose([                      # Convert tensor to PIL image
        transforms.ToTensor(),                          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])
    train_dataset = FFHQDataset.FFHQDataset(train_data, dataset_path, transform=transform)
    validation_dataset = FFHQDataset.FFHQDataset(validation_data, dataset_path, transform=transform)
    test_dataset = FFHQDataset.FFHQDataset(test_data, dataset_path, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=5, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
    
    untrained_model = torchvision.models.resnet50(weights=None)
    num_features = untrained_model.fc.in_features
    untrained_model.fc = nn.Linear(num_features, 9)  # Assuming 9 classes
    # if model.pth exists, load the model
    if os.path.exists('model.pth'):
        untrained_model.load_state_dict(torch.load('model.pth'))
        trained_model = deepcopy(untrained_model)
    else:
        trained_model = train_model(
            untrained_model, 
            train_dataloader, 
            num_epochs=3, 
            save=True)

    eval(trained_model, validation_loader)
            
if __name__ == '__main__':
    main('data')