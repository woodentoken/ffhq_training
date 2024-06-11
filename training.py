import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import FFHQDataset
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from copy import deepcopy

def split_data(data_set, split_ratios=(0.8, 0.1, 0.1)):
    train_split_index = int(len(data_set) * split_ratios[0])
    train_set = data_set[:train_split_index]
    
    validation_split_index = train_split_index + int(len(data_set) * split_ratios[1])
    validation_set = data_set[train_split_index:validation_split_index]
    
    test_split_index = validation_split_index + int(len(data_set) * split_ratios[2])
    test_set = data_set[validation_split_index:test_split_index]
    
    for subset in (train_set, validation_set, test_set):
        subset.reset_index(drop=True, inplace=True)
        print(len(subset))
    
    return train_set, validation_set, test_set

def subset_data(data_set, subset_size=0):
    if subset_size == 0:
        subset_size = len(data_set)
        
    return data_set[:subset_size]

# Training loop
def train_model(model, dataloader, num_epochs=1, save=True, resolution=(128, 128)):
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
        
        exp_lr_scheduler.step()
        epoch_loss = running_loss / len(inputs)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
    print('Training complete')

    if save:
        torch.save(model.state_dict(), f'model_{resolution}.pth')
        
    return model

def eval(model, test_loader, criterion=nn.MSELoss()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    sigmoid = nn.Sigmoid()
    
    # reset the epoch loss (probably, epoch will always be 1 for testing)
    mse_running_loss = 0.0
    
    print(f'test loader size: {len(test_loader)}')
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # reset the batch loss
        mse_batch_running_loss = 0.0
        with torch.no_grad():
            outputs = sigmoid(model(inputs))
            
            mse_loss = criterion(outputs, labels)
            mse_running_loss += mse_loss.item() * inputs.size(0)
            mse_batch_running_loss += mse_loss.item() * inputs.size(0)

            preds = outputs.data
    
        batch_loss = mse_batch_running_loss / len(inputs)
        print(f'Batch Loss: {batch_loss:.4f}')
    # compute epoch loss with mse
    epoch_loss = mse_running_loss / len(test_loader.dataset)
            
    #epoch_loss = mse_running_loss / len(inputs)
    print(f'\nEpoch Loss: {epoch_loss:.4f}')
    
    # print the last 5 predictions and labels
    print(f"ground labels: {labels[-3:].round(decimals=2)}")
    print(f"predictions: {preds[-3:].round(decimals=2)}")

def main(type_run='sample', resolution=(128, 128)):
    input_path = os.getcwd()
    if type_run == 'data':
        dataset_path = '/ffhq_dataset/'
        data_source = input_path + dataset_path + 'processed_image_data_set_128x128.csv'
    else:
        dataset_path = '/sample/'
        data_source = input_path + dataset_path + 'processed_image_data_set_sample.csv'

    data = pd.read_csv(data_source)
    attributes = data.drop(columns=['image_path'])

    train_data, validation_data, test_data = split_data(data, split_ratios =(0.8, 0.01, 0.18));

    transform = transforms.Compose([
        transforms.ToTensor(),                          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])
    
    motion_blur_transform = transforms.Compose([
        #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        #transforms.Resize(resolution),
        #transforms.GaussianBlur(kernel_size=11),
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        
        # normalization makes the image look weird, but it's necessary for the model, just a heads up
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    Image.open(os.getcwd() + validation_data['image_path'].values[5]).convert('RGB').show()
    
    train_dataset = FFHQDataset.FFHQDataset(train_data, dataset_path, transform=transform)
    validation_dataset = FFHQDataset.FFHQDataset(validation_data, dataset_path, transform=motion_blur_transform)
    #validation_dataset = FFHQDataset.FFHQDataset(validation_data, dataset_path, transform=transform)
    test_dataset = FFHQDataset.FFHQDataset(test_data, dataset_path, transform=transform)
    
    # visualize one image from the validation dataset
    image, label = validation_dataset[5]    
    from torchvision.utils import save_image, make_grid
    img1 = image
    save_image(make_grid(img1), f'img_{resolution}_sample.png')

    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print(len(test_loader))

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
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