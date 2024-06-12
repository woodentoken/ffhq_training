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
import numpy as np
import matplotlib.pyplot as plt

def split_data(data_set, split_ratios=(0.8, 0.1, 0.1)):
    train_split_index = int(len(data_set) * split_ratios[0])
    train_set = data_set[:train_split_index]
    
    # sum the values of all classes in the train set
    #print(len(train_set)/(train_set.drop(columns=['image id', 'image_path', 'age']).sum()))
    label_sums = train_set.drop(columns=['image id', 'image_path', 'age']).sum()
    norm_label_sums = 100*label_sums/len(train_set)
    weighties_label_sums = 1/(label_sums/len(train_set))
    
    # dataset composition
    print(norm_label_sums)
    print(weighties_label_sums)

    # weighting for MSE loss
    train_weighting = 1/(label_sums/len(train_set))
    train_weighting = train_weighting.to_list()
    #print(train_weighting)
    
    # plot a bar graph of the class plt.rcParams.update({'font.size': 14})tribution
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax2=ax.twinx()
    norm_label_sums.plot(kind='bar', ax=ax, position=1, width=0.3, color='b', label='Class Distribution')
    weighties_label_sums.plot(kind='bar', ax=ax2, position=0.5, width=0.3, color='r', label='Class Weighting')
    plt.title('FFHQ Condensed Class Distribution')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentage of dataset')
    ax.yaxis.label.set_color('b')
    ax2.set_ylabel('MSE Weighting')
    ax2.yaxis.label.set_color('r')
    fig.tight_layout()
    plt.savefig('class_distribution.png')
    
    validation_split_index = train_split_index + int(len(data_set) * split_ratios[1])
    validation_set = data_set[train_split_index:validation_split_index]
    
    test_split_index = validation_split_index + int(len(data_set) * split_ratios[2])
    test_set = data_set[validation_split_index:test_split_index]
    
    for subset in (train_set, validation_set, test_set):
        subset.reset_index(drop=True, inplace=True)
    
    return train_set, validation_set, test_set, train_weighting

def subset_data(data_set, subset_size=0):
    if subset_size == 0:
        subset_size = len(data_set)
        
    return data_set[:subset_size]

# Training loop
def train_model(model, dataloader, loss_weighting, num_epochs=1, save=True, resolution=(128, 128)):
    def mse_loss(input, target):
        return torch.sum((input - target) ** 2)

    def weighted_mse_loss(input, target, weight):
        return torch.sum(weight * (input - target) ** 2) / weight.sum()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define loss function and optimizer
    numeric_criterion = nn.MSELoss()  # Mean Squared Error
    binary_criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = model.to(device)
    
    class WeightedMSELoss(nn.Module):
        def __init__(self, weight):
            super(WeightedMSELoss, self).__init__()
            self.weight = torch.Tensor(weight).to(device)

        def forward(self, input, target):
            return torch.sum(torch.mean(self.weight * (input - target) ** 2)) / self.weight.sum()
        
    class MSELoss(nn.Module):
        def __init__(self):
            super(MSELoss, self).__init__()

        def forward(self, input, target):
            return torch.sum(torch.mean((input - target) ** 2))
        
    mseloss = MSELoss()
    wmseloss = WeightedMSELoss(loss_weighting)
    training_loss = []
    sigmoid = nn.Sigmoid()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        mse_loss_per_cat = [0.0]*len(loss_weighting)

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = sigmoid(model(inputs))
            #loss = numeric_criterion(outputs, labels)
            
            # for batch_index, output in enumerate(outputs):
            #     for index, category_value in enumerate(output):
            #         # print(f'pred: {category_value}')
            #         # print(f'lab: {labels[batch_index][index]}\n')
            #         #category_loss = weighted_mse_loss(category_value, labels[batch_index][index], loss_weighting[index])
            #         category_loss = mseloss(category_value, labels[batch_index][index])
            #         mse_loss_per_cat[index] += category_loss.item()
                    
            loss = wmseloss(outputs, labels)
            #loss = sum(mse_loss_per_cat)
            #print(loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.data
        
        exp_lr_scheduler.step()
        epoch_loss = running_loss / len(inputs)
        training_loss.append({'epoch': epoch+1, 'loss': epoch_loss, 'resolution': resolution[0]})
        tl_df = pd.DataFrame(training_loss).set_index('resolution')
        print(tl_df)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
    to_csv(tl_df, 'training_loss.csv')
    print('Training complete')

    if save:
        resolution_string = f'{resolution[0]}x{resolution[1]}'
        print(f'Saving model at model_{resolution_string}.pth')
        torch.save(model.state_dict(), f'model_{resolution_string}.pth')
        print('Model saved')
        
    return model

def eval(model, test_loader, criterion=nn.MSELoss()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    sigmoid = nn.Sigmoid()
    
    # reset the epoch loss (probably, epoch will always be 1 for testing)
    mse_running_loss = 0
    mae_running_loss = 0
    rmse_running_loss = 0
    mse_batch_loss = 0
    
    num_batches = len(test_loader)
    
    # this is per batch
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        batch_size = len(inputs)
        #print(batch_size)
        label_count = len(labels[0])
        
        # reset the batch loss
        mse_batch_running_loss = 0.0
        mse_batch_running_loss2 = 0.0
        mse_batch_loss_per_cat = [0.0]*len(labels[0])

        with torch.no_grad():
            outputs = sigmoid(model(inputs))
            #print(outputs)
            
            # track the loss per category
            mse_loss_per_cat = [0.0]*len(outputs[0])
            running_mse_loss_per_cat = [0.0]*len(outputs[0])
            # this is per actual entry
            for batch_index, output in enumerate(outputs):
                for index, category_value in enumerate(output):
                    # print(f'pred: {category_value}')
                    # print(f'lab: {labels[batch_index][index]}\n')
                    category_loss = criterion(category_value, labels[batch_index][index])
                    mse_loss_per_cat[index] += category_loss.item()
            
            running_mse_loss_per_cat = [running_mse_loss_per_cat[i] + mse_loss_per_cat[i] for i in range(len(mse_loss_per_cat))]
            running_mse_loss_per_cat_normalized = [x / label_count/batch_size for x in running_mse_loss_per_cat]
            #print(running_mse_loss_per_cat_normalized)
            
            # sum the loss per category by index
            mse_loss_per_cat = [x for x in mse_loss_per_cat]
            mse_batch_loss = sum(mse_loss_per_cat)/label_count/batch_size
            #print(mse_batch_loss)
                
            # # this does the same thing as above but in aggregate
            mse_loss = criterion(outputs, labels)
            mae_loss = nn.L1Loss()(outputs, labels)
            rmse_loss = torch.sqrt(mse_loss)
            #print(f'loss: {mse_loss}')
            
            mse_running_loss += mse_batch_loss * inputs.size(0)
            mae_running_loss += mae_loss * inputs.size(0)
            rmse_running_loss += rmse_loss * inputs.size(0)
            
            mae_batch_running_loss = mae_running_loss * inputs.size(0)
            
            mse_batch_running_loss += mse_batch_loss * inputs.size(0)
            mse_batch_running_loss2 += mse_loss * inputs.size(0)

            preds = outputs.data
    
        average_batch_loss = mse_batch_running_loss / batch_size
        average_mae_batch_loss = mae_batch_running_loss / batch_size
        average_batch_loss2 = mse_batch_running_loss2 / batch_size
        #print(f'Batch Loss: {average_batch_loss:.4f}')
        #print(f'Batch2 Loss: {average_batch_loss2:.4f}')
        
    # compute epoch loss with mse
    num_batches = len(test_loader.dataset)
    e_l = average_batch_loss/num_batches
    #print(e_l)
    epoch_loss = mse_running_loss / num_batches
    mae_epoch_loss = mae_running_loss / num_batches
    rmse_epoch_loss = rmse_running_loss / num_batches
    #print(f'\nEpoch Loss: {epoch_loss:.4f}')
    
    # print the last 5 predictions and labels
    print(f"ground labels: {labels[-3:].round(decimals=2)}")
    print(f"predictions: {preds[-3:].round(decimals=2)}")
    
    return epoch_loss, mae_epoch_loss.item(), rmse_epoch_loss.item()

import os.path
def to_csv(dataframe, file_name):
    if os.path.exists(file_name):
        dataframe.to_csv(file_name, mode='a', header=False)
    else:
        dataframe.to_csv(file_name)

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

    train_data, validation_data, test_data, train_weighting = split_data(data, split_ratios =(0.5, 0.15, 0.15));
    transform = transforms.Compose([
        transforms.Resize(resolution),  # Resize image to 128x128
        transforms.ToTensor(),                          # Convert image to tensor
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])
    
    transform2 = transforms.Compose([
        transforms.Resize(resolution),  # Resize image to 128x128
        #transforms.RandomRotation((180,180)),
        transforms.ToTensor(),                          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])
    #print(validation_data)
    
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
    resolution_string = f'{resolution[0]}x{resolution[1]}'
    untrained_model = torchvision.models.resnet50(weights=None)
    num_features = untrained_model.fc.in_features
    untrained_model.fc = nn.Linear(num_features, 4)  # Assuming 9 classes
    # if model.pth exists, load the model
    if os.path.exists(f'model_{resolution_string}.pth'):
        print('Loading model')
        untrained_model.load_state_dict(torch.load(f'model_{resolution_string}.pth'))
        trained_model = deepcopy(untrained_model)
        print('Model loaded')
    else:
        print('Training model')
        train_dataset = FFHQDataset.FFHQDataset(train_data, dataset_path, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        trained_model = train_model(
            untrained_model, 
            train_dataloader, 
            num_epochs=10, 
            save=True,
            resolution=resolution,
            loss_weighting=train_weighting)
        print('Model trained')
    
    print('Starting evaluation')
    loss_tracker = []
    
    blur_run = False
    if blur_run == True:
        for blur_sigma in np.arange(0, 5.1, 0.1):
            print(blur_sigma )
            validation_dataset = FFHQDataset.FFHQDataset(validation_data, dataset_path, transform=transform, blur_degree=blur_sigma)
            validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=True)
            
            # visualize one image from the validation dataset
            image, label = validation_dataset[0]    
            from torchvision.utils import save_image, make_grid
            img1 = image
            save_image(make_grid(img1), f'transform_images/img_{resolution_string}_blur{round(blur_sigma,2)}_sample.png')
            loss_tracker.append({'blur_degree': blur_sigma, 'MSE loss': eval(trained_model, validation_loader)[0], 'MAE loss': eval(trained_model, validation_loader)[1], 'RMSE loss': eval(trained_model, validation_loader)[2]})
            
        to_csv(pd.DataFrame(loss_tracker), 'blur_loss.csv')
    elif blur_run == 'kaleb':
        print('here')
        test_data = {'image_path': ['/kaleb/kaleb_images/IMG_2842.jpg'], 'age': [32], 'positive': [0], 'negative': [1], 'neutral': [0]}
        test_data = pd.DataFrame(test_data)
        test_dataset = FFHQDataset.FFHQDataset(test_data, '/kaleb/', transform=transform2, blur_degree=0)
        
    # visualize one image from the validation dataset
        image, label = test_dataset[0]    
        from torchvision.utils import save_image, make_grid
        img1 = image
        save_image(make_grid(img1), f'transform_images/img_{resolution_string}_kaleb_sample.png')
        
        eval(trained_model, DataLoader(test_dataset, batch_size=1, shuffle=False))
    else:
        validation_dataset = FFHQDataset.FFHQDataset(validation_data, dataset_path, transform=transform)
        validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=True)
        loss_tracker.append({'resolution': {resolution_string}, 'MSE loss': eval(trained_model, validation_loader)[0], 'MAE loss': eval(trained_model, validation_loader)[1], 'RMSE loss': eval(trained_model, validation_loader)[2]})
        to_csv(pd.DataFrame(loss_tracker), 'loss.csv')
    print('Evaluation complete')
        
    print(loss_tracker)
            
if __name__ == '__main__':
    resolutions = [(8,8), (16,16), (24,24), (32,32), (48,48), (64,64), (96,96), (128,128)]
    for resolution in resolutions:
        main('data', resolution)
    #main('data', (128,128))