from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torch
import os

class FFHQDataset(Dataset):
    def __init__(self, dataframe, dataset_path, transform):
        self.image_paths = dataframe['image_path'].values
        self.attributes = dataframe.drop(columns=['image_path', 'age'])
        self.dataset_path = dataset_path
        self.transform = transform
        print(dataset_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_path = os.getcwd() + img_path
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.attributes is not None:
            label = self.attributes.iloc[idx, 1:].values.astype('float')
            return image, torch.FloatTensor(label)
        else:
            return image, torch.zeros(1)  # Return a dummy tensor for label