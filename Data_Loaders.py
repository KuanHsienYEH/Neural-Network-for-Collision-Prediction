import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
        return len(self.normalized_data)
        

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        
        x = self.normalized_data[idx, :6].astype(np.float32)
        y = self.normalized_data[idx, 6].astype(np.float32)
        return {'input': x, 'label': y}


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        train_size = int(0.8 * len(self.nav_dataset))
        test_size = len(self.nav_dataset) - train_size

        # Randomly split the dataset
        self.train_dataset, self.test_dataset = data.random_split(self.nav_dataset, [train_size, test_size])

        # Create DataLoader for both training and testing
        self.train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)




def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()