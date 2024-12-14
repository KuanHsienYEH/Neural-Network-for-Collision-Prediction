from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()


    loss_function = nn.BCELoss()  #
    optimizer = optim.Adam(model.parameters(), lr=0.001)  
    
    train_losses = []
    test_losses = []


    test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    test_losses.append(test_loss)



    for epoch_i in range(no_epochs):
        model.train()
        running_loss = 0.0

        for idx, sample in enumerate(data_loaders.train_loader):
            inputs = sample['input']
            labels = sample['label'].unsqueeze(1)

            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = loss_function(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            
            running_loss += loss.item() * inputs.size(0)  

        avg_train_loss = running_loss / len(data_loaders.train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        model.eval()  
        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        test_losses.append(test_loss)
        
        print(f'Epoch {epoch_i+1}/{no_epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}')

    torch.save(model.state_dict(), 'saved/saved_model.pkl')


if __name__ == '__main__':
    no_epochs = 20
    train_model(no_epochs)
