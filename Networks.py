import torch
import torch.nn as nn
import torch.optim as optim
from Data_Loaders import Data_Loaders

class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, output_size=1):
        super(Action_Conditioned_FF, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary class

        

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.sigmoid(x)
        return x


    def evaluate(self, model, test_loader, loss_function):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input']
                labels = batch['label']
                outputs = model(inputs)  # Forward pass
                loss = loss_function(outputs, labels.unsqueeze(1))  # Compute loss
                total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
        
        average_loss = total_loss / len(test_loader.dataset)  # Average loss over dataset
        return average_loss

def main():
    model = Action_Conditioned_FF()

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)

    loss_function = nn.BCELoss() 
    test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)

    print(f'Test Loss: {test_loss}')
   
if __name__ == '__main__':
    main()
    
