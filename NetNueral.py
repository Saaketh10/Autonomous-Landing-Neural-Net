import torch
import torch.nn as nn
import torch.optim as optim


#This is a sample data set I created. We can replace this with actual information.
#Torch.Tensor is a multi-dimensional matrix which contains elements of a single data type, ex: floats,integers,complex numbers, etc. 
#Here we are using a float
X = torch.tensor([
    [[1, 2, 2],
     [3, 4, 5],
     [6, 7, 8]],
    
    [[3, 1, 2],
     [2, 5, 5],
     [8, 9, 7]],
    
    [[2, 2, 2],
     [1, 2, 3],
     [4, 5, 6]],
], dtype=torch.float32)


#Possible output values
Y = torch.tensor([0 ,1, 1], dtype=torch.float32)


#Define Nueral Network
#self.fc means fully connected layer, which is defined below
#ReLU activation for the hidden layer (Replaces negative values with 0, and leaves positive)(Activation Function)
#Sigmoid activation for the output layer (Gives us a 0 or 1 value)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init()
        self.fc = nn.Sequential(
            nn.Flatten(), #This layer is used to flatten the input data, converting the 3x3 matrix into a flat vector of 9 elements
            nn.Linear(9, 16),  # Fully connected layer with 9 input features (3x3 matrix), and 16 output features using matrix multipiclation
            nn.ReLU(), #Adds Non-Linearity 
            nn.Linear(16, 1),  # 16 input features given and 1 output feature. 
            nn.Sigmoid()  # Final Layer, and maps output to value 0 or 1. Sigmoid is for binary classification
        )
    #This defines the forward pass of the nueral network, determing how data is processed through layers of the model
    def forward(self, x):
        return self.fc(x)
    
#Nueral Network
model = NeuralNet()


#criterion calculates the loss during training, and optimzer adjusts to minimize loss 
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification//// Also Known as Log Loss equating to high accuracy values///Look up Formuala
optimizer = optim.Adam(model.parameters(), lr=0.001) #Adam is an optimizng algorithim which updates the weights of the nueral network
#lr is the learning rate, and set slower to combat risk of overshooting for the solution and gradual training



#Trains the Model
#Epoch is number of iterations over the entire data set
epochs = 1000
for epoch in range(epochs): #This loop iterates through each epoch, improving the model
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(X)  # Forward pass input X through the Nueral Net
    loss = criterion(outputs, Y)  # Compute the loss between predicted outputs and true label (Y)
    loss.backward()  # Backpropagation, computes gradient loss and adjusts the weights and biases to minimize loss
    optimizer.step()  # Update the weights using the gradients

#Checks if current epoch is multiple of 100, and prints epoch number and current loss. This lets us monitor the training progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')







#Evaluating Trained Nueral Network
model.eval() #Sets model in Evaluation Mode. Pytorch has 2 modes, training and evaluating. Evulation mode does not update weights and gradients like the training, but used to infer and make predicitions
with torch.no_grad(): #Disables gradient computation bc were not updating the model
    test_input = torch.tensor([[[1, 1, 2], #This tests an example input to evulate with newly trained model
                                [3, 4, 4],
                                [5, 6, 7]]], dtype=torch.float32)
    prediction = model(test_input) #Passes the test input through the Nueral Netowrk
    if prediction.item() > 0.5: #Checks Prediciton Accurary. Predicted Value is interpreted as a probablity, so we use 0.5 to make the binary classification
        print("(output 1)")  
    else:
        print("(output 0)")


#0 means adjacent values do not match, 1 means they all match