# import os
# import torch
# from torchvision.datasets import CIFAR10
# from torch.utils.data import DataLoader
# from torchvision import transforms
#
#
# class MLP(torch.nn.Module):
#     '''
#       Multilayer Perceptron.
#     '''
#
#     def __init__(self):
#         super().__init__()
#         self.layers = torch.nn.Sequential(
#             torch.nn.Flatten(),
#             torch.nn.Linear(32 * 32 * 3, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 32),
#             torch.nn.ReLU(),
#             torch.nn.Linear(32, 10)
#         )
#
#     def forward(self, x):
#         '''Forward pass'''
#         return self.layers(x)
#
#
# if __name__ == '__main__':
#
#     # Set fixed random number seed
#     torch.manual_seed(42)
#
#     # Prepare CIFAR-10 dataset
#     dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
#     trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
#
#     # Initialize the MLP
#     mlp = MLP()
#
#     # Define the loss function and optimizer
#     loss_function = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
#
#     # Run the training loop
#     for epoch in range(0, 5):  # 5 epochs at maximum
#
#         # Print epoch
#         print(f'Starting epoch {epoch + 1}')
#
#         # Set current loss value
#         current_loss = 0.0
#
#         # Iterate over the DataLoader for training data
#         for i, data in enumerate(trainloader, 0):
#
#             # Get inputs
#             inputs, targets = data
#
#             # Zero the gradients
#             optimizer.zero_grad()
#
#             # Perform forward pass
#             outputs = mlp(inputs)
#
#             # Compute loss
#             loss = loss_function(outputs, targets)
#
#             # Perform backward pass
#             loss.backward()
#
#             # Perform optimization
#             optimizer.step()
#
#             # Print statistics
#             current_loss += loss.item()
#             if i % 500 == 499:
#                 print('Loss after mini-batch %5d: %.3f' %
#                       (i + 1, current_loss / 500))
#                 current_loss = 0.0
#
#     # Process is complete.
#     print('Training process has finished.')