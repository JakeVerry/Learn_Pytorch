"""
Contains the functionality for training a PyTorch image classification model.
"""

import os
import torch

from torchvision import transforms
from timeit import default_timer as timer

import data_setup, engine, model_setup, utils


# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "C:\\Users\\jakev\\Code\\Learn_Pytorch\\Practice\\data\\train"
test_dir = "C:\\Users\\jakev\\Code\\Learn_Pytorch\\Practice\\data\\test"

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create transforms
data_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# Create DataLoaders and get class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=data_transforms,
                                                                               batch_size=BATCH_SIZE)

# Create model
model = model_setup.TinyVGG(input_shape=3, 
                            hidden_units=HIDDEN_UNITS, 
                            output_shape=len(class_names)).to(device)

# Set up loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start timer
start_time = timer()

# Start training with engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=NUM_EPOCHS,
             device=device)

# End timer 
end_time = timer()
total_time = end_time - start_time
print(f"[INFO] Total training time: {total_time:.3f} seconds")

utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pt")
