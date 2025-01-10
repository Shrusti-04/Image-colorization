import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np

# Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0002
TEST_SPLIT = 0.2

# Dataset Class
class LandscapeDataset(Dataset):
    def __init__(self, gray_dir, color_dir, transform=None):
        self.gray_images = sorted(glob.glob(f"{gray_dir}/*.jpg"))
        self.color_images = sorted(glob.glob(f"{color_dir}/*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.gray_images)

    def __getitem__(self, idx):
        gray_image = Image.open(self.gray_images[idx]).convert("RGB")
        color_image = Image.open(self.color_images[idx]).convert("RGB")
        if self.transform:
            gray_image = self.transform(gray_image)
            color_image = self.transform(color_image)
        return gray_image, color_image

# Model Definition
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)

# Training Loop with Loss Tracking
def train(generator, dataloader, optimizer, criterion, epoch):
    generator.train()
    total_loss = 0
    total_mae = 0
    num_batches = len(dataloader)
    
    for batch_idx, (gray_images, color_images) in enumerate(dataloader):
        gray_images, color_images = gray_images.to(DEVICE), color_images.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        output = generator(gray_images)
        loss = criterion(output, color_images)
        mae = nn.L1Loss()(output, color_images)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mae += mae.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}], Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.4f}, MAE: {mae.item():.4f}")
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}, Average MAE: {avg_mae:.4f}")
    return avg_loss, avg_mae

# Evaluation Function
def evaluate(generator, dataloader):
    generator.eval()
    total_loss = 0
    total_mae = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for gray_images, color_images in dataloader:
            gray_images, color_images = gray_images.to(DEVICE), color_images.to(DEVICE)
            output = generator(gray_images)
            loss = criterion(output, color_images)
            mae = nn.L1Loss()(output, color_images)
            total_loss += loss.item()
            total_mae += mae.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    return avg_loss, avg_mae

# Save Model Checkpoint
def save_checkpoint(generator, epoch):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch}.pth")

# Display Results
def display_results(generator, test_loader, epoch, num_images=10):
    generator.eval()
    plt.figure(figsize=(15, 5 * num_images))
    
    with torch.no_grad():
        # Get a batch of test images
        gray_images, color_images = next(iter(test_loader))
        gray_images = gray_images[:num_images].to(DEVICE)
        color_images = color_images[:num_images]
        predicted_images = generator(gray_images).cpu()
        
        # Create a figure with three columns (grayscale, original, predicted)
        for i in range(num_images):
            # Convert to numpy and denormalize
            gray_img = gray_images[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
            color_img = color_images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
            pred_img = predicted_images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
            
            # Plot grayscale input
            plt.subplot(num_images, 3, i*3 + 1)
            plt.imshow(gray_img)
            plt.title('Grayscale Input')
            plt.axis('off')
            
            # Plot original color image
            plt.subplot(num_images, 3, i*3 + 2)
            plt.imshow(color_img)
            plt.title('Original Color')
            plt.axis('off')
            
            # Plot predicted color image
            plt.subplot(num_images, 3, i*3 + 3)
            plt.imshow(pred_img)
            plt.title('Predicted Color')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'test_results_epoch_{epoch}.png')
    plt.close()

# Main Function
def main():
    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Dataset and DataLoader
    dataset = LandscapeDataset("landscape Images/gray", "landscape Images/color", transform=transform)
    test_size = int(len(dataset) * TEST_SPLIT)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model, Loss, and Optimizer
    generator = Generator().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)

    # Lists to store metrics
    train_losses = []
    train_maes = []
    test_losses = []
    test_maes = []

    # Training
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_mae = train(generator, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        
        # Evaluate on test set
        test_loss, test_mae = evaluate(generator, test_loader)
        test_losses.append(test_loss)
        test_maes.append(test_mae)
        
        print(f"\nTest Set - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")
        
        # Save checkpoint and display results
        save_checkpoint(generator, epoch)
        display_results(generator, test_loader, epoch)

    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('MSE Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Train MAE')
    plt.plot(test_maes, label='Test MAE')
    plt.title('Mean Absolute Error over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

    print("\nFinal Metrics:")
    print(f"Training - Final Loss: {train_losses[-1]:.4f}, Final MAE: {train_maes[-1]:.4f}")
    print(f"Testing  - Final Loss: {test_losses[-1]:.4f}, Final MAE: {test_maes[-1]:.4f}")

if __name__ == "__main__":
    main()
