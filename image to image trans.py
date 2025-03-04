import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import random

# ✅ Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Dataset Class (Keeping the same as previous)
class FacadesDataset(Dataset):
    def __init__(self, root, mode="train", transform=None):
        self.transform = transform
        self.image_dir = os.path.join(root, mode)
        self.image_dir = os.path.normpath(self.image_dir)  # Normalize path
        self.image_filenames = sorted(os.listdir(self.image_dir))
    
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        input_image = image.crop((0, 0, width // 2, height))
        target_image = image.crop((width // 2, 0, width, height))
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        return input_image, target_image

# ✅ Data Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Use Absolute Path
dataset = FacadesDataset(root=r"C:\\Users\\Mohammed Haris\\OneDrive\\Desktop\\progidy\\image to image translation\\data\\facades", mode="train", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ✅ U-Net Generator
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.encoder = models.vgg19(pretrained=True).features[:23]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ✅ PatchGAN Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x, y):
        return self.model(torch.cat((x, y), dim=1))

# ✅ Initialize Weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

generator = UNetGenerator().to(device)
discriminator = PatchDiscriminator().to(device)
initialize_weights(generator)
initialize_weights(discriminator)

# ✅ Loss and Optimizers
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# ✅ Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, (input_image, target_image) in enumerate(dataloader):
        input_image, target_image = input_image.to(device), target_image.to(device)

        # Train Generator
        gen_optimizer.zero_grad()
        fake_image = generator(input_image)
        fake_pred = discriminator(input_image, fake_image)
        gen_loss = adversarial_loss(fake_pred, torch.ones_like(fake_pred)) + l1_loss(fake_image, target_image)
        gen_loss.backward()
        gen_optimizer.step()

        # Train Discriminator
        disc_optimizer.zero_grad()
        real_pred = discriminator(input_image, target_image)
        fake_pred = discriminator(input_image, fake_image.detach())
        disc_loss = adversarial_loss(real_pred, torch.ones_like(real_pred)) + adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        disc_loss.backward()
        disc_optimizer.step()

        print(f"Epoch {epoch+1}, Step {i+1}, Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")
    
    # Save generated images every epoch
    vutils.save_image(fake_image, f"generated_epoch_{epoch+1}.png", normalize=True)

# ✅ Generate and Display Random Image
def generate_image():
    generator.eval()
    random_index = random.randint(0, len(dataset) - 1)  # Select a random image
    sample_image, _ = dataset[random_index]
    sample_image = sample_image.unsqueeze(0).to(device)
    with torch.no_grad():
        fake_image = generator(sample_image)
    fake_image_cpu = fake_image.squeeze().cpu().permute(1, 2, 0)
    return sample_image.squeeze().cpu().permute(1, 2, 0), fake_image_cpu

sample_image, generated_image = generate_image()

plt.subplot(1, 2, 1)
plt.imshow(sample_image.numpy())
plt.title("Input Image")

plt.subplot(1, 2, 2)
plt.imshow(generated_image.numpy())
plt.title("Generated Image")

plt.show()
