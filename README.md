Pix2Pix with U-Net and PatchGAN
This project implements an image-to-image translation model using a U-Net-based Generator and a PatchGAN Discriminator. It is trained on the Facades dataset.

📌 Features
Uses a U-Net-based Generator for high-quality image generation
Implements a PatchGAN Discriminator for adversarial training
Supports training on the Facades dataset
Random image generation from the dataset for evaluation
Trained for 10 epochs (modifiable)
📂 Dataset
The model uses the Facades dataset for training. Ensure your dataset is structured as follows:


data/facades/
│── train/    # Training images
│── val/      # Validation images
│── test/     # Test images
Modify the root path in the code to match your dataset location.

🛠 Installation
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Install dependencies:

pip install torch torchvision matplotlib pillow
Run the training script:

python your_script_name.py


🔥 Model Architecture
Generator (U-Net)
Encoder based on VGG19 features
Decoder with transposed convolutions for upsampling
Discriminator (PatchGAN)
Operates on patches instead of the whole image
Classifies real vs. fake patches in the image
🎯 Training
Uses Binary Cross-Entropy Loss (BCE) for adversarial training
Uses L1 Loss for pixel-wise reconstruction
Adam optimizer with lr=0.0001

To train the model, run:


python your_script_name.py
📸 Generating Images
After training, a random image is selected from the dataset and processed by the model.


def generate_image():
    generator.eval()
    random_index = random.randint(0, len(dataset) - 1)  
    sample_image, _ = dataset[random_index]
    sample_image = sample_image.unsqueeze(0).to(device)
    with torch.no_grad():
        fake_image = generator(sample_image)
    return sample_image.squeeze().cpu().permute(1, 2, 0), fake_image.squeeze().cpu().permute(1, 2, 0)
🖼 Output Example
Input Image	Generated Image
📌 Notes
Ensure the dataset path is correct before running the code.
The training process will save generated images at each epoch.
Modify num_epochs in the script if you want longer training.
📜 License
This project is open-source. Feel free to modify and use it for your applications.



2/2







