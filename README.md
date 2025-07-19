SAR-to-Optical Image Translation Web App
A Flask-based web application for predicting and visualizing color (optical) images from grayscale SAR (Synthetic Aperture Radar) images using a state-of-the-art deep learning pipeline. The solution combines the Swin Transformer and HRNet architectures with an intuitive web interface for real-time demonstration and experimentation.

Features
Upload SAR grayscale images: Quickly upload and visualize transformation results.

Swin Transformer Encoder: Extracts deep, multi-scale features from SAR input.

HRNet Decoder: Reconstructs high-fidelity, colorized optical imagery.

Side-by-side visualizations: Instantly compare the input SAR and generated optical results.

Simple authentication: Admin login interface (for demonstration only).

Dataset : https://drive.google.com/drive/folders/1dRoVXt38WogKHsQv5ACwV-40zBnsR5Zl?usp=drive_link

Installation
Requirements

Python 3.7+

PyTorch with compatible CUDA (if using a GPU)

torchvision

timm

lpips

Flask

matplotlib

numpy

tqdm

Pillow

Setup
Clone the repository and navigate to the project directory:

bash
git clone <your-repo-url>
cd <your-project-directory>
Install dependencies:

bash
pip install torch torchvision timm lpips flask matplotlib tqdm pillow
Prepare model weights:

Place your trained model weights file at: model/model.pt

Ensure folder structure:

Make sure you have a static/ directory for uploaded/test images.

Organize your paired SAR and optical images for any model retraining.

Usage
Run the Flask server:

bash
python your_app_file.py
Access the web interface:

Open your browser and navigate to http://127.0.0.1:5000

Authenticate (demo admin):

Username: admin

Password: admin

Predict SAR-to-Optical translation:

Go to the “Predict” page.

Upload your .jpg or .png grayscale SAR image.

View the original and translated images side by side.

Code Structure
PairedImageDataset: Loads paired SAR and optical images with filename matching and preprocessing.

SwinHRNetModel: Main PyTorch model combining Swin Transformer encoder and HRNet-inspired decoder.

Flask app: Web interface for upload, authentication, and result visualization.

Visualization: Results are shown as an inline PNG in the app, no need for manual file management.

Security Notes
The included admin credentials are for demonstration only. For deployment, implement secure user authentication.

Uploaded files are temporarily stored in static/ and overwritten on new uploads.

Tips and Extensions
Training/Inference: Adapt the dataset and model classes for your data, retrain as needed.

Model Customization: Plug in different encoder/decoder architectures as desired.

Deployment: For real-world use, add error handling, input validation, HTTPS, and robust authentication.

Example Workflow
Login as admin.

Select and upload a SAR image on the Predict page.

Instantly view the translated color image alongside your original upload.

(Optional) Use your dataset and model for retraining or further fine-tuning.

License
This project is intended for research and demonstration purposes. Please refer to the source repository for licensing details.

Acknowledgements
Swin Transformer backbone via timm

HRNet design inspiration from published literature

Perceptual loss computation with lpips

Contact
