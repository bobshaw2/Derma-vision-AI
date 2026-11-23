🩺 DermAssist

An Interactive AI Web Application for Skin Lesion Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Deep%20Learning-AI-purple?style=for-the-badge" />
</p><p align="center">
  <b>Turning Skin Images into AI-Powered Insights 🔬</b>
</p>---

🌟 About the Project

DermAssist is an interactive AI-powered web application developed to analyse skin lesion images using Deep Learning and Image Classification.

The application uses a trained EfficientNet-B0 model to classify uploaded dermoscopic images into seven different skin lesion categories. It provides users with a predicted category, a confidence score, and a class-wise probability distribution through a simple and interactive web interface.

The project combines Artificial Intelligence, Computer Vision, Deep Learning, and Web Development to demonstrate how machine learning can be applied to medical image analysis.

«💡 Project Goal: To develop an accessible and user-friendly AI application that demonstrates skin lesion image classification and supports educational research in AI-based healthcare applications.»

🎯 Objectives

- Develop an AI-powered application for skin lesion image classification.
- Train a deep learning model using the HAM10000 dataset.
- Implement EfficientNet-B0 for image classification.
- Build an interactive web interface using Streamlit.
- Display predicted categories and confidence scores.
- Provide a session-based analysis history.
- Explore the practical application of AI in healthcare research.

✨ Key Features

Feature| Description
🖼️ Image Upload| Upload skin lesion images in JPG, JPEG, and PNG formats.
🧠 AI Classification| Uses the trained EfficientNet-B0 model for image classification.
📊 Prediction Results| Displays the predicted skin lesion category.
🎯 Confidence Score| Shows the model's confidence score for its prediction.
📈 Probability Distribution| Displays probabilities across the seven supported categories.
📜 Analysis History| Maintains a session-based record of previous analyses.
🎨 Interactive Dashboard| Provides a clean and user-friendly Streamlit interface.
ℹ️ About AI| Provides information about the model and application.

🧬 Supported Skin Lesion Categories

The model is trained to classify images into seven categories from the HAM10000 dataset.

Code| Category| Description
AKIEC| Actinic Keratoses| Precancerous skin lesions associated with sun exposure.
BCC| Basal Cell Carcinoma| A common type of skin cancer.
BKL| Benign Keratosis-like Lesions| A group of benign keratotic skin lesions.
DF| Dermatofibroma| A generally benign skin growth.
MEL| Melanoma| A potentially serious type of skin cancer.
NV| Melanocytic Nevi| Commonly known as moles.
VASC| Vascular Lesions| Lesions associated with blood vessels.

Note: These are the seven categories supported by the trained model, not every possible skin condition.

🗂️ Dataset Information

DermAssist uses the HAM10000 (Human Against Machine with 10,000 training images) dataset.

- Dataset name: HAM10000
- Approximate image count: 10,015
- Image type: Dermoscopic images
- Number of categories: 7
- Application: Skin lesion image classification research

The dataset contains dermoscopic images collected for academic research and machine learning applications.

📚 Dataset: "HAM10000 on Kaggle" (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

⚙️ Technology Stack

Technology| Purpose
🐍 Python| Main programming language
🔥 PyTorch| Deep learning framework
🧠 EfficientNet-B0| Image classification architecture
🎨 Streamlit| Interactive web application
🖼️ PIL / Image Processing| Image loading and preprocessing
📊 Pandas| Dataset and metadata handling
💻 Visual Studio Code| Development environment

🔄 How DermAssist Works

The application follows a simple image classification workflow:

       👤 USER
          |
          v
  🖼️ Upload Skin Image
          |
          v
  ⚙️ Image Preprocessing
          |
          v
  🧠 EfficientNet-B0 Model
          |
          v
  🔍 Image Classification
          |
          v
  📊 Prediction + Confidence
          |
          v
  🎨 Display Results in App

Step-by-Step Workflow

1. Image Upload: The user uploads a supported skin lesion image.
2. Preprocessing: The image is resized, transformed, and prepared according to the model's input requirements.
3. Model Processing: The processed image is passed to the trained EfficientNet-B0 model.
4. Classification: The model calculates class probabilities and selects its predicted category.
5. Result Display: The application displays the predicted category, confidence score, and probability distribution.
6. History: The analysis can be viewed in the current application session.

🏗️ Model Architecture

DermAssist uses EfficientNet-B0, a convolutional neural network (CNN) architecture designed for efficient image classification.

EfficientNet uses a balanced scaling approach to improve model efficiency while maintaining useful image recognition capabilities.

Model Workflow

- Input: Preprocessed dermoscopic image.
- Feature Extraction: EfficientNet-B0 learns visual patterns from the image.
- Classification: The final classification layer produces scores for seven categories.
- Output: Predicted class and class-wise probability distribution.

The model uses transfer learning to build on learned image features and adapt the network for the seven-class skin lesion classification task.

💻 Installation and Setup

Follow these steps to run DermAssist locally.

1. Clone the Repository

git clone https://github.com/YOUR-USERNAME/DermAssist.git

2. Navigate to the Project Folder

cd DermAssist

3. Create a Virtual Environment

Windows:

python -m venv .venv

Activate it:

.venv\Scripts\activate

4. Install Dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

PyTorch note: Install a PyTorch build compatible with your Python version and hardware. If you want GPU acceleration, choose the appropriate CUDA-enabled build and verify that PyTorch detects your NVIDIA GPU.

5. Prepare the Dataset

Download the HAM10000 dataset from Kaggle and place the extracted images and metadata in the locations expected by your training script.

Your dataset may contain:

raw_dataset/
│
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/

6. Train the Model

If your dataset is prepared and your training script is configured:

python train.py

The training script should save the trained model to the model location expected by the application.

7. Run the Application

streamlit run app.py

The application will open in your browser using Streamlit's local server.

«⚠️ Check the file paths, dependencies, and dataset configuration in your project before running these commands.»

📁 Project Structure

DermAssist/
│
├── app.py
│   └── Main Streamlit application
│
├── model.py
│   └── Model loading and prediction functions
│
├── train.py
│   └── Model training script
│
├── requirements.txt
│   └── Project dependencies
│
├── models/
│   └── skin_model.pth
│       └── Trained model checkpoint
│
├── raw_dataset/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
│
└── README.md

The structure above is a guide based on the project files; adjust it to match your actual repository.

📸 Application Screenshots

Add your actual application screenshots here to demonstrate the interface and working features.

🏠 Dashboard

"DermAssist Dashboard" (screenshots/dashboard.png)

🔬 Skin Analysis

"DermAssist Skin Analysis" (screenshots/skin-analysis.png)

📊 Prediction Results

"DermAssist Prediction Results" (screenshots/prediction-results.png)

«Create a "screenshots" folder in your repository and add your actual screenshots using these filenames, or update the image paths above.»

🚀 Future Enhancements

Some possible improvements for future versions include:

- 📈 Further model evaluation using accuracy, precision, recall, and F1-score.
- 🧠 Integration of Explainable AI techniques such as Grad-CAM.
- 📚 Training and evaluation using more diverse datasets.
- 📱 Development of a mobile-friendly experience.
- ☁️ Cloud deployment for wider accessibility.
- 🔐 Improved data privacy and secure image handling.
- 🧪 External validation and expert evaluation.

⚠️ Limitations and Medical Disclaimer

DermAssist is an educational and research prototype. It is not a clinically validated medical diagnostic system.

- The model supports only seven categories from the HAM10000 dataset.
- It is trained on dermoscopic images; ordinary mobile photographs may not provide reliable results.
- Prediction confidence does not represent medical certainty.
- The application cannot confirm or rule out a skin disease.
- The results must not be used as a substitute for professional medical advice.

Always consult a qualified dermatologist for medical evaluation and diagnosis.

👩‍💻 Developer

<p align="center">
  <b>Preyashi Dutta</b><br/>
  B.Tech – Cybersecurity<br/>
  Individual Academic Project
</p>Project: DermAssist – An Interactive AI Web Application for Skin Lesion Analysis

Developed using: Python | PyTorch | EfficientNet-B0 | Streamlit

🙏 Acknowledgements

- The creators of the HAM10000 dataset.
- The PyTorch development community.
- The Streamlit development community.
- The researchers behind the EfficientNet architecture.

📚 References

1. P. Tschandl, C. Rosendahl, and H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," Scientific Data, vol. 5, 2018.
2. M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," International Conference on Machine Learning (ICML), 2019.
3. "PyTorch Documentation" (https://pytorch.org/docs/stable/index.html)
4. "Streamlit Documentation" (https://docs.streamlit.io/)

---

<p align="center">
  ⭐ If you find this project interesting, consider starring the repository!
</p><p align="center">
  <b>DermAssist – Exploring the Future of AI in Skin Lesion Analysis 💜</b>
</p>
