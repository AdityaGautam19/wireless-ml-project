📡 Wireless AI Analyzer

Wireless AI Analyzer is a comprehensive AI-powered platform designed to analyze wireless signals, classify their modulation types, and predict their signal-to-noise ratio (SNR). The project combines deep learning, signal processing, and interactive visualization to create an end-to-end tool for researchers, engineers, and students interested in wireless communication and AI applications.

At its core, the system is built using PyTorch for modeling, Streamlit for a sleek and interactive user interface, and NumPy/Matplotlib for signal manipulation and visualization. The interface features a premium metallic wave animated background and glass-like panels, giving it a modern, high-tech look that makes exploring complex signal data engaging and intuitive.

🔹 Features
Modulation Classification: Supports 11 types of modulation including BPSK, QPSK, 8PSK, AM-DSB, AM-SSB, CPFSK, GFSK, PAM4, QAM16, QAM64, and WBFM.
SNR Prediction: Estimates the signal-to-noise ratio (SNR) for each input signal in decibels.
Interactive Visualizations: Plot the I/Q channels of signals, visualize probability distributions, and explore top-3 predicted modulations.
Random Signal Generation: Test the model with simulated signals when real data is unavailable.
Premium UI Design: Includes metallic animated wave background, glass panels for predictions and plots, and shiny metallic headers.
Flexible Input: Accepts .npy files or generates random signals with the same format for quick experimentation.

## 📸 Screenshots

Below is a screenshot of the **Wireless AI Analyzer UI**, showing the metallic wave background and signal analysis panel.

#### UI Screenshot
![UI](https://raw.githubusercontent.com/AdityaGautam19/wireless-ml-project/main/assets/screenshots/ui_screenshot.png)
🔹 Dataset

The project uses the RML2016.10a dataset available on Kaggle. It contains I/Q samples labeled with modulation type and SNR.

Format: The dataset is stored as a .pkl file containing a dictionary where keys are (modulation, snr) pairs and values are lists of I/Q signals.
Loading: Signals are loaded in the training notebook, split into training and testing sets, and converted to PyTorch tensors for model training.
Setup: The dataset can be downloaded automatically using the Kaggle API with the provided download_dataset.py script.

🔹 Training Workflow

Training is performed in the notebook/training.ipynb notebook and follows these steps:

    Data Loading: Load the .pkl dataset using Python's pickle module.
    Preprocessing: Flatten and structure the I/Q signals, encode modulation labels using LabelEncoder, and split data into training and testing sets.
    Tensor Conversion: Convert signals and labels to PyTorch tensors.
    Model Definition: Define the CNN architecture with classification and regression heads.
    Training Loop: Optimize using Adam optimizer with CrossEntropy and MSE losses combined.
    Saving: Save the trained model weights as model.pth.

📂Project Structure
####  Architecture Screenshot

![UI](https://github.com/AdityaGautam19/wireless-ml-project/blob/main/assets/screenshots/Architecture_screenshot.png)


🔹 Streamlit UI

The interactive front-end is built with Streamlit and allows users to:

Upload .npy files containing I/Q signals.
Generate random test signals.
View predicted modulation type, top-3 probabilities, and SNR for each signal.
Explore I/Q channel plots and probability bar charts.
Enjoy a premium animated metallic wave background with shiny headers and glass-like panels.

## 🎥 Demo Video

Watch the **Wireless AI Analyzer** in action:
### 🎥 Demo Video
[Watch the demo](assets/demo/demoUI.mp4)

<video width="700" controls>
  <source src="assets/demo/demoUI.mp4" type="video/mp4">
  
</video>

🔹 Installation
Clone the repository:
    git clone https://github.com/yourusername/wireless-ai-analyzer.git
    cd wireless-ai-analyzer
Install dependencies:
    pip install -r requirements.txt
Download the Kaggle dataset (requires kaggle.json API key):
    python download_dataset.py
Run the Streamlit UI:
    streamlit run app.py

🔹 Future Work

Extend to real-time signal streaming.
Add support for more modulation types and larger datasets.
Deploy as a web app for wider accessibility.
Enhance UI with multi-layer animations or signal spectrogram visualizations.

👤 Author
Aditya Gautam
Pre-final CSE student with AI specialization
GitHub: https://github.com/AdityaGautam19?tab=repositories
Email: manasgautam19@email.com
