# 😷 Face Mask Detection System (WebApp)

A real-time face mask detection webapp which use a lightweight TensorFlow Lite model and deployed using Streamlit. The app uses DenseNet for accurate classification and MediaPipe for face detection.

## 🚀 Features

- 🔍 Real-time face mask detection via webcam image capture or image upload
- 📂 Sample image browsing from `sample_images/`
- ⚙️ Lightweight TFLite model optimized for performance
- 🎯 Accurate predictions using DenseNet-based architecture
- 📱 User-friendly Streamlit interface
- 🧪 Clean, modular test suite with `pytest`

## 📦 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Face & Mask Detection**: [MediaPipe](https://mediapipe.dev)
- **Model Architecture**: [DenseNet](https://arxiv.org/abs/1608.06993)
- **Model Format**: [TensorFlow Lite (.tflite)](https://www.tensorflow.org/lite)
- **Testing**: `pytest`


## 🛠️ Installation

### Prerequisites (Python, packages)
- Python 3.11, due to tensorflow compatibilty issues.

### Step-by-step instructions
Follow these steps to get the app running on your local machine:

#### 1. Clone the Repository
```bash
git clone https://github.com/ateebkhan96/face-mask-detection-system.git
cd face-mask-detection-system
```
#### 2. Create a Virtual Envrionement
In order do avoid future dependencies issue, due to usage of python3.11, we create virtual envionment
```bash
py -3.11 -m venv face-mask-env
```
#### 3. Activate Virtual Environment
- On Windows 
    ```
    face-mask-env\Scripts\activate
    ```
- On Linux/Mac
    ```
    source face-mask-env\bin\activate
    ```
#### 4. Installing Dependencies
```
pip install -r requirements.txt
```
#### 5. Check Model Avability
The pre-trained TensorFlow Lite model is already included in the models directory of the GitHub repository. No additional download is required

#### 6. Run the Application
Start the Streamlit application by running
``` 
streamlit run app.py
```
#### 7. Running the Tests
In order to check if the application functions are working fine, the test_app.py file is used. These tests help ensure the system remains robust, reliable, and free of regressions during development.

To run the ```test_app.py``` file, execute (make sure virutal environment is acitve): 

```
python test_app.py
```


## 📓 Model Training Notebook

The project includes a Jupyter Notebook ([Face_Mask_Detection_using_DenseNet.ipynb](https://colab.research.google.com/drive/1KJMB7ND4v4TIaQVymTAX2Od4v8Nm4x6C?usp=sharing)) to train the DenseNet model . You can open this notebook directly on google colab (copy is already uploaded in this repo):

    1. Understand the model training process.
    2. Experiment with hyperparameters and retrain the model.

`Note: Make sure to configure the variables within the notebook before training.`