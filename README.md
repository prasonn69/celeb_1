# celeb_1
# 🧠 Celebrity Image Classifier & Predictor

A simple CNN-based image classifier built using TensorFlow and deployed via Streamlit. This app lets you train a model on a custom dataset of sports celebrities and then predict the celebrity in an uploaded image.

---

## 🚀 Features

* Load and preprocess a dataset of celebrity images
* Train a Convolutional Neural Network (CNN) from scratch
* Validate performance with accuracy and loss metrics
* Save and load trained model
* Upload an image through a Streamlit interface to get predictions

---

## 🧰 Requirements

* Python 3.x
* TensorFlow
* Streamlit
* NumPy
* PIL (Pillow)

You can install all dependencies with:

```bash
pip install tensorflow streamlit numpy pillow
```

---

## 📁 Project Structure

```
├── app.py                 # Main Python file
├── celebrity_model.h5     # Saved model after training
├── README.md              # Project documentation
└── /Sports-celebrity images/
     ├── class1/
     ├── class2/
     └── ...
```

> **Note:** The dataset folder must be structured like this, with each class having its own subfolder of images.

---

## 🏃‍♂️ How to Run

### Step 1: Prepare Your Dataset

Organize images like this:

```
/Sports-celebrity images/
   ├── Messi/
   ├── Ronaldo/
   ├── Serena/
   └── ...
```

Each subfolder is treated as a class label.

---

### Step 2: Run the App

```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

A simple CNN model with the following layers:

* Rescaling
* Conv2D (32 filters) + MaxPooling
* Conv2D (64 filters) + MaxPooling
* Flatten
* Dense (128 units, ReLU)
* Output layer (Softmax over number of classes)

---

## 📊 Output

After training, you'll see:

* Model Accuracy and Loss
* A user-friendly image upload section
* Predicted celebrity name based on uploaded image

---

## 📝 Notes

* Model is trained for 10 epochs on 80% of the dataset
* Validation is done on the remaining 20%
* Images are resized to **224x224**
* Label mode is **categorical** (for multi-class classification)

---

## 🔮 To-Do

* [ ] Add real-time confidence scores
* [ ] Support drag-and-drop uploads
* [ ] Convert to TensorFlow Lite for mobile use
* [ ] Enhance model with data augmentation

---

## 👨‍💻 Author

Made with ❤️ by Prason

