
# Plant Disease Identification Using Convolutional Neural Network (CNN)

This project aims to build a Plant Disease Detection system using a Convolutional Neural Network (CNN) to classify plant diseases based on images. The model utilizes a deep learning approach to detect various plant diseases by analyzing image data from the PlantVillage dataset.

## Overview

The plant disease detection system is designed to help farmers and agricultural professionals quickly identify diseases in plants, thus enabling efficient pest control and improving crop yield. By utilizing a CNN, the model can classify different plant diseases with high accuracy.

---

## Features

- **Convolutional Neural Network (CNN)**: The core model is a CNN architecture designed to process image data and classify plant diseases.
- **Image Preprocessing**: The images are resized, normalized, and augmented to ensure better training performance.
- **Multi-Class Classification**: Capable of identifying multiple plant diseases from a variety of plant species.
- **Visualization**: Training and validation accuracy/loss graphs are displayed to monitor the model's performance over time.

---

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Scikit-learn
- Matplotlib

---

## Dependencies

You can install all the required dependencies using `pip`:

```bash
pip install numpy
pip install opencv-python
pip install keras
pip install scikit-learn
pip install matplotlib
pip install tensorflow
```

---

## Dataset

The dataset used in this project is the **PlantVillage Dataset** available on Kaggle. The dataset contains images of different plant diseases, organized by plant species and disease type.

1. Visit the [PlantVillage Dataset](https://www.kaggle.com/datasets) to download the dataset.
2. Organize the images by plant species and disease for the training process.

---

## Project Workflow

1. **Image Loading**: The images are loaded, resized, and preprocessed into arrays suitable for model training.
2. **Model Architecture**: A CNN model is built with multiple convolutional layers followed by fully connected layers to classify the images.
3. **Data Augmentation**: Data augmentation techniques (rotation, shifting, shearing, zooming, etc.) are used to generate new variations of the images to improve model generalization.
4. **Model Training**: The CNN model is trained on the plant disease images.
5. **Model Evaluation**: The trained model is evaluated on a separate test set to check its accuracy.

---

## Training the Model

The model is trained using the following steps:

### Step 1: Data Preprocessing

Images are preprocessed by resizing them to a fixed size (256x256) and normalizing the pixel values between 0 and 1.

### Step 2: Model Architecture

The architecture of the model is defined as follows:

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
# Additional layers follow...
```

This model uses several convolutional layers to extract features from the images, followed by fully connected layers for classification. Dropout layers are used to prevent overfitting.

### Step 3: Model Training

The model is compiled using the Adam optimizer and binary cross-entropy loss function. Training is done for a specified number of epochs with batch size 32.

```python
history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
)
```

### Step 4: Evaluation and Visualization

The model's performance is evaluated using accuracy and loss metrics. Graphs of training and validation accuracy/loss are displayed for visualization.

```python
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
```

---

## Model Evaluation

After training, the model is evaluated on the test set to determine its accuracy. This helps ensure that the model generalizes well to unseen data.

```python
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")
```

---

## Saving the Model

The trained model is saved using pickle so that it can be used for future predictions.

```python
pickle.dump(model, open('cnn_model.pkl', 'wb'))
```

---

## Screenshots

### Training Accuracy and Loss

![Training Accuracy](https://via.placeholder.com/600x400.png?text=Training+Accuracy+Graph)

### Model Architecture

![Model Architecture](https://via.placeholder.com/600x400.png?text=Model+Architecture+Diagram)

---

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training script:

```bash
python train_model.py
```

---

## Conclusion

This project demonstrates the application of deep learning, specifically Convolutional Neural Networks, in identifying plant diseases. By training a model on the PlantVillage dataset, it achieves high accuracy in classifying images of diseased plants. The system can be expanded and deployed for real-time usage in agriculture.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Special thanks to the developers and contributors of the PlantVillage dataset.
- Thanks to the TensorFlow and Keras teams for providing powerful deep learning frameworks.
- Appreciation for the community contributions and research in plant disease identification.
