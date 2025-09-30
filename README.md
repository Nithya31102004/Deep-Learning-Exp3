# Deep-Learning-Exp3

**DL-Convolutional Deep Neural Network for Image Classification**

**AIM**

To develop a convolutional neural network (CNN) classification model for the given dataset.

**THEORY**

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

**Neural Network Model**
<img width="1005" height="565" alt="image" src="https://github.com/user-attachments/assets/6d377246-9891-4193-b8a8-5631ee71572e" />


**DESIGN STEPS**

STEP 1:Preprocess the MNIST dataset by scaling the pixel values to the range [0, 1] and converting labels to one-hot encoded format.

STEP 2:Build a convolutional neural network (CNN) model with specified architecture using TensorFlow Keras.

STEP 3:Compile the model with categorical cross-entropy loss function and the Adam optimizer.

STEP 4:Train the compiled model on the preprocessed training data for 5 epochs with a batch size of 64.

STEP 5:Evaluate the trained model's performance on the test set by plotting training/validation metrics and generating a confusion matrix and classification report. Additionally, make predictions on sample images to demonstrate model inference.

**PROGRAM**
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape

single_image= X_train[0]
single_image.shape

plt.imshow(single_image,cmap='gray')
y_train.shape

X_train.min()
X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()
X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=10,
          batch_size=128,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('7.png')

type(img)

img = image.load_img('7.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)
```

Name:NITHYA T

Register Number:2305001023


**OUTPUT**

**Training Loss per Epoch**

<img width="1027" height="188" alt="image" src="https://github.com/user-attachments/assets/0db0ffa1-30e2-4cc0-8da1-baada99133ba" />

**Confusion Matrix**

<img width="1486" height="615" alt="image" src="https://github.com/user-attachments/assets/f69d3491-c736-421e-8d36-ff66dde3776e" />


**Classification Report**

<img width="826" height="342" alt="image" src="https://github.com/user-attachments/assets/444b9426-26b3-478d-8437-8013e19b2497" />

**New Sample Data Prediction**

<img width="941" height="839" alt="image" src="https://github.com/user-attachments/assets/fe7608af-e62b-4fa5-bb93-86185ae848bf" />
<img width="892" height="856" alt="image" src="https://github.com/user-attachments/assets/29be63c1-790c-431a-adc0-18bcf9b7baa4" />
<img width="834" height="852" alt="image" src="https://github.com/user-attachments/assets/7d88d3e1-1094-4777-bbe4-9f7fd41d467b" />

**RESULT**
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed successfully.
