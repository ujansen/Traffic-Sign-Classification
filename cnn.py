import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, BatchNormalization

image_data = []
labels = []
CLASSES = 43
current_path = os.getcwd()

for image_num in range(CLASSES):
    filepath = os.path.join(current_path, 'Train', str(image_num))
    images = os.listdir(filepath)

    for image in images:
        try:
            pic = Image.open(filepath + '\\' + image)
            pic = pic.resize((30, 30))
            pic = np.array(pic)
            image_data.append(pic)
            labels.append(image_num)
        except:
            print('Could not load image.')
            
image_data = np.array(image_data)
labels = np.array(labels)
print (f'Training data dimensions: {image_data.shape}\nNo. of images: {labels.shape}')

X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size = 0.2, random_state = 0)
y_train = to_categorical(y_train, CLASSES)
y_test = to_categorical(y_test, CLASSES)

classifier = Sequential()
classifier.add(Conv2D(filters = 32, kernel_size = (4, 4), activation = 'relu', input_shape = X_train.shape[1: ]))
classifier.add(Conv2D(filters = 32, kernel_size = (4, 4), activation = 'relu'))
classifier.add(MaxPool2D(pool_size = (2, 2)))
classifier.add(BatchNormalization(axis = 3, momentum = 0.8))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(filters = 64, kernel_size = (4, 4), activation = 'relu'))
classifier.add(Conv2D(filters = 64, kernel_size = (4, 4), activation = 'relu'))
classifier.add(MaxPool2D(pool_size = (2, 2)))
classifier.add(BatchNormalization(axis = 3, momentum = 0.8))
classifier.add(Dropout(0.2))
classifier.add(Flatten())
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = CLASSES, activation = 'softmax'))

classifier.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

classifier.summary()

EPOCHS = 20
classifier.fit(X_train, y_train, batch_size = 64, epochs = EPOCHS, validation_data = (X_test, y_test))

from sklearn.metrics import accuracy_score
import pandas as pd
y_test = pd.read_csv('Test.csv')

actual_labels = y_test['ClassId'].values
images = y_test['Path'].values

test_image_data = []
for image in images:
    pic = Image.open(image)
    pic = pic.resize((30, 30))
    test_image_data.append(np.array(pic))
    
X_test = np.array(test_image_data)

prediction = classifier.predict_classes(X_test)

print(f'Accuracy: {accuracy_score(actual_labels, prediction)}')

classifier.save('traffic_sign_classifier.h5')
    
    