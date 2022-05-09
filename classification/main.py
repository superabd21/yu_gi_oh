import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
import keras
from keras.preprocessing.image import array_to_img
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard
from time import strftime
import itertools

# Constants
DATA_YUGI = "C:/Users/abod2/pycharmProjects/ML/Monster/cards2"
CATGEORIES = ["Monster", "Spell Card", "Trap Card"]
IMAGE_SIZE = 130
ELEMENTS_Size = 1745 * 3
TOTAL_INPUT = IMAGE_SIZE * IMAGE_SIZE * 3
LOG_DIR = 'tensorboard_yugi_logs/'

x, y = [], []


# create the data and it has been taken randomly from each cat.
def create_data():
    for category in CATGEORIES:
        path = os.path.join(DATA_YUGI, category)
        label = CATGEORIES.index(category)
        random_path_list = random.sample(os.listdir(path), 1745)
        for img in random_path_list:
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
                x.append(resized_array)
                y.append(label)
            except Exception as e:
                pass


create_data()
# shuffle the data to prevent Bias
x = np.asarray(x)
y = np.asarray(y)
x, y = shuffle(x, y, random_state=0)

print(x.shape)

# create the train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# just small test to see the image
print(X_train.shape)
plt.imshow(X_train[100])
plt.xlabel(CATGEORIES[y_train[100]], fontsize=50)
plt.show()

# store some properties for later
nr_images, x, y, c = X_train.shape
print(f"images={nr_images} \t| width={x} \t| height={y} \t| channels={c}")
print(X_train[0][0][0][0])

# make the number in numpy smaller and in float , 255 refer to rgb
# smaller number is better for learning Rate and it will help us when
# we're calculating the loss and weights
X_train, X_test = X_train / 255.0, X_test / 255.0

# show ten images with their labels
plt.figure(figsize=(30, 5))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(CATGEORIES[y_train[i]], fontsize=15)
    plt.imshow(X_train[i])
    plt.show()

# until now, we are dealing with four Dims , so it will be easier to make in single Vector (flattend)
X_train = X_train.reshape(X_train.shape[0], IMAGE_SIZE * IMAGE_SIZE * 3)
X_test = X_test.reshape(X_test.shape[0], IMAGE_SIZE * IMAGE_SIZE * 3)
print(f"Shape of X_test is {X_test.shape} and Shap of X_train is {X_train.shape}")

# Create a Validation DataSet to select best  model
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.22, random_state=42)
print(X_val.shape)


# Tensorboard (Viz)
def get_tensorboard(model_name):
    folder_name = f'{model_name} at {strftime("%H %M")}'
    dir_paths = os.path.join(LOG_DIR, folder_name)
    try:
        os.makedirs(dir_paths)
    except OSError as err:
        print(err.strerror)
    else:
        print("Successfully created")

    return TensorBoard(log_dir=dir_paths)



# model_1 = Sequential([
#     Dense(units=128, input_dim=TOTAL_INPUT, activation="relu", name='m1_hidden1'),
#     Dense(units=64, activation="relu", name='m1_hidden2'),
#     Dense(units=16, activation="relu", name='m1_hidden3'),
#     Dense(units=3, activation="softmax", name='m1_output')
#
# ])
# model_1.compile(optimizer="adam",
#                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#
# print(model_1.summary())

# Fit the Model and we have to pass our data in model many times using epochs.
# we split up the data and process one piece at the time using batch.
sample_per_batch = 5
nr_epochs = 4
print(X_train.shape[0])
print(X_test.shape[0])
print(X_val.shape[0])
print(f"the number of iteration it need for epoch is :{X_train.shape[0] / sample_per_batch}")

# Neural Network with Keras
model_3 = Sequential()

model_3.add(Dropout(0.2, seed=42, input_shape=(TOTAL_INPUT,)))
model_3.add(Dense(128, activation="relu", name="m3_hidden1"))
model_3.add(Dropout(0.25, seed=42))
model_3.add(Dense(64, activation="relu", name="m3_hidden2"))
model_3.add(Dense(16, activation="relu", name="m3_hidden3"))
model_3.add(Dense(3, activation="softmax", name="m3_output"))

model_3.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model_3.fit(X_train, y_train, batch_size=sample_per_batch, epochs=nr_epochs,
            callbacks=[get_tensorboard("Model 3")], validation_data=(X_val, y_val))

# just small check on the data
for num in range(700):
    test_img = np.expand_dims(X_val[num], axis=0)
    predicted_val = model_3.predict(test_img)
    predicted_class = np.argmax(predicted_val, axis=1)
    if predicted_class[0] != y_val[num]:
        (print(f"not correct, Actual Value is :{y_val[num]} vs predicted:{predicted_class[0]}"))
print("every thing else sounds ok")

test_loss, test_accuarcy = model_3.evaluate(X_test, y_test)
print(f"Test loss is : {test_loss:0.3} and test accuracy is : {test_accuarcy:0.1%}")

predictions = np.argmax(model_3.predict(X_test), axis=1)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predictions)

nr_rows = conf_matrix.shape[0]
nr_cols = conf_matrix.shape[1]

plt.figure(figsize=(7, 7), dpi=150)
plt.imshow(conf_matrix, cmap=plt.cm.YlOrBr)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("predicted label", fontsize=12)
plt.ylabel("Actual label", fontsize=12)
ticks_marks = np.arange(3)
plt.xticks(ticks_marks, CATGEORIES)
plt.yticks(ticks_marks, CATGEORIES)
plt.colorbar()
for i, j in itertools.product(range(nr_rows), range(nr_cols)):
    plt.text(j, i, conf_matrix[i, j], horizontalalignment="center",
             color="white" if conf_matrix[i, j] > 500 else "black", fontsize=18)
plt.show()

recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
print(recall)
precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
print(precision)
avg_recall = np.mean(recall)
print(f"Model 3 recall score is {avg_recall:.2%}")
avg_precision = np.mean(precision)
print(f"Model 3 precision score is {avg_recall:.2%}")

f_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
print(f"Model 3 score is {f_score:.2%}")

model_2 = Sequential()

model_2.add(Dropout(0.3, seed=42, input_shape=(TOTAL_INPUT,)))
model_2.add(Dense(50, activation="relu", name="m2_hidden1"))
model_2.add(Dense(16, activation="relu", name="m2_hidden2"))
# model_2.add(Dense(10, activation="relu", name="m2_hidden3"))
model_2.add(Dense(3, activation="softmax", name="m2_output"))

model_2.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model_2.fit(X_train, y_train, batch_size=sample_per_batch, epochs=nr_epochs,
            callbacks=[get_tensorboard("Model 2")], validation_data=(X_val, y_val))
print("model 2")
for num in range(700):
    test_img = np.expand_dims(X_val[num], axis=0)
    predicted_val = model_2.predict(test_img)
    predicted_class = np.argmax(predicted_val, axis=1)
    if predicted_class[0] != y_val[num]:
        (print(f"not correct, Actual Value is :{y_val[num]} vs predicted:{predicted_class[0]}"))

print("every thing sounds ok")

test_loss_2, test_accuarcy_2 = model_2.evaluate(X_test, y_test)
print(f"Test loss is : {test_loss_2:0.3} and test accuracy is : {test_accuarcy_2:0.1%}")

predictions_2 = np.argmax(model_2.predict(X_test), axis=1)
conf_matrix_2 = confusion_matrix(y_true=y_test, y_pred=predictions_2)

plt.figure(figsize=(7, 7), dpi=150)
plt.imshow(conf_matrix_2, cmap=plt.cm.YlOrBr)
plt.title("Confusion Matrix 2", fontsize=16)
plt.xlabel("predicted label", fontsize=12)
plt.ylabel("Actual label", fontsize=12)
ticks_marks = np.arange(3)
plt.xticks(ticks_marks, CATGEORIES)
plt.yticks(ticks_marks, CATGEORIES)
plt.colorbar()
for i, j in itertools.product(range(nr_rows), range(nr_cols)):
    plt.text(j, i, conf_matrix_2[i, j], horizontalalignment="center",
             color="white" if conf_matrix_2[i, j] > 500 else "black", fontsize=18)
plt.show()
