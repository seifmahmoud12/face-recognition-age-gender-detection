import os
import cv2
import itertools
import calendar
import mysql.connector
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.models import load_model
import pandas as pd
from datetime import date
from datetime import datetime, timedelta
now = datetime.now()
hrs = now.hour;mins = now.minute;secs = now.second;
zero = timedelta(seconds = secs+mins*60+hrs*3600)
st = now - zero # this take me to 0 hours.
time1 = st + timedelta(seconds=15*3600+15*60) # this gives 10:30 AM
time2 = st + timedelta(seconds=16*3600+50*60)  # this gives 4:30 PM
current_time = now.strftime("%H:%M:%S")
days=date.today()
# db = mysql.connector.connect(
#         host="localhost",
#         user="root",
#         passwd="123456",
#     database="attendences"
#
#     )
def detect_face(img):
    img = img[70:195,78:172]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (50, 50))
    return img
# def print_progress(val, val_len, folder, bar_size=20):
#     progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
#     if val == 0:
#         print("", end = "\n")
#     else:
#         print("[%s] (%d samples)\t label : %s \t\t" % (progr, val+1, folder), end="\r")

# #
dataset_folder = "data/"
# #
names = []
images = []
for folder in os.listdir(dataset_folder):
    files = os.listdir(os.path.join(dataset_folder, folder))[:150]
    if len(files) < 80:
        continue
    for i, name in enumerate(files):
        if name.find(".jpg") > -1 :
            img = cv2.imread(os.path.join(dataset_folder + folder, name))
            img = detect_face(img) # detect face using mtcnn and crop to 100x100
            if img is not None :
                images.append(img)
                names.append(folder)
#
#print_progress(i, len(files), folder)
#
#
def img_augmentation(img):
    h, w = img.shape
    center = (w // 2, h // 2)
    M_rot_5 = cv2.getRotationMatrix2D(center, 5, 1.0)
    M_rot_neg_5 = cv2.getRotationMatrix2D(center, -5, 1.0)
    M_rot_10 = cv2.getRotationMatrix2D(center, 10, 1.0)
    M_rot_neg_10 = cv2.getRotationMatrix2D(center, -10, 1.0)
    M_trans_3 = np.float32([[1, 0, 3], [0, 1, 0]])
    M_trans_neg_3 = np.float32([[1, 0, -3], [0, 1, 0]])
    M_trans_6 = np.float32([[1, 0, 6], [0, 1, 0]])
    M_trans_neg_6 = np.float32([[1, 0, -6], [0, 1, 0]])
    M_trans_y3 = np.float32([[1, 0, 0], [0, 1, 3]])
    M_trans_neg_y3 = np.float32([[1, 0, 0], [0, 1, -3]])
    M_trans_y6 = np.float32([[1, 0, 0], [0, 1, 6]])
    M_trans_neg_y6 = np.float32([[1, 0, 0], [0, 1, -6]])

    imgs = []
    imgs.append(cv2.warpAffine(img, M_rot_5, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_rot_neg_5, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_rot_10, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_rot_neg_10, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_3, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_3, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_6, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_6, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_y3, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_y3, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_y6, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_y6, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.add(img, 10))
    imgs.append(cv2.add(img, 30))
    imgs.append(cv2.add(img, -10))
    imgs.append(cv2.add(img, -30))
    imgs.append(cv2.add(img, 15))
    imgs.append(cv2.add(img, 45))
    imgs.append(cv2.add(img, -15))
    imgs.append(cv2.add(img, -45))

    return imgs
#
#
    #plt.imshow(images, cmap="gray")
# # # #
img_test = images[1]

augmented_image_test = img_augmentation(img_test)

plt.figure(figsize=(15,10))
for i, img in enumerate(augmented_image_test):
    plt.subplot(4,5,i+1)
    plt.imshow(img, cmap="gray")
#plt.show()
# # # #
augmented_images = []
augmented_names = []
for i, img in enumerate(images):
    try :
        augmented_images.extend(img_augmentation(img))
        augmented_names.extend([names[i]] * 20)
    except :
         print(i)
images.extend(augmented_images)
names.extend(augmented_names)
#print(len(images), len(names))

unique, counts = np.unique(names, return_counts = True)
for item in zip(unique, counts):
   print(item)
plt.imshow(images[17], cmap="gray")
#plt.show()
# def print_data(label_distr, label_name):
#     plt.figure(figsize=(12, 6))
#
#     my_circle = plt.Circle((0, 0), 0.7, color='white')
#     plt.pie(label_distr, labels=label_name, autopct='%1.1f%%')
#     plt.gcf().gca().add_artist(my_circle)
#     plt.show()
#print(len(augmented_images), len(augmented_names))
# # #
unique = np.unique(names)
label_distr = {i: names.count(i) for i in names}.values()
#print_data(label_distr, unique)
n =3150
#
def randc(labels, l):
    return np.random.choice(np.where(np.array(labels) == l)[0], n, replace=False)

mask = np.hstack([randc(names, l) for l in np.unique(names)])

names = [names[m] for m in mask]
images = [images[m] for m in mask]

#
label_distr = {i:names.count(i) for i in names}.values()
#print_data(label_distr, unique)
#
le = LabelEncoder()

le.fit(names)

labels = le.classes_

name_vec = le.transform(names)

categorical_name_vec = to_categorical(name_vec)

#
# print("number of class :", len(labels))
# print(labels)
#
# print(name_vec)
#
#
# print(categorical_name_vec)
#
# x_train, x_test, y_train, y_test = train_test_split(np.array(images, dtype=np.float32),   # input data
#                                                     np.array(categorical_name_vec),       # target/output data
#                                                     test_size=0.20,
#                                                     random_state=42)
#
#
# print(x_train.shape, y_train.shape, x_test.shape,  y_test.shape)
#
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# x_train.shape, x_test.shape
#
#
# def cnn_model(input_shape):
#     model = Sequential()
#
#     model.add(Conv2D(64,
#                      (3, 3),
#                      padding="valid",
#                      activation="relu",
#                      input_shape=input_shape))
#     model.add(Conv2D(64,
#                      (3, 3),
#                      padding="valid",
#                      activation="relu",
#                      input_shape=input_shape))
#
#     model.add(MaxPool2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(128,
#                      (3, 3),
#                      padding="valid",
#                      activation="relu"))
#     model.add(Conv2D(128,
#                      (3, 3),
#                      padding="valid",
#                      activation="relu"))
#     model.add(MaxPool2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#
#     model.add(Dense(128, activation="relu"))
#     model.add(Dense(64, activation="relu"))
#     model.add(Dense(len(labels)))  # equal to number of classes
#     model.add(Activation("softmax"))
#
#     model.summary()
#
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
#
#
# input_shape = x_train[1].shape
# #
# EPOCHS = 20
# BATCH_SIZE = 32
#
# model = cnn_model(input_shape)
#
# history = model.fit(x_train,
#                     y_train,
#                     epochs=EPOCHS,
#                     batch_size=BATCH_SIZE,
#                     shuffle=True,
#                     validation_split=0.25   # 25% of train dataset will be used as validation set
#                     )
# # test_loss, test_acc = model.evaluate(x_test, y_test)
# print("Test loss:", test_loss)
# print("Test accuracy:", test_acc)
#
# # Generate predictions for test set
# y_pred=model.predict(x_test)
#
# # Convert predictions from one-hot encoding to class labels
# y_pred_labels = [labels[np.argmax(pred)] for pred in y_pred]
# y_test_labels = [labels[np.argmax(true)] for true in y_test]
#
# # Print classification report and confusion matrix
# print(classification_report(y_test_labels, y_pred_labels))
# cm = confusion_matrix(y_test_labels, y_pred_labels)
# print("Confusion matrix:\n", cm)
#
# # Plot training and validation loss over epochs
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.show()
#
# # Plot training and validation accuracy over epochs
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='lower right')
# plt.show()
#
# # Plot confusion matrix
# plt.imshow(cm, cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.xticks(range(len(labels)), labels, rotation=90)
# plt.yticks(range(len(labels)), labels)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.show()

# model.save("model-cnn-facerecognition.h5")
# # predict test data
#y_pred=model.predict(x_test)




def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),
                  (max(xt, x0 + w), yt),
                  color,
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),
                  (x0 + w, y0 + baseline),
                  color,
                  -1)
    cv2.putText(img,
                label,
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA)
    return img


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# --------- load Keras CNN model -------------
model = load_model("model-cnn-facerecognition.h5")
print("[INFO] finish load model...")



cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in faces:

            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (50, 50))
            face_img = face_img.reshape(1, 50, 50, 1)

            result = model.predict(face_img)
            idx = result.argmax(axis=1)
            confidence = result.max(axis=1) * 100


            def takearrivalAttendence(name):
                with open('attendancearrival.csv', 'r+') as f:
                    mypeople_list = f.readlines()
                    nameList = []

                    for line in mypeople_list:
                        entry = line.split(',')
                        nameList.append(entry[0])
                        now=datetime.now()

                    if name not in nameList:
                        now = datetime.now()
                        day = calendar.day_name[days.weekday()]
                        # now.strftime("%d/%m/%Y %H:%M:%S")
                        date=now.strftime("%Y-%m-%d")
                        timestring = now.strftime("%H:%M:%S")
                        f.writelines(f'\n{name},{timestring},{date},{day}')



            def takedepartureAttendence(name):
                with open('attendnancedeparture.csv', 'r+') as f:
                    mypeople_list = f.readlines()
                    nameList = []
                    for line in mypeople_list:
                        entry = line.split(',')
                        nameList.append(entry[0])
                        now = datetime.now()
                    if name not in nameList and time1 <=now< time2 :
                        now = datetime.now()
                        day = calendar.day_name[days.weekday()]
                        # now.strftime("%d/%m/%Y %H:%M:%S")
                        date = now.strftime("%Y-%m-%d")
                        timestring = now.strftime("%H:%M:%S")
                        f.writelines(f'\n{name},{timestring},{date},{day}')
            if confidence > 80:

                label_text ="%s" % (labels[idx])
                takearrivalAttendence(label_text)
                takedepartureAttendence(label_text)





            else:
                label_text = "N/A"
            frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0, 255, 255), text_color=(50, 50, 50))






        cv2.imshow('Detect Face', frame)

    else:
        break
    if cv2.waitKey(2) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()