#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import glob
import csv
import sys
import shutil
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import  Conv1D, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt
import keras
import keras.callbacks as KC
from keras import regularizers
from keras.engine.topology import Input
from keras.preprocessing.image  import ImageDataGenerator
from history_checkpoint_callback import HistoryCheckpoint, TargetHistory

train_dir = 'clean_organized_images/train/'
valid_dir = 'clean_organized_images/valid/'
test_dir = 'clean_organized_images/test/'
SIZE = 128
TRAIN_RATIO = 0.8
RESHAPED = 0
NB_CLASSES = 2
OPTIMIZER = SGD()
BATCH_SIZE = 174 #訓練データ数
NB_EPOCH = 50
VALIDATION_SPLIT = 0.4
VERBOSE = 1
COLOR_MODE = 0
USE_DATAGEN = False
LR=0.001
DROPOUT=0.5

if COLOR_MODE == 1: #color
    FILTERS = 16
else: #gray
    FILTERS = 24

if USE_DATAGEN:
    aug_str = 'with_aug'
else:
    aug_str = 'without_aug'

if COLOR_MODE == 0:
    color_mode = 'gray'
else:
    color_mode = 'color'
    
SAVED_MODEL_PATH = 'models/cat_dog_classification_{}_{}_organized_model.hdf5'.format(aug_str, color_mode)
END_SAVED_MODEL_PATH = 'models/cat_dog_classification_end_{}_{}_organized_model.hdf5'.format(aug_str, color_mode)
IMG_PATH = 'chart/{}_{}_organized.png'.format(aug_str, color_mode)

ERROR_TRAIN = 'error_images/train/'
ERROR_VALID = 'error_images/valid/'
ERROR_TEST = 'error_images/test/'

def create_images_answers(dir_path, filename=False):
    files = glob.glob(os.path.join(dir_path, '*.png'))
    files.sort()
    images = [resize_for_model(cv2.imread(file, COLOR_MODE)) for file in files]
    if filename == False:
        return images
    else:
        return images, files

def prepare_data():
    # 猫(0)と犬(1)の画像を取得してフラグを追加したにシャッフル加工してデータとして返す。
    
    train_dog_images, train_dog_filenames = create_images_answers(train_dir + 'dog/', filename=True)
    train_cat_images, train_cat_filenames = create_images_answers(train_dir + 'cat/', filename=True)
    valid_dog_images, valid_dog_filenames = create_images_answers(valid_dir + 'dog/', filename=True)
    valid_cat_images, valid_cat_filenames = create_images_answers(valid_dir + 'cat/', filename=True)
    test_dog_images, test_dog_filenames = create_images_answers(test_dir + 'dog/', filename=True)
    test_cat_images, test_cat_filenames = create_images_answers(test_dir + 'cat/', filename=True)

    train_images = np.array(train_dog_images + train_cat_images)
    valid_images = np.array(valid_dog_images + valid_cat_images)
    test_images = np.array(test_dog_images + test_cat_images)
    train_filenames = train_dog_filenames + train_cat_filenames
    valid_filenames = valid_dog_filenames + valid_cat_filenames
    test_filenames = test_dog_filenames + test_cat_filenames

    train_answers = np.array([0] * len(train_dog_images) + [1] * len(train_cat_images))
    valid_answers = np.array([0] * len(valid_dog_images) + [1] * len(valid_cat_images))
    test_answers = np.array([0] * len(test_dog_images) + [1] * len(test_cat_images))

    return (train_images, train_answers, train_filenames), (valid_images, valid_answers, valid_filenames), (test_images, test_answers, test_filenames)

def resize_for_model(image):
    # np形式のimageを特定の大きさにresizeする。
    return cv2.resize(image, (SIZE, SIZE))

def remove_log_files(dir):
    files = glob.glob(os.path.join(dir, '*.local'))
    for file in files:
        os.remove(file)

def predict_classes_for_functional(model, test_data):
    #Functional APIを使ったモデルはpredict_classesメソッドが無いので代用
    return np.argmax(model.predict(test_data), axis=1)
    
def model_evalute(model, X, Y, y, filenames, model_name, mode=1):
    if mode == 1:
        mode_type = 'train'
        error_dir = ERROR_TRAIN
        print('evaluate train data')
    if mode == 2:
        mode_type = 'val'
        error_dir = ERROR_VALID
        print('evaluate validation data')
    if mode == 3:
        mode_type = 'test'
        error_dir = ERROR_TEST
        print('evaluate test data')

    score = model.evaluate(X, Y, verbose=VERBOSE)
    print('Eval score:', score[0])
    print('Eval acc:', score[1])

    predicted_row_answers = model.predict(X)
    predict_answers = np.argmax(predicted_row_answers, axis=1)

    cat_wrong_cnt = 0
    dog_wrong_cnt = 0

    """
     間違った画像を表示する
     plt.figure(figsize=(50,50))
     columns = 5
     for i, image in enumerate(X_test):
         plt.subplot(len(X_test) / columns + 1, columns, i + 1)
         predicted_num = predict_answers[i]
         answer = y_test[i]

         if predicted_num != answer:
             if predicted_num == 0:
                 label = 'cat'
                 cat_wrong_cnt += 1
             else:
                 label = 'dog'
                 dog_wrong_cnt += 1
             plt.title(label, fontsize=40)
             plt.axis('off')
             plt.imshow(image)
    """
    for i in range(len(X)-1):
        predicted_num = predict_answers[i]
        answer = y[i]

        if predicted_num != answer:
            filename = filenames[i]
            print('wrong prediction:', filename)
            print('wrong prediction score', predicted_row_answers[i])
            shutil.copy(filename, error_dir + filename.split('/')[-1])
            if predicted_num == 0:
                label = 'cat'
                cat_wrong_cnt += 1
            else:
                label = 'dog'
                dog_wrong_cnt += 1

    print('answer is cat, but dog is predicted:', cat_wrong_cnt)
    print('answer is dog, but cat is predicted:', dog_wrong_cnt)

    row = [model_name, mode_type, len(y[y == 1]) - dog_wrong_cnt, cat_wrong_cnt, dog_wrong_cnt, len(y[y == 0]) - cat_wrong_cnt]
    with open('result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def empty_error_dir():
    shutil.rmtree('error_images')
    os.mkdir('error_images')
    for new_dir in (ERROR_TRAIN, ERROR_VALID, ERROR_TEST):
        os.mkdir(new_dir)

if __name__ == '__main__':
    empty_error_dir()
    (X_train, y_train, train_filenames),(X_valid, y_valid,valid_filenames), (X_test, y_test, test_filenames) = prepare_data()
    print(len(X_train), 'X_train amount')
    print(len(y_train), 'y_train amount')
    print(len(X_valid), 'X_valid amount')    
    print(len(y_valid), 'y_valid amount')

    print(len(X_test), 'X_test amount')
    print(len(y_test), 'y_test amount')

    print(X_train.shape, 'X_train shape')
    print(X_valid.shape, 'X_valid shape')    
    print(X_test.shape, 'X_test shape')
    if len(X_train.shape) > 3:
        SHAPE = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    else:
        SHAPE = (X_train.shape[1], X_train.shape[2], 1)
    print(SHAPE, 'shape')

    if SHAPE[2] == 1:
        X_train = X_train.reshape(X_train.shape[0],  SIZE, SIZE, 1)
        X_valid = X_valid.reshape(X_valid.shape[0],  SIZE, SIZE, 1)
        X_test = X_test.reshape(X_test.shape[0],  SIZE, SIZE, 1)

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_valid /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    #functional API
    input_layer = Input(shape=SHAPE)

    #参考: https://keras.io/getting-started/sequential-model-guide/
    layer2 = Conv2D(FILTERS, 3, activation='relu', padding='same')(input_layer)
   # layer_bn = BatchNormalization()(layer2)
    layer3 = Conv2D(FILTERS, 3, padding='same')(layer2)
   # layer_bn = BatchNormalization()(layer3)
    layer_act = Activation('relu')(layer3)
    layer4 = MaxPooling2D()(layer_act)
    layer5 = Dropout(DROPOUT)(layer4)

    """
    layer6 = conv2d(filters * 2, 3, activation='relu', padding='same')(layer5)
    layer_bn = batchnormalization()(layer6)
    layer7 = conv2d(filters * 2, 3, activation='relu', padding='same')(layer_bn)
    layer_bn = batchnormalization()(layer7)
    layer_act = activation('relu')(layer_bn)
    layer8 = maxpooling2d()(layer_act)
    layer9 = dropout(dropout)(layer8)
    #"""

    flatten = Flatten()(layer5)
    #layer10 = Dense(10, activation='relu')(flatten)
    #layer11 = Dropout(DROPOUT)(layer10)
    output = Dense(NB_CLASSES, activation='softmax')(flatten)
    model = Model(input_layer, output)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LR), metrics=['accuracy'])

   # BATCH_SIZE = len(X_train)
    if len(sys.argv) > 1:
        print('推論モード')
        weights_file = sys.argv[1]
        model.load_weights(weights_file)
        model_name = SAVED_MODEL_PATH
    else:
        model_name = END_SAVED_MODEL_PATH
        print('学習モード')
        callbacks = [KC.TensorBoard(),
                    HistoryCheckpoint(filepath='chart/LearningCurve_{history}.png'
                                    , verbose=1
                                    , period=2
                                    , targets=[TargetHistory.Loss, TargetHistory.Accuracy, TargetHistory.ValidationLoss, TargetHistory.ValidationAccuracy]
                                   ),
                    KC.ModelCheckpoint(filepath=SAVED_MODEL_PATH,
                    verbose=1,
                    save_weights_only=True, #model全体を保存
                    save_best_only=True,
                    period=10)]

        if USE_DATAGEN:
            datagen = ImageDataGenerator(rotation_range=10,
                                        #width_shift_range=0.2,
                                        #height_shift_range=0.2,
                                        vertical_flip=True)
            datagen.fit(X_train)

            remove_log_files('logs/')
            history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE), epochs=NB_EPOCH, verbose=VERBOSE, validation_data=(X_valid, Y_valid), shuffle=True, callbacks=callbacks, samples_per_epoch=len(X_train))
        else:
            history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_data=(X_valid, Y_valid), shuffle=True, callbacks=callbacks)

        model.save_weights(END_SAVED_MODEL_PATH)

    for i, (X, Y, y, filenames) in enumerate(((X_train, Y_train, y_train, train_filenames), (X_valid, Y_valid, y_valid, valid_filenames), (X_test, Y_test, y_test, test_filenames))):
        model_evalute(model, X, Y, y, filenames, model_name, mode=i+1)
    shutil.make_archive('error_images', 'zip', root_dir = 'error_images')
