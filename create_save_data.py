import cv2
import numpy as np
import os
import glob

train_dir = 'clean_images/train_images/'
valid_dir = 'clean_images/valid_images/'
test_dir = 'clean_images/test_images/'
COLOR_MODE = 1
if COLOR_MODE == 0:
    save_dir = 'data/gray/'
else:
    save_dir = 'data/color/'
SIZE = 64

def create_images_answers(dir_path):
    files = glob.glob(os.path.join(dir_path, '*.jpg'))
    files.sort()
    images = [resize_for_model(cv2.imread(file, COLOR_MODE)) for file in files]

    return images

def resize_for_model(image):
    # np形式のimageを特定の大きさにresizeする。
    return cv2.resize(image, (SIZE, SIZE))

def save_data():
    # 猫(0)と犬(1)の画像を取得してフラグを追加したにシャッフル加工してデータとして返す。
    
    train_dog_images = create_images_answers(train_dir + 'dog/')
    train_cat_images = create_images_answers(train_dir + 'cat/')
    valid_dog_images = create_images_answers(valid_dir + 'dog/')
    valid_cat_images = create_images_answers(valid_dir + 'cat/')
    test_dog_images = create_images_answers(test_dir + 'dog/')
    test_cat_images = create_images_answers(test_dir + 'cat/')

    train_images = np.array(train_dog_images + train_cat_images)
    np.save(save_dir + 'train_images.npy', train_images)
    valid_images = np.array(valid_dog_images + valid_cat_images)
    np.save(save_dir + 'valid_images.npy', valid_images)
    test_images = np.array(test_dog_images + test_cat_images)
    np.save(save_dir + 'test_images.npy', test_images)

    train_answers = np.array([0] * len(train_dog_images) + [1] * len(train_cat_images))
    np.save(save_dir + 'train_answers.npy', train_answers)
    valid_answers = np.array([0] * len(valid_dog_images) + [1] * len(valid_cat_images))
    np.save(save_dir + 'valid_answers.npy', valid_answers)
    test_answers = np.array([0] * len(test_dog_images) + [1] * len(test_cat_images))
    np.save(save_dir + 'test_answers.npy', test_answers)

if __name__ == '__main__':
    save_data()