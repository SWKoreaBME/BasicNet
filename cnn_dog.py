"""
This code has been created by CCP DIA team.
Since this codes may include IP of other developers, please do not copy for commercial purpose.
"""


# Setting
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Dropout
from keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.initializers import glorot_uniform
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import plot_model
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import os

from sklearn.model_selection import train_test_split
import logging



base_path = 'C:/Users/YY/Documents/Data/CCP/Crawled/Dog/'
X = []
img_list = []




def load_and_preprocess_data():
    y = []
    img_dataset_list = []

    classes_list = os.listdir(base_path)
    # classes_list = ['Beagle', 'Bulldog', 'Dalmatian' ... ]
    # 위 dataset_path안에 있는 폴더명을 리스트로 만듦
    logging.debug('classes_list {}'.format(classes_list))

    for class_name in classes_list:
        # img_list: 'Beagle_0.jpg'와 같은 이미지명을 담은 리스트임
        img_list = os.listdir(os.path.join(base_path, class_name))

        for img_name in img_list:
            # 각각의 이미지의 path name을 정의한다.
            img_path_name = os.path.join(base_path, class_name, img_name)
            # logging.debug('img_path {}'.format(img_path_name))

            # 이미지 로드
            img = image.load_img(img_path_name, target_size=(128, 128))

            # 이미지를 np.array로 바꿔줌
            # shape은 (height, width, channels) = (128, 128, 3)
            img_input = image.img_to_array(img)
            # logging.debug('img_input.shape {}'.format(img_input.shape))

            img_dataset_list.append(img_input)

            y.append(class_name)

    # 변형한 img_input_resize를 모두 담은 리스트인 img_dataset_list를 np.array로 바꿔줌
    img_dataset_arr = np.array(img_dataset_list)
    logging.debug('img_dataset_arr {}'.format(img_dataset_arr.shape))
    # (m, 128, 128, 3) for RGB

    # 자료형 변경
    img_dataset_arr = img_dataset_arr.astype('float64')

    # 정규화
    img_dataset_arr /= 255

    # Dimension ordering
    # 사진이 흑백이라면 (img_channels = 1)
    # th는 channel이 앞에 오는 것을 말한다.
    if K.image_dim_ordering() is 'th':
        logging.debug('img_dataset_arr.shape {}'.format(img_dataset_arr.shape))  # (m, 3, 128, 128)
    else:
        logging.debug('img_dataset_arr.shape {}'.format(img_dataset_arr.shape))  # (m, 128, 128, 3)

    print(img_dataset_arr.shape)
    # TODO

    return img_dataset_arr, y

img_dataset_arr, y = load_and_preprocess_data()

dog_dict = {'Beagle':0, 'Bulldog':1, 'Dalmatian':2, 'German Shepherd':3, 'Maltese':4,
            'Poodle':5, 'Pug':6, 'Shih Tzu':7, 'Siberian Husky':8, 'Yorkshire Terrier':9}

X = img_dataset_arr
Y = pd.Series(y).map(dog_dict)
Y = np.eye(10)[np.array(Y).astype(int)]

print("X 1개의 shape: ", X[0].shape)
print("Y label 1개의 shape: ", Y[0].shape)

X_train, X_test ,Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)


in_shape = (128, 128, 3)


def build_model(in_shape, nb_classes):
    x_input = Input(in_shape)
    x = ZeroPadding2D(padding=(2, 2))(x_input)

    h = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid', input_shape=in_shape)(x)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(h)

    h = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(h)

    h = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(h)

    h = Flatten()(h)
    h = Dense(units=64, activation='relu')(h)
    h = Dropout(rate=0.5)(h)
    y = Dense(units=nb_classes, activation='softmax', name='preds')(h)

    model = Model(inputs=x_input, outputs=y, name='WHW')

    return model


model = build_model(nb_classes=10, in_shape=in_shape)



# 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))



# Test the result
preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))



# 다른 이미지로 테스트
def img_test():
    img_path = 'C:/Users/YY/Documents/Data/CCP/Crawled/Dog/Maltese/Maltese_60.jpg'
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = x/255
    my_image = scipy.misc.imread(img_path)
    result = np.argmax(model.predict(x))
    return result, my_image

result, my_image = img_test()
print(result)

imshow(my_image)
plt.show()






