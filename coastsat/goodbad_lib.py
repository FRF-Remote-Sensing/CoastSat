from torch.utils.data.dataset import Dataset  # For custom data-sets
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
import cv2
import numpy as np
import os


# dataset class
class RAFTERDataset(Dataset):
    def __init__(self, output_imClassifs, output_imRGBs, output_filename, im_classRGB):
        self.filelist = output_filename
        self.imClassifs = output_imClassifs
        self.im_classRGB = im_classRGB
        self.imRGBs = output_imRGBs
        self.img_rows = int(224)
        self.img_cols = int(224)

    def __getitem__(self, idx):
        return self.load_file(idx)

    def __len__(self):
        return len(self.filelist)

    def load_file(self, idx):

        image = self.imRGBs[idx]
        label = self.im_classRGB[idx]
        name = self.filelist[idx]

        resized_array = np.zeros((self.img_rows, self.img_cols, 6))
        resized_array[:, :, 0] = cv2.resize(image[:, :, 0], (int(self.img_cols), int(self.img_rows)), interpolation=cv2.INTER_CUBIC)
        resized_array[:, :, 1] = cv2.resize(image[:, :, 1], (int(self.img_cols), int(self.img_rows)), interpolation=cv2.INTER_CUBIC)
        resized_array[:, :, 2] = cv2.resize(image[:, :, 2], (int(self.img_cols), int(self.img_rows)), interpolation=cv2.INTER_CUBIC)
        resized_array[:, :, 3] = cv2.resize(label[:, :, 0], (int(self.img_cols), int(self.img_rows)), interpolation=cv2.INTER_CUBIC)
        resized_array[:, :, 4] = cv2.resize(label[:, :, 1], (int(self.img_cols), int(self.img_rows)), interpolation=cv2.INTER_CUBIC)
        resized_array[:, :, 5] = cv2.resize(label[:, :, 2], (int(self.img_cols), int(self.img_rows)), interpolation=cv2.INTER_CUBIC)

        resized_array_3 = np.zeros((self.img_rows, self.img_cols, 3))
        resized_array_3[:, :, 0] = np.mean(resized_array[:, :, :2], axis=-1)
        resized_array_3[:, :, 1] = np.mean(resized_array[:, :, 2:4], axis=-1)
        resized_array_3[:, :, 2] = np.mean(resized_array[:, :, 4:6], axis=-1)

        sample = {'image': resized_array_3}

        return sample, name

class RAFTERNet():
    def __init__(self):
        self.img_rows = 224
        self.img_cols = 224
        self.bands = 3

    def get_VGGnet(self):

        base_model = VGG16(input_shape=(self.img_rows, self.img_cols, 3),  # Shape of our images
                           include_top=False,  # Leave out the last fully connected layer
                           weights='imagenet')
        # Flatten the output layer to 1 dimension
        x = tf.keras.layers.Flatten()(base_model.output)

        # Add a fully connected layer with 512 hidden units and ReLU activation
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.models.Model(base_model.input, x)

        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=.001), loss='binary_crossentropy', metrics=[])

        return model

    def load_model(self):
        try:
            model = self.get_VGGnet()
            model.load_weights('./coastsat/chk/best_chk.h5')
            print("loaded model ")
            return model

        except:
            model = self.get_VGGnet()
            print("couldn't load model")
            model.summary()
            return model

    def get_data(self, dataset):
        name_batch = []
        dataset_len = len(dataset)
        img_batch = np.ndarray((dataset_len, self.img_rows, self.img_cols, self.bands), dtype=np.float32)
        for j in range(dataset_len):
            sample, name = dataset[j]
            img_batch[j] = sample['image']
            name_batch.append(name)

        return img_batch, name_batch
