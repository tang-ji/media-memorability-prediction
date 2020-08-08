import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

from keras.applications import ResNet152
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import *

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

class Model_Final:
    def __init__(self, IMG_SIZE=(224, 224)):
        self.model_res = ResNet152(weights="imagenet", include_top=False, pooling='avg')
        self.IMG_SIZE = IMG_SIZE
        input_layer = self.model_res.layers[0]
        self.model = load_model("model")
        self.model_final = Model(self.model_res.input, self.model(self.model_res.output))
        
        image_output = self.model_final.output[:]
        last_conv_layer = self.model_final.get_layer('conv5_block3_out')
        grads = K.gradients(image_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        self.iterate = K.function([self.model_final.input], [pooled_grads, last_conv_layer.output[0]])
        
    def load_image(self, image_path):
        image = load_img(image_path, target_size=self.IMG_SIZE)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        return image
    
    def predict(self, image_path):
        image = self.load_image(image_path)
        return self.model_final.predict(image)
    
    def get_heatmap(self, image):
        pooled_grads_value, conv_layer_output_value = self.iterate([image])
        for i in range(512):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap
    
    def get_transformed_image(self, image_path, show=True):
        image_name = image_path.split(os.path.sep)[-1].split('.')[0]
        image = self.load_image(image_path)
        image_original = cv2.imread(image_path)
        heatmap = self.get_heatmap(image)
        heatmap = cv2.resize(heatmap, (image_original.shape[1], image_original.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + image_original
        if not os.path.exists('heatmaps/'):
            os.makedirs('heatmaps/')
        cv2.imwrite('heatmaps/{}_heatmap.jpg'.format(image_name), superimposed_img)
        if show:
            f, ax = plt.subplots(1, 2, figsize=(10,5), dpi=80)
            ax[0].set_title("Original image")
            ax[0].imshow(plt.imread(image_path))
            ax[0].axis('off')

            ax[1].set_title("Heatmap")
            ax[1].imshow(plt.imread('heatmaps/{}_heatmap.jpg'.format(image_name)))
            ax[1].axis('off')
        return