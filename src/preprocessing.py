from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import progressbar
import numpy as np

def preprocessing_from_image_list(images_list, img_path, IMG_SIZE=(224, 224)):
    widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(images_list), widgets=widgets).start()
    batch_images = []
    for i, img in enumerate(images_list):
        image = load_img(img_path + img, target_size=IMG_SIZE)
        image = img_to_array(image)
        batch_images.append(image)
        pbar.update(i)
    batch_images = np.vstack(batch_images)

    return batch_images