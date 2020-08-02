from keras.applications import ResNet152
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import progressbar
import h5py
import numpy as np

IMG_SIZE=(224, 224)

def get_features_from_image_list(images_list, img_path, batch_size=16, symmetric=False):
    model = ResNet152(weights="imagenet", include_top=False, pooling='avg')
    size = 2048
    widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(images_list), widgets=widgets).start()
    features_full = []
    for i in np.arange(0, len(images_list), batch_size):
        batch_paths = images_list[i:i + batch_size]
        batch_images = []
        for img in batch_paths:
            image = load_img(img_path + img, target_size=IMG_SIZE)
            image = img_to_array(image)
            if symmetric:
                image = np.array(list(map(lambda x:x[::-1], image)))
            image = np.expand_dims(image, axis=0)
            batch_images.append(image)     

        batch_images = np.vstack(batch_images)
        features = model.predict(batch_images, batch_size=batch_size)
        features = features.reshape((features.shape[0], size))
        features_full.append(features)
        try:
            pbar.update(i)
        except:
            pass
    pbar.finish()

    return np.vstack(features_full)
