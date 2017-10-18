import bson
from math import ceil

import io

import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, img_to_array
from keras.utils.np_utils import to_categorical


class LabelEncoderMissing(BaseEstimator, TransformerMixin):
    def fit(self, y):
        y = np.asarray(y).astype(str)
        self.classes_ = np.append(np.unique(y), 'other')
        return self

    def transform(self, y):
        y = np.asarray(y).astype(str)
        classes = np.unique(y)
        set_diff = np.setdiff1d(classes, self.classes_)
        if len(set_diff) > 0:
            np.place(y, np.in1d(y, set_diff), 'other')

        return np.searchsorted(self.classes_, y)

    def inverse_transform(self, y):
        y = np.asarray(y)
        y_inv = self.classes_[y]
        other_mask = y_inv == 'other'
        np.place(y_inv, other_mask, -999)
        return y_inv.astype(int)


class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.encoder = LabelEncoderMissing()

    def fit(self, X, y, validation_data=None, img_dataset_filepath=None):
        y_ = self.encoder.fit_transform(y)
        if validation_data is not None:
            X_valid, y_valid = validation_data
            y_valid_ = self.encoder.transform(y_valid)
        else:
            raise NotImplementedError
        self.base_estimator.fit(X=X, y=y_, validation_data=(X_valid, y_valid_),
                                img_dataset_filepath=img_dataset_filepath)
        return self

    def predict(self, X, img_dataset_filepath=None):
        y_pred_ = self.base_estimator.predict(X=X, img_dataset_filepath=img_dataset_filepath)
        y_pred = self.encoder.inverse_transform(y_pred_)
        return y_pred


class KerasDataLoader(BaseEstimator, TransformerMixin):
    def __init__(self,
                 num_classes,
                 target_size,
                 batch_size):
        self.num_classes = num_classes
        self.target_size = (target_size, target_size)
        self.batch_size = batch_size

    def fit(self, X, y=None, validation_data=None, img_dataset_filepath=None):
        return self

    def transform(self, X, y=None, validation_data=None, img_dataset_filepath=None):
        """Todo:
            pass datagen and flow args from experiment config
        """

        if y is None:
            y = np.zeros((X.shape[0], 1))

            datagen_args = {'rescale': 1. / 255
                            }
            flow_args = {'target_size': self.target_size,
                         'batch_size': self.batch_size,
                         'shuffle': False}
        else:
            datagen_args = {'rescale': 1. / 255,
                            'rotation_range': 10,
                            'width_shift_range': 0.2,
                            'height_shift_range': 0.2,
                            'shear_range': 0.2,
                            'zoom_range': 0.2,
                            'channel_shift_range': 0.2,
                            'fill_mode': 'nearest'
                            }
            flow_args = {'target_size': self.target_size,
                         'batch_size': self.batch_size,
                         'shuffle': True}
        y = self._prep_targets(y)
        X_flow, X_steps = build_bson_datagen(X, y, img_dataset_filepath, datagen_args, flow_args)
        if validation_data is not None:
            X_valid, y_valid = validation_data
            y_valid = self._prep_targets(y_valid)
            valid_flow, valid_steps = build_bson_datagen(X_valid, y_valid, img_dataset_filepath, datagen_args,
                                                          flow_args)
        else:
            valid_flow, valid_steps = None, None

        return {'X': (X_flow, X_steps),
                'valid': (valid_flow, valid_steps)}

    def _prep_targets(self, y):
        targets = to_categorical(np.array(y), num_classes=self.num_classes)
        return targets


def build_bson_datagen(X, y, bson_filepath, datagen_args, flow_args):
    datagen = bsonImageDataGenerator(**datagen_args)

    flow = datagen.flow_from_bson(X, y, bson_filepath, **flow_args)
    steps = ceil(X.shape[0] / flow_args['batch_size'])

    return flow, steps


class bsonImageDataGenerator(ImageDataGenerator):
    def flow_from_bson(self, X, y, bson_filepath,
                       target_size=(64, 64), color_mode='rgb', channel_order='tf',
                       batch_size=32, shuffle=True, seed=None):
        return bsonIterator(X, y, bson_filepath, self,
                             target_size, color_mode, channel_order,
                             batch_size, shuffle, seed)


class bsonIterator(Iterator):
    """Note:
        Tensorflow channels order only rgb only
    """

    def __init__(self, X, y, bson_filepath,
                 image_data_generator,
                 target_size, color_mode, channel_order,
                 batch_size, shuffle, seed):
        self.X = X
        self.y = y
        self.bson_filepath = bson_filepath
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.color_mode = color_mode
        self.channel_order = channel_order
        self.image_shape = self.target_size + (3,)
        self.data_format = K.image_data_format()

        self.samples = X.shape[0]

        super().__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        index_array_ = index_array[0]

        batch_x = np.zeros((len(index_array_),) + self.image_shape, dtype=K.floatx())
        batch_y = self.y[index_array_]

        with open(self.bson_filepath, 'rb') as bson_file:
            grayscale = self.color_mode == 'grayscale'
            # build batch of image data
            for i, j in enumerate(index_array_):
                img_metadata = self.X.iloc[j]
                img = load_bson_img(bson_file, img_metadata, grayscale=grayscale, target_size=self.target_size)
                x = img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x

        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


def load_bson_img(bson_file, img_metadata, grayscale=False, target_size=(64, 64)):
    """
    Note:
        This implementation is just taking the first image for the product, sometimes there are up to 4 images
    """
    bson_file.seek(img_metadata['offset'])
    item_data = bson_file.read(img_metadata['length'])
    item = bson.BSON(item_data).decode()
    img_byte = (item['imgs'][0]['picture'])
    img = Image.open(io.BytesIO(img_byte))
    img = img.resize(target_size)
    return img
