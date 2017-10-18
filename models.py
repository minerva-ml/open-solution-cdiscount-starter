import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.applications.mobilenet import MobileNet, relu6, DepthwiseConv2D
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.generic_utils import CustomObjectScope

from utils import NeptuneMonitor


class BasicKerasClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self,
	             architecture_cfg,
	             training_cfg,
	             callbacks_cfg):
		self.architecture_cfg = architecture_cfg
		self.training_cfg = training_cfg
		self.callbacks_cfg = callbacks_cfg

		self.model = self._build_model(**self.architecture_cfg)
		self.callbacks = self._create_callbacks(**self.callbacks_cfg)

	def fit(self, X, y=None, **kwargs):
		train_flow, train_steps = X['X']
		valid_flow, valid_steps = X['valid']

		self.model.fit_generator(train_flow,
		                         steps_per_epoch=train_steps,
		                         validation_data=valid_flow,
		                         validation_steps=valid_steps,
		                         callbacks=self.callbacks,
		                         **self.training_cfg)
		self.model = self._load_best_model(self.model_filepath)
		return self

	def predict(self, X, **kwargs):
		self.model = self._load_best_model(self.model_filepath)
		test_flow, test_steps = X['X']
		predictions = self.model.predict_generator(test_flow, test_steps, verbose=1)
		return self._format_predictions(predictions)

	def reset(self):
		self.model = self._build_model(**self.architecture_config)

	def _build_model(self, **kwargs):
		return NotImplementedError

	def _load_best_model(self, filepath):
		return load_model(filepath)

	def _create_callbacks(self, **kwargs):
		self.model_filepath = ''
		return NotImplementedError

	def _format_predictions(self, predictions, **kwargs):
		return NotImplementedError


class KerasInception(BasicKerasClassifier):
	def _build_model(self, input_size, classes, trainable_threshold):
		base_model = self._load_pretrained_model(input_size)
		for i, layer in enumerate(base_model.layers):
			if i < trainable_threshold:
				layer.trainable = False
			else:
				layer.trainable = True

		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		predictions = Dense(classes, activation='softmax', name='output')(x)

		model = Model(inputs=base_model.input, outputs=predictions)
		sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
		model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
		return model

	def _load_pretrained_model(self, input_size, **kwargs):
		return InceptionV3(include_top=False, weights='imagenet', input_shape=(input_size, input_size, 3))

	def _load_best_model(self, filepath):
		return load_model(filepath)

	def _create_callbacks(self, models_dir, model_name):
		self.model_filepath = os.path.join(models_dir, '{}.h5'.format(model_name))

		model_checkpoint = ModelCheckpoint(self.model_filepath, monitor='val_loss', save_best_only=True)
		neptune = NeptuneMonitor(model_name)

		return [model_checkpoint, neptune]

	def _format_predictions(self, predictions):
		return predictions


class KerasMobileNet(BasicKerasClassifier):
	def _build_model(self, input_size, classes):
		base_model = self._load_pretrained_model(input_size)
		for layer in base_model.layers:
			layer.trainable = True

		x = base_model.output
		x = Flatten()(x)
		x = Dense(1024, activation='relu', name='fc1')(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		predictions = Dense(classes, activation='softmax', name='output')(x)

		model = Model(inputs=base_model.input, outputs=predictions)
		sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
		model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
		return model

	def _load_pretrained_model(self, input_size, **kwargs):
		return MobileNet(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))

	def _load_best_model(self, filepath):
		return load_mobilenets_model(filepath)

	def _create_callbacks(self, models_dir, model_name):
		self.model_filepath = os.path.join(models_dir, '{}.h5'.format(model_name))

		model_checkpoint = ModelCheckpoint(self.model_filepath, monitor='val_loss', save_best_only=True)
		neptune = NeptuneMonitor(model_name)

		return [model_checkpoint, neptune]

	def _format_predictions(self, predictions):
		return predictions


def load_mobilenets_model(filepath):
	with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}):
		model = load_model(filepath)
	return model
