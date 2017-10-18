from time import time
import logging

from keras.callbacks import Callback

try:
	from deepsense import neptune
except Exception:
	pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Starter Pipeline')

registered_actions = {}
registered_pipelines = {}


def register_action(func):
	registered_actions[func.__name__] = func
	return func


def register_pipeline(func):
	registered_pipelines[func.__name__] = func
	return func


def safe_sample(x, size):
	try:
		return x.sample(size, replace=False)
	except ValueError:
		return x


def timeit(logger=None):
	def real_decorator(function):
		def wrapper(*args, **kwargs):
			start_time = time()

			function(*args, **kwargs)

			end_time = time()
			elapsed_time = (end_time - start_time) / 3600.
			message = 'Time spent on {0} was {1:.2f} hours'.format(function.__name__, elapsed_time)
			if logger:
				logger.info(message)
			else:
				print(message)

		return wrapper

	return real_decorator


class NeptuneMonitor(Callback):
	def __init__(self, model_name):
		self.model_name = model_name
		self.ctx = neptune.Context()
		self.epoch_id = 0
		self.batch_id = 0

	def on_batch_end(self, batch, logs={}):
		self.batch_id += 1

		self.ctx.job.channel_send('{} Batch Log-loss training'.format(self.model_name),
		                          self.batch_id, logs['loss'])
		self.ctx.job.channel_send('{} Batch Accuracy training'.format(self.model_name),
		                          self.batch_id, logs['acc'])

	def on_epoch_end(self, epoch, logs={}):
		self.epoch_id += 1

		self.ctx.job.channel_send('{} Log-loss training'.format(self.model_name),
		                          self.epoch_id, logs['loss'])
		self.ctx.job.channel_send('{} Log-loss validation'.format(self.model_name),
		                          self.epoch_id, logs['val_loss'])
		self.ctx.job.channel_send('{} Accuracy training'.format(self.model_name),
		                          self.epoch_id, logs['acc'])
		self.ctx.job.channel_send('{} Accuracy validation'.format(self.model_name),
		                          self.epoch_id, logs['val_acc'])


def neptune_post_pipeline_score(score):
	ctx = neptune.Context()
	ctx.job.channel_send('Final Validation Score', 0, score)
