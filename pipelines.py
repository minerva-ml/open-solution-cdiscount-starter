import os

from sklearn.externals import joblib

from preprocessing import LabelEncoderWrapper, KerasDataLoader
from models import KerasMobileNet, load_mobilenets_model, KerasInception
from postprocessing import PredictionAverage
from utils import register_pipeline


@register_pipeline
def InceptionPipeline(num_classes, epochs, workers, models_dir):
    pipe_legs_params = {'inceptionv3_180_{}'.format(num_classes): (KerasInception, 180, 128)}
    pipe_legs = []
    for name, (model, target_size, batch_size) in pipe_legs_params.items():
        leg = DeepPipeline([('loader', KerasDataLoader(num_classes, target_size, batch_size)),
                            ('model', model(architecture_cfg={'input_size': target_size, 'classes': num_classes,
                                                              'trainable_threshold': 172},  # top 2 inception blocks id
                                            training_cfg={'epochs': epochs, 'workers': workers, 'verbose': 1},
                                            callbacks_cfg={'models_dir': models_dir, 'model_name': name}))])
        pipe_legs.append((name, leg))

    pipe_avg = PredictionAverage(pipe_legs)

    pipeline = LabelEncoderWrapper(pipe_avg)

    return pipeline


@register_pipeline
def SimpleStarterPipeline(num_classes, epochs, workers, models_dir):
    target_size = 128
    batch_size = 256
    name = 'mobilenet_{}_{}'.format(target_size, num_classes)
    leg = DeepPipeline([('loader', KerasDataLoader(num_classes, target_size, batch_size)),
                        ('model', KerasMobileNet(architecture_cfg={'input_size': target_size, 'classes': num_classes},
                                                 training_cfg={'epochs': epochs, 'workers': workers, 'verbose': 1},
                                                 callbacks_cfg={'models_dir': models_dir, 'model_name': name}))])

    pipe_avg = PredictionAverage([(name, leg)])

    pipeline = LabelEncoderWrapper(pipe_avg)

    return pipeline


@register_pipeline
def MobilenetEnsemblePipeline(num_classes, epochs, workers, models_dir):
    pipe_legs_params = {'mobilenet_128_{}'.format(num_classes): (128, 128),
                        'mobilenet_160_{}'.format(num_classes): (160, 64),
                        'mobilenet_192_{}'.format(num_classes): (192, 32),
                        }
    pipe_legs = []
    for name, (target_size, batch_size) in pipe_legs_params.items():
        leg = DeepPipeline([('loader', KerasDataLoader(num_classes, target_size, batch_size)),
                            ('model', KerasMobileNet(
                                architecture_cfg={'input_size': target_size, 'classes': num_classes},
                                training_cfg={'epochs': epochs, 'workers': workers, 'verbose': 1},
                                callbacks_cfg={'models_dir': models_dir, 'model_name': name}))])
        pipe_legs.append((name, leg))

    pipe_avg = PredictionAverage(pipe_legs)

    pipeline = LabelEncoderWrapper(pipe_avg)

    return pipeline


class DeepPipeline():
    def __init__(self, steps):
        self.loader = dict(steps)['loader']
        self.deep_model = dict(steps)['model']

    def fit(self, X, y, validation_data, img_dataset_filepath):
        self.loader.fit(X, y, validation_data, img_dataset_filepath)
        datagens = self.loader.transform(X, y, validation_data, img_dataset_filepath)
        self.deep_model.fit(datagens)

    def predict(self, X, img_dataset_filepath):
        datagens = self.loader.transform(X, y=None, validation_data=None, img_dataset_filepath=img_dataset_filepath)
        prediction = self.deep_model.predict(datagens)
        return prediction


def pipeline_dump(pipeline, filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    encoder = pipeline.encoder
    encoder_filepath = os.path.join(filepath, 'encoder.pkl')
    joblib.dump(encoder, encoder_filepath)

    for name, leg in pipeline.base_estimator.steps:
        model_filepath = os.path.join(filepath, '{}.h5'.format(name))
        leg.deep_model.model.save(model_filepath)


def pipeline_load(pipeline, filepath):
    encoder_filepath = os.path.join(filepath, 'encoder.pkl')
    pipeline.encoder = joblib.load(encoder_filepath)

    for name, leg in pipeline.base_estimator.steps:
        leg_filepath = os.path.join(filepath, '{}.h5'.format(name))
        leg.deep_model.model = load_mobilenets_model(leg_filepath)
    return pipeline
