import os
from datetime import datetime
from argparse import ArgumentParser
import yaml
import struct

import bson
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import register_action, neptune_post_pipeline_score, safe_sample, registered_actions, \
    registered_pipelines
from pipelines import pipeline_load, pipeline_dump


@register_action
def run_pipeline(args):
    train_valid_split(args)
    sample(args)
    train_pipeline(args)
    evaluate_pipeline(args)
    predict_pipeline(args)


@register_action
def train_pipeline(args):
    train_meta, valid_meta = _load_meta_training(args)

    if args.sample_validation:
        valid_meta = valid_meta.sample(args.sample_validation, replace=False, random_state=1234)

    if args.dev_mode:
        train_meta = train_meta.sample(1024, replace=False, random_state=1234)
        valid_meta = valid_meta.sample(128, replace=False, random_state=1234)
        epochs = 2
    else:
        epochs = _parse_neptune_params(args, 'epochs')

    train_filepath = os.path.join(args.raw_data_dir, 'train.bson')

    pipeline_name = _parse_neptune_params(args, 'pipeline_name')
    Pipeline = registered_pipelines[pipeline_name]
    pipeline = Pipeline(num_classes=_parse_neptune_params(args, 'top_categories') + 1,
                        epochs=epochs,
                        workers=args.nb_workers,
                        models_dir=os.path.join(args.models_dir, 'single_models'),
                        )
    pipeline.fit(X=train_meta, y=train_meta['category_id'],
                 validation_data=(valid_meta, valid_meta['category_id']),
                 img_dataset_filepath=train_filepath)
    pipeline_filepath = os.path.join(os.path.join(args.models_dir, 'pipelines'),
                                     '{}_{}'.format(args.name, pipeline_name))
    pipeline_dump(pipeline, pipeline_filepath)


@register_action
def evaluate_pipeline(args):
    train_meta, valid_meta = _load_meta_training(args)

    if args.sample_validation:
        valid_meta = valid_meta.sample(args.sample_validation, replace=False, random_state=1234)

    if args.dev_mode:
        valid_meta = valid_meta.sample(128, replace=False, random_state=1234)

    train_filepath = os.path.join(args.raw_data_dir, 'train.bson')

    pipeline_name = _parse_neptune_params(args, 'pipeline_name')
    Pipeline = registered_pipelines[pipeline_name]
    pipeline = Pipeline(num_classes=_parse_neptune_params(args, 'top_categories') + 1,
                        epochs=_parse_neptune_params(args, 'epochs'),
                        workers=args.nb_workers,
                        models_dir=os.path.join(args.models_dir, 'single_models'),
                        )
    pipeline_filepath = os.path.join(os.path.join(args.models_dir, 'pipelines'),
                                     '{}_{}'.format(args.name, pipeline_name))
    pipeline = pipeline_load(pipeline, pipeline_filepath)

    y_pred = pipeline.predict(X=valid_meta, img_dataset_filepath=train_filepath)
    y_true = valid_meta['category_id']
    score = accuracy_score(y_true, y_pred)
    neptune_post_pipeline_score(score)


@register_action
def predict_pipeline(args):
    test_meta = _load_meta_testing(args)

    if args.dev_mode:
        test_meta = test_meta.sample(128, replace=False, random_state=1234)

    pipeline_name = _parse_neptune_params(args, 'pipeline_name')
    Pipeline = registered_pipelines[pipeline_name]
    pipeline = Pipeline(num_classes=_parse_neptune_params(args, 'top_categories') + 1,
                        epochs=_parse_neptune_params(args, 'epochs'),
                        workers=args.nb_workers,
                        models_dir=os.path.join(args.models_dir, 'single_models'),
                        )
    pipeline_filepath = os.path.join(os.path.join(args.models_dir, 'pipelines'),
                                     '{}_{}'.format(args.name, pipeline_name))
    pipeline = pipeline_load(pipeline, pipeline_filepath)

    test_filepath = os.path.join(args.raw_data_dir, 'test.bson')
    y_test_pred = pipeline.predict(X=test_meta, img_dataset_filepath=test_filepath)

    submission = test_meta[['_id']]
    submission['category_id'] = y_test_pred
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    submission_filepath = os.path.join(args.submissions_dir,
                                       '{}_{}.csv'.format('{}_{}'.format(args.name, pipeline_name), timestr))
    submission.to_csv(submission_filepath, index=None)


@register_action
def sample(args):
    meta_data_filepath = os.path.join(args.meta_data_processed_dir, 'meta_train_v1.csv')
    meta_train = pd.read_csv(meta_data_filepath)

    top_cat = _parse_neptune_params(args, 'top_categories')
    img_per_cat = _parse_neptune_params(args, 'images_per_category')
    meta_train_sampled = _sample_train(meta_train, top_cat, img_per_cat)
    sampled_filepath = meta_data_filepath.replace('meta_train_v1',
                                                  'meta_train_v1_topcat{}_imgnr{}'.format(top_cat, img_per_cat))
    meta_train_sampled.to_csv(sampled_filepath, index=None)


def _sample_train(meta, top_cat, img_per_cat):
    top_ids = meta.groupby('category_id').size().sort_values(ascending=False).reset_index()[:top_cat][
        'category_id'].tolist()
    meta_top_categories = meta[meta['category_id'].isin(top_ids)]
    meta_top_categories = meta_top_categories.groupby('category_id').apply(
        lambda x: safe_sample(x, img_per_cat))
    meta_top_categories = meta_top_categories.sample(frac=1, random_state=1234).reset_index(drop=True)
    return meta_top_categories


@register_action
def train_valid_split(args):
    meta_data_filepath = os.path.join(args.meta_data_dir, 'meta_train.csv')

    meta_train_filepath = os.path.join(args.meta_data_processed_dir, 'meta_train_v1.csv')
    meta_valid_filepath = os.path.join(args.meta_data_processed_dir, 'meta_valid_v1.csv')

    meta_data = pd.read_csv(meta_data_filepath)
    meta_train, meta_valid = train_test_split(meta_data, train_size=args.train_ratio, random_state=args.seed)

    meta_train.to_csv(meta_train_filepath, index=None)
    meta_valid.to_csv(meta_valid_filepath, index=None)


@register_action
def create_metadata(args):
    _extract_meta(args, train=True)
    _extract_meta(args, train=False)


def _extract_meta(args, train=True):
    if train:
        prefix = 'train'
    else:
        prefix = 'test'

    raw_data_filepath = os.path.join(args.raw_data_dir, '{}.bson'.format(prefix))
    meta_data_filepath = os.path.join(args.meta_data_dir, 'meta_{}.csv'.format(prefix))

    meta = []
    with open(raw_data_filepath, 'rb') as f:
        offset = 0
        while True:
            print(offset)
            f.seek(offset)

            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break
            # Decode item length:
            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length, "%i vs %i" % (len(item_data), length)

            # Check if we can decode
            item = bson.BSON(item_data).decode()
            if train:
                row = (item['_id'], item['category_id'], offset, length, len(item['imgs']))
            else:
                row = (item['_id'], offset, length, len(item['imgs']))
            meta.append(row)
            offset += length

        if train:
            meta_df = pd.DataFrame(data=meta, columns=['_id', 'category_id', 'offset', 'length', 'num_pictures'])
        else:
            meta_df = pd.DataFrame(data=meta, columns=['_id', 'offset', 'length', 'num_pictures'])

        meta_df.to_csv(meta_data_filepath, index=False)


def _load_meta_training(args):
    top_cat = _parse_neptune_params(args, 'top_categories')
    img_per_cat = _parse_neptune_params(args, 'images_per_category')

    meta_valid_filepath = os.path.join(args.meta_data_processed_dir, 'meta_valid_v1.csv')
    train_filename = 'meta_train_v1_topcat{}_imgnr{}'.format(top_cat, img_per_cat)
    meta_train_filepath = meta_valid_filepath.replace('meta_valid_v1', train_filename)

    train = pd.read_csv(meta_train_filepath)
    valid = pd.read_csv(meta_valid_filepath)
    return train, valid


def _load_meta_testing(args):
    meta_test_filepath = os.path.join(args.meta_data_dir, 'meta_test.csv')

    test = pd.read_csv(meta_test_filepath)
    return test


def _parse_neptune_params(args, query_param):
    params = args.properties
    parsed = [param['value'] for param in params if param['key'] == query_param][0]
    return parsed


def prepare_environment(args):
    dir_paths = [args.submissions_dir, args.meta_data_processed_dir]
    for fold in ['valid', 'test']:
        dir_paths.append(os.path.join(args.predictions_dir, fold))
    for model_type in ['single_models', 'pipelines']:
        dir_paths.append(os.path.join(args.models_dir, model_type))

    for dir_path in dir_paths:
        os.makedirs(dir_path, exist_ok=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('action')
    parser.add_argument('-e', '--experiment_config_file', default='experiment_config.yaml')
    parser.add_argument('-c', '--data_config_file', default='data_config.yaml')
    parser.add_argument('-sv', '--sample_validation', type=int, default=10000)
    parser.add_argument('-w', '--nb_workers', type=int, default=4)
    parser.add_argument('-m', '--dev_mode', action='store_true')
    parser.add_argument('-r', '--train_ratio', type=float, default=0.8)
    parser.add_argument('-s', '--seed', type=int, default=1234)

    args = parser.parse_args()
    with open(args.experiment_config_file) as f:
        exp_config = yaml.load(f)

    with open(args.data_config_file) as f:
        data_config = yaml.load(f)

    config_merged = {**exp_config, **data_config}

    for key, value in config_merged.items():
        setattr(args, key, value)

    return args


if __name__ == '__main__':
    args = parse_args()
    prepare_environment(args)
    registered_actions[args.action](args)
