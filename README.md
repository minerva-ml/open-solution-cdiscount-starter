# What is Cdiscount starter?
This is ready to use, end-to-end sample solution for the currently running [Kaggle Cdiscount challenge](https://www.kaggle.com/c/cdiscount-image-classification-challenge).

It involves data loading and augmentation, model training (many different architectures), ensembling and submit generator.

# How to run Cdiscount starter?

## Installation

1. Install the requirements
   ```bash
   pip install -r requirements.txt
   ```

1. Install [neptune](https://neptune.ml "Machine Learning Lab") by simply
   ```bash
   pip install neptune-cli
   ```

1. Finish [neptune](https://neptune.ml "Machine Learning Lab") installation by running
   ```bash
   neptune login
   ```

1. Finally, open [neptune](https://neptune.ml "Machine Learning Lab") and create project `cdiscount`. Check the _project key_ because you will use it later (most likely it is: CDIS).

Now, you are ready to run the code and train some models...

## Run code
**remark about the competition data**: We have uploaded the data to the [neptune](https://neptune.ml "Machine Learning Lab") platform. It is available in the `/public/cdiscount` directory. Moreover, we created the `meta_data` file for large .bson files in the `/public/Cdiscount/meta` directory. It makes the process way faster.

You can run this end-to-end solution in two ways:
+ If you wish to work on your own machine you can run
   ```bash
   neptune run run_manager.py -- run_pipeline
   ```
+ Deploying on cloud via [neptune](https://neptune.ml "Machine Learning Lab") is super easy
  + just go
    ```bash
    source run_neptune_command.sh
    ```

  + more advanced option is to run
    ```bash
    neptune send run_manager.py \
    --config experiment_config.yaml \
    --pip-requirements-file neptune_requirements.txt \
    --project-key CDIS \
    --environment keras-2.0-gpu-py3 \
    --worker gcp-gpu-medium \
    -- run_pipeline
    ```

## Collect results and upload to Kaggle
Navigate to `/output/project_data/submissions`, get your submission file, upload it to Kaggle and check your rank in the competition!

# Advanced options
## custom data directories
If you do not wish to use default data directories, you can specify custom paths in the `data_config.yaml`
```yaml
raw_data_dir: /public/Cdiscount
meta_data_dir: /public/Cdiscount/meta
meta_data_processed_dir: /output/project_data/meta_processed
models_dir: /output/project_data/models
predictions_dir: /output/project_data/predictions
submissions_dir: /output/project_data/submissions
```

## meta data creating
If you want to create meta data locally you should run
```bash
python run_manager create_metadata
```
and your metadata will be stored in the `meta_data_dir`

## data sampling
Since the dataset is very large we suggest that you sample training dataset to a manageable size. Something like 1000 most common categories and 1000 images per category seems reasonable to start with. Nevertheless, You can tweak it however you want in the `experiment_config.yaml` file
```yaml
properties:
  - key: top_categories
    value: 100
  - key: images_per_category
    value: 100
  - key: epochs
    value: 10
  - key: pipeline_name
    value: InceptionPipeline
```

## hyperparameter space search
If you like to search the hyperparameter space, [neptune](https://neptune.ml "Machine Learning Lab") can do this for you. Check out [hyperparameter optimization](https://docs.neptune.ml/advanced-topics/hyperparameter-optimization/).

## training without [neptune](https://neptune.ml "Machine Learning Lab")
We give you an option to run this code without neptune. The transition is seamless, just follow these steps:
1. Download the competition data to some folder `your_raw_data_dir`

1. specify data directories in the `data_config.yaml`

1. run python code
   ```bash
     python run_manager.py run_pipeline
   ```

# Final remarks
Please feel free to modify this code in order to improve your score. Add new models, pre- and post-processing routines or ensembling methods.

Have fun competing on this Kaggle challenge!
