# What is this 
This is an end-to-end sample solution to the kaggle cdiscount challenge 
https://www.kaggle.com/c/cdiscount-image-classification-challenge

It involves data loading, data augmentation, model training (many different architectures) and ensembling. 
Have fun extending this code to build even better solution.

# How to use it

## Prerequisites

Download the competition data and store it in some folder `your_raw_data_dir`

Install the requirements
```bash
pip install -r requirements.txt
```

Install neptune by simply
```bash
pip install neptune-cli
```
Now you can finish your installation by running

```bash
neptune login
```

open `neptune.ml` and create project `cdiscount`. Check the project key because you will
use it later (most likely it's CDIS).

## Setup 
Open data_config.yaml and specify paths to your data directories. You can just leave them as they are if you'd like.
```yaml
raw_data_dir: /public/Cdiscount
meta_data_dir: /public/Cdiscount/meta
meta_data_processed_dir: /output/project_data/meta_processed
models_dir: /output/project_data/models
predictions_dir: /output/project_data/predictions
submissions_dir: /output/project_data/submissions
```

We made things easy by uploading the data for you!
You don't have to wait for days to download it. It is available in the /public/cdiscount directory.
Moreover we created the meta_data for large .bson files in the `/public/Cdiscount/meta`.
That makes the process way faster.

# Run pipeline
Since the dataset is very large we suggest that you sample training dataset to a managable size.
Something like 1000 most common categories and 1000 images per category seems reasonable.
You can tweak it however you want!

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

If you would like to search the hyperparameter space and have neptune do that for you. Just read https://docs.neptune.ml/advanced-topics/hyperparameter-optimization/ .

Now all you need to do is run this pipeline.

If you want to work on your own machine you can run

```bash
neptune run experiment_manager.py -- run_pipeline
```

Deploying this on cloud via neptune is super easy. 
You can either run:

```bash
neptune send experiment_manager.py \
--pip-requirements-file neptune_requirements.txt \
--project-key CDIS \
--environment keras-2.0-gpu-py3 \
--worker gcp-gpu-medium \
-- run_pipeline -m
```

or if you are feeling lazy just go:
```bash
source run_neptune_command.sh
```

If you simply want to use python of course you can.
The transition is seamless.

```bash
python experiment_manager.py run_pipeline
```

Navigate to
`/output/project_data/submissions`
get your submission and check your kaggle rank!

Improve your score by extending this pipeline.
Add new models, pre and post processing or ensembling.
Read more on our blog at https://blog.deepsense.ai/

