neptune send run_manager.py \
--config experiment_config.yaml \
--pip-requirements-file requirements.txt \
--project-key CDIS \
--environment keras-2.0-gpu-py3 \
--worker gcp-gpu-medium \
-- run_pipeline