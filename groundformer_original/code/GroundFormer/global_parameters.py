import os

# QA annotations of NExT-GQA / NExT-QA (CSV, gsub_*.json, frame2time_*.json)
DEFAULT_DATASET_DIR = "../../datasets/"
# STAR keeps annotations and features together, see docs/star.md
DEFAULT_STAR_DIR = "../../../data/star/"
# where the pretrained language models are cached
DEFAULT_MODEL_DIR = "../../../data/pretrain_models/"

TRANSFORMERS_PATH = os.path.join(DEFAULT_MODEL_DIR, "transformers")
