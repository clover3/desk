import logging

from desk_util.io_helper import init_logging

LOG = logging.getLogger(__name__)


init_logging()
LOG.info("Loading library")
from sklearn.cluster import KMeans, DBSCAN
LOG.info("Loading BertHiddenStatesExtractor")
import fire
import numpy as np
import logging
import os
import pickle
import sys
from typing import List, Tuple, Union, Iterator
from tqdm import tqdm

LOG.info("Loading 1")
import torch
LOG.info("Loading 2")

LOG.info("Loading 3")
from transformers import BertModel, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification

LOG.info("Loading 4")
from chair.list_lib import left
from chair.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv
from desk_util.path_helper import get_model_save_path
from rule_gen.cpath import output_root_path
from rule_gen.reddit.classifier_loader.torch_misc import get_device
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
LOG.info("Loading Done")
