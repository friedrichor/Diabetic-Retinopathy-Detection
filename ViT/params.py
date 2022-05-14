import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


num_classes = 2

model = 'weights/model-15.pth'
path_train = 'data/data_split/train_enh/'
path_test = 'data/data_split/test/'
path_json = 'class_indices.json'


