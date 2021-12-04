import Data

import INVARIANT
import FLAGS

FLAGS.CHECK()

dataset = Data.Dataset(
    label_folder = FLAGS.DATA.TRAIN.LABEL_FOLDER,
    image_folder = FLAGS.DATA.TRAIN.IMAGE_FOLDER,
    batch_size = FLAGS.DATA.TRAIN.BATCH_SIZE,
    image_shape = INVARIANT.IMAGE_SHAPE,
    object_map_param = INVARIANT.OBJECT_MAP_PARAM)

train_data,validation_data = dataset.split(
    validation_ratio = FLAGS.DATA.TRAIN.VALIDATION_SPLIT_RATIO ,
    random_state = FLAGS.DATA.TRAIN.VALIDATION_SPLIT_RANDOM_STATE)