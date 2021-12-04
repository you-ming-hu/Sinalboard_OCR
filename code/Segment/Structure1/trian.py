import Data

import INVARIANT
import FLAGS

FLAGS.CHECK()

dataset = Data.Dataset(
    label_folder = FLAGS.TRAIN.LABEL_FOLDER,
    image_folder = FLAGS.TRAIN.IMAGE_FOLDER,
    batch_size = FLAGS.TRAIN.BATCH_SIZE,
    image_shape = INVARIANT.IMAGE_SHAPE,
    object_map_param = INVARIANT.OBJECT_MAP_PARAM)