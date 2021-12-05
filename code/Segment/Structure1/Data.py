import tensorflow as tf
import json
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self,label_folder,image_folder,batch_size,image_shape,object_map_param):
        self.label_folder = label_folder
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.object_map_param = object_map_param
    
    def split(self,validation_ratio,random_state):
        labels = [l.as_posix() for l in pathlib.Path(self.label_folder).iterdir()]
        train_labels, validation_labels = train_test_split(labels,test_size=validation_ratio,random_state=random_state)
        train_data = self.create_dataset(train_labels,False)
        validation_data = self.create_dataset(validation_labels,True)
        return train_data, validation_data
    
    def create_dataset(self,labels,is_validation):
        dataset = tf.data.Dataset.from_tensor_slices(labels)
        dataset = dataset.map(
            lambda path: tf.py_function(
                self.decode_json,
                [path],
                [tf.string, tf.int32, tf.float64, tf.float64, tf.float64]),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.cache()
        if not is_validation:
            dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.map(
            lambda image_path, group, center, wh, offset: tf.py_function(
                self.image_preprocess,
                [image_path, group, center, wh, offset],
                [tf.string, tf.float32, tf.int32, tf.float64, tf.float64, tf.float64]),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        if not is_validation:
            dataset = dataset.map(
            lambda image_path, image, group, center, wh, offset: tf.py_function(
                self.argumentation,
                [image_path, image, group, center, wh, offset],
                [tf.string, tf.float32, tf.int32, tf.float64, tf.float64, tf.float64]),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(
            lambda image_path, image, group, center, wh, offset: tf.py_function(
                self.transform,
                [image_path, image, group, center, wh, offset],
                [tf.string, tf.float32]+[tf.float64]*len(self.object_map_param)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
        
    def decode_json(self,path):
        label = json.loads(tf.io.read_file(path).numpy())

        image_path = tf.strings.join([self.image_folder,label['ImageName']],'/')
        
        group = np.array([[b['type']] if b['type']!=255 else [6] for b in label['bboxes']])
        center = np.array([b['rectangle']['center'] for b in label['bboxes']])
        wh = np.array([b['rectangle']['ltwh'][2:] for b in label['bboxes']])
        
        rectangle = np.array([b['rectangle']['points'] for b in label['bboxes']])
        polygon = np.array([b['polygon']['points'] for b in label['bboxes']])
        offset = polygon - rectangle
        
        return image_path, group, center, wh, offset
    
    def image_preprocess(self, image_path, group, center, wh, offset):
        image = tf.io.decode_jpeg(tf.io.read_file(image_path))
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.image.resize(image,self.image_shape)
        image = tf.clip_by_value(image,0,1)
        return image_path, image, group, center, wh, offset
    
    def argumentation(self, image_path, image, group, center, wh, offset):
        group = group.numpy()
        center = center.numpy()
        wh = wh.numpy()
        offset = offset.numpy()
        
        if np.random.choice([True,False],p=[0.4,0.6]):
            if np.random.choice([True,False],p=[0.7,0.3]):
                image = tf.image.random_contrast(image,0.5,2)
                image = tf.clip_by_value(image,0,1)

            if np.random.choice([True,False],p=[0.7,0.3]):
                image = tf.image.random_hue(image,0.25)
                image = tf.clip_by_value(image,0,1)

            if np.random.choice([True,False],p=[0.7,0.3]):
                image = tf.image.random_brightness(image,0.25)
                image = tf.clip_by_value(image,0,1)

            if np.random.choice([True,False],p=[0.7,0.3]):
                image = tf.image.random_saturation(image,0.5,2)
                image = tf.clip_by_value(image,0,1)
            
        if np.random.choice([True,False],p=[0.3,0.7]):
            image, center, wh, offset = self.random_resize(image, center, wh, offset)
        
        return image_path, image, group, center, wh, offset
    
    def transform(self, image_path, image, group, center, wh, offset):
        group = group.numpy()
        center = center.numpy()
        wh = wh.numpy()
        offset = offset.numpy()
        
        determine = np.ceil(np.max(wh*np.array(self.image_shape[::-1]),axis=-1)).astype(int)
        bboxes_order = np.argsort(determine)
        determine = determine[bboxes_order]
        group = group[bboxes_order]
        center = center[bboxes_order]
        wh = wh[bboxes_order]
        offset = offset[bboxes_order]
        
        object_maps = []
        for grid_count, object_determine_size_range in self.object_map_param:
            smallest, biggest = object_determine_size_range
            mask = ((smallest<=determine) & (determine<=biggest))

            blood = 1/(grid_count*2)
            grid_center_coor = np.linspace(0+blood,1-blood,grid_count)
            
            matched_group = group[mask]
            cxcy = center[mask]
            matched_wh = wh[mask]
            points_offset = offset[mask]

            belonging = (cxcy*grid_count).astype(int)
            match_grid_center_coor = np.stack([grid_center_coor[belonging[:,0]],grid_center_coor[belonging[:,1]]],axis=-1)

            center_offset = (cxcy - match_grid_center_coor) * grid_count * 2
            
            answer = np.concatenate([matched_group,center_offset,matched_wh,points_offset.reshape(-1,8)],axis=-1)
            object_map = np.zeros((grid_count,grid_count,1+(1+(2+2)+(4*2))))
            object_map[belonging[:,1],belonging[:,0],0] = 1
            object_map[belonging[:,1],belonging[:,0],1:] = answer

            object_maps.append(object_map)
        return image_path, image, *object_maps
    
    
    def random_resize(self,image, center, wh, offset):
        w_rate,h_rate = np.random.uniform(0.5,1,2)
        new_shape = (np.array([h_rate,w_rate]) * image.shape[:2]).astype(int)
        
        resized_image = tf.image.resize(image,new_shape)
        resized_image = tf.clip_by_value(resized_image,0,1)
        
        center = center * np.array([w_rate,h_rate])
        wh = wh * np.array([w_rate,h_rate])
        offset = offset * np.array([w_rate,h_rate])
        
        image_x_diff = self.image_shape[1]-new_shape[1]
        image_y_diff = self.image_shape[0]-new_shape[0]
        
        image_x_offset = np.random.randint(0,image_x_diff+1)
        image_y_offset = np.random.randint(0,image_y_diff+1)
        
        new_image = tf.pad(resized_image,
                           [
                               [image_y_offset,image_y_diff-image_y_offset],
                               [image_x_offset,image_x_diff-image_x_offset],
                               [0,0]
                           ])
        
        image_offset = np.array([image_x_offset/self.image_shape[1],image_y_offset/self.image_shape[0]])
        
        center = center + image_offset
        
        return new_image, center, wh, offset