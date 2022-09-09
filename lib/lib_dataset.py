import tensorflow as tf
from collections import Counter
import os
class Dataset:
    def __init__(self,dataset_tag,path_to_dataset,image_size, color_mode, batch_size):
        self.tag=dataset_tag
        self.path_root=path_to_dataset
        self.image_size=image_size
        self.color_mode=color_mode
        self.batch_size=batch_size

    
    def load_dataset(self):
        root_dataset=self.path_root
        image_size=self.image_size
        color_mode=self.color_mode
        batch_size=self.batch_size
        
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=True,
            featurewise_std_normalization=False,
            samplewise_std_normalization=True,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            brightness_range=None,
            shear_range=0.0,
            zoom_range=(0.95,0.95),#####
            channel_shift_range=0.0,
            fill_mode='nearest',
            cval=0.0,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=1./255,#######
            preprocessing_function=None,
            data_format=None,# 'channels_last'
            validation_split=0.2,
            dtype=None,#tf.float32
        )

        self.ds_train = self.datagen.flow_from_directory(
            directory=os.path.join (root_dataset),
            target_size=image_size,
            color_mode=color_mode, #"rgb",#grayscale', #
            classes=None,# 'sparse' for integer based instead of hot-encode
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True,
            seed=101,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            follow_links=False,
            subset='training',
            interpolation='nearest'
        )

        self.ds_validate = self.datagen.flow_from_directory(
            directory=os.path.join (root_dataset),
            target_size=image_size,
            color_mode=color_mode, #"rgb",#grayscale', #
            classes=None,# 'sparse' for integer based instead of hot-encode
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True,
            seed=101,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            follow_links=False,
            subset='validation',
            interpolation='nearest',
        )

        #class_names=ds_train.class_indices
        #print (class_names)
        #class_names = list(class_names.keys())
        return (self.ds_train, self.ds_train.labels),(self.ds_validate,self.ds_validate.labels)

    
    def as_dictionary(self, data_sub_set):
        #data_sub_set=self.ds_train
        counter = Counter(data_sub_set.classes)
        class_indices=data_sub_set.class_indices

        class_name =list(class_indices)
        class_indexes=list(class_indices.values())
        class_count=list(counter.values())

        data_Dictionary={}
        for i in range(len(class_name)):
            data_Dictionary[class_name[i]]=class_count[i]
        return data_Dictionary 
    
    def trainSet_as_dictionary(self):
        return self.as_dictionary(self.ds_train)
    def validateSet_as_dictionary(self):
        return self.as_dictionary(self.ds_validate)
    

        
    