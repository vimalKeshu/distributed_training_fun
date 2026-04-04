import tensorflow as tf 
import numpy as np
import os 
import glob
from PIL import Image


# Converting the values into features
# _int64 is used for numeric values
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# _bytes is used for string/char values
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# tfrecords file name
tfrecord_filename = '/home/vimal/Downloads/data/training/vimal.tfrecords'

# Initiating the writer and creating the tfrecords file.
writer = tf.python_io.TFRecordWriter(tfrecord_filename)

# Loading training dataset for label vimal images
images = glob.glob('/home/vimal/Downloads/data/training/vimal/*.jpg')

# label 0 for vimal 
label = 0 
count=0
# Convert images into tfrecords
for image in images[:]:
  count=count+1
  print(count)
  img = np.array(Image.open(image))
  height = img.shape[0]
  width = img.shape[1]  
  feature = { 'label': _int64_feature(label),
              'height': _int64_feature(height),
              'width': _int64_feature(width),
              'image': _bytes_feature(img.tostring()) }
  # Create an example protocol buffer
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  # Writing the serialized example.
  writer.write(example.SerializeToString())
writer.close()
