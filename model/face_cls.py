# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from PIL import Image  
from scipy.misc import imread, imsave, imshow, imresize
import numpy as np

from net import cls_net_att as cls_net

from dataset import dataset_common
from preprocessing import cls_preprocessing

# scaffold related configuration
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 224,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')

# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './checkpoints_att',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')

FLAGS = tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES

def get_checkpoint():
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    return checkpoint_path

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with tf.Graph().as_default():
        out_shape = [FLAGS.train_image_size] * 2

        image_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        shape_input = tf.placeholder(tf.int32, shape=(2,))

        features = cls_preprocessing.preprocess_for_eval(image_input, out_shape, data_format=FLAGS.data_format, output_rgb=False)
        features = tf.expand_dims(features, axis=0)

        with tf.variable_scope(FLAGS.model_scope, default_name=None, values=[features], reuse=tf.AUTO_REUSE):
            model = cls_net.Model(
                        resnet_size=18,
                        bottleneck=False,
                        num_classes=2,
                        num_filters=64,
                        kernel_size=7,
                        conv_stride=2,
                        first_pool_size=3,
                        first_pool_stride=2,
                        block_sizes=cls_net._get_block_sizes(resnet_size=18),
                        block_strides=[1, 2, 2, 2],
                        resnet_version=2,
                        data_format=FLAGS.data_format,
                        dtype=tf.float32)
            logits,_ = model(features, training=False)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver.restore(sess, get_checkpoint())

            np_image = imread('./demo/test.jpg')
            res = sess.run([logits], feed_dict = {image_input : np_image, shape_input : np_image.shape[:-1]})

            print(res)
            # img_to_draw = draw_toolbox.bboxes_draw_on_img(np_image, labels_, scores_, bboxes_, thickness=2)
            # imsave('./demo/test_out.jpg', img_to_draw)

class Model():
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        with tf.Graph().as_default():
            out_shape = [FLAGS.train_image_size] * 2

            self.image_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

            features = cls_preprocessing.preprocess_for_eval(self.image_input, 
                        out_shape, data_format=FLAGS.data_format, output_rgb=False)
            features = tf.expand_dims(features, axis=0)

            with tf.variable_scope(FLAGS.model_scope, default_name=None, values=[features], reuse=tf.AUTO_REUSE):
                # model = cls_net.Model(
                model = cls_net.Att_Model(
                            resnet_size=14,
                            bottleneck=False,
                            num_classes=2,
                            num_filters=64,
                            kernel_size=7,
                            conv_stride=2,
                            first_pool_size=3,
                            first_pool_stride=2,
                            block_sizes=cls_net._get_block_sizes(resnet_size=14),
                            block_strides=[1, 2, 2, 2],
                            resnet_version=2,
                            data_format=FLAGS.data_format,
                            dtype=tf.float32)
                self.logits,_ = model(features, training=False)

            saver = tf.train.Saver()
            config = tf.ConfigProto()  
            config.gpu_options.allow_growth=True  
            self.sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            self.sess.run(init)

            saver.restore(self.sess, get_checkpoint())

    def process(self, image):
        # np_image = Image.fromarray(image)
        np_image =image
        res = self.sess.run([self.logits], feed_dict = {self.image_input : np_image})
        return res

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
