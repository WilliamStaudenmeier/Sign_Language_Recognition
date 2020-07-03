
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile
import sys
import os

from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None

os. chdir("C:\\Users\\William\Desktop\hand_gestures") 

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

COMPRESSION_TENSOR_NAME = 'pool_3/_reshape:0'
COMPRESSION_TENSOR_SIZE = 2048
MODEL_WIDTH = 299
MODEL_HEIGHT = 299
MODEL_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_inputPiclists(inputPicdir, testpercentage, validation_percentage):
  
    if not gfile.Exists(inputPicdir):
        print("Image directory '" + inputPicdir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(inputPicdir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == inputPicdir:
            continue

        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(inputPicdir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        
           
        
            
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        trainimages = []
        testimages = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
           
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
         
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                             (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testpercentage + validation_percentage):
                testimages.append(base_name)
            else:
                trainimages.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': trainimages,
            'testing': testimages,
            'validation': validation_images,
            }
    return result


def get_inputPicpath(inputPiclists, label_name, index, inputPicdir, category):
 
   
    if label_name not in inputPiclists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = inputPiclists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.', label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(inputPicdir, sub_dir, base_name)
    return full_path


def get_COMPRESSION_path(inputPiclists, label_name, index, COMPRESSION_dir, category):
   
    return get_inputPicpath(inputPiclists, label_name, index, COMPRESSION_dir,
                        category) + '.txt'


def create_inception_graph():
  
    with tf.Graph().as_default() as graph:
        model_filename = os.path.join(FLAGS.model_dir, 'classify_inputPicgraph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            COMPRESSION_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    COMPRESSION_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return graph, COMPRESSION_tensor, jpeg_data_tensor, resized_input_tensor


def run_COMPRESSION_on_image(sess, inputPicdata, inputPicdata_tensor, COMPRESSION_tensor):
   
    COMPRESSION_values = sess.run(
        COMPRESSION_tensor,
        {inputPicdata_tensor: inputPicdata})
    COMPRESSION_values = np.squeeze(COMPRESSION_values)
    return COMPRESSION_values


def maybe_download_and_extract():
  
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
  
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def write_list_of_floats_to_file(list_of_floats, file_path):
   
    s = struct.pack('d' * COMPRESSION_TENSOR_SIZE, *list_of_floats)
    with open(file_path, 'wb') as f:
        f.write(s)


def read_list_of_floats_from_file(file_path):
   
    with open(file_path, 'rb') as f:
        s = struct.unpack('d' * COMPRESSION_TENSOR_SIZE, f.read())
        return list(s)


COMPRESSION_path_2_COMPRESSION_values = {}


def create_COMPRESSION_file(COMPRESSION_path, inputPiclists, label_name, index,
                           inputPicdir, category, sess, jpeg_data_tensor,
                           COMPRESSION_tensor):
  
 
    inputPicpath = get_inputPicpath(inputPiclists, label_name, index,
                              inputPicdir, category)
    if not gfile.Exists(inputPicpath):
        tf.logging.fatal('File does not exist %s', inputPicpath)
    inputPicdata = gfile.FastGFile(inputPicpath, 'rb').read()
    try:
        COMPRESSION_values = run_COMPRESSION_on_image(
            sess, inputPicdata, jpeg_data_tensor, COMPRESSION_tensor)
    except:
        raise RuntimeError('Error during processing file %s' % inputPicpath)

    COMPRESSION_string = ','.join(str(x) for x in COMPRESSION_values)
    with open(COMPRESSION_path, 'w') as COMPRESSION_file:
        COMPRESSION_file.write(COMPRESSION_string)


def get_or_create_COMPRESSION(sess, inputPiclists, label_name, index, inputPicdir,
                             category, COMPRESSION_dir, jpeg_data_tensor,
                             COMPRESSION_tensor):
   
    label_lists = inputPiclists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(COMPRESSION_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    COMPRESSION_path = get_COMPRESSION_path(inputPiclists, label_name, index,
                                        COMPRESSION_dir, category)
    if not os.path.exists(COMPRESSION_path):
        create_COMPRESSION_file(COMPRESSION_path, inputPiclists, label_name, index,
                           inputPicdir, category, sess, jpeg_data_tensor,
                           COMPRESSION_tensor)
    with open(COMPRESSION_path, 'r') as COMPRESSION_file:
        COMPRESSION_string = COMPRESSION_file.read()
    did_hit_error = False
    try:
        COMPRESSION_values = [float(x) for x in COMPRESSION_string.split(',')]
    except ValueError:
      
        did_hit_error = True
    if did_hit_error:
        create_COMPRESSION_file(COMPRESSION_path, inputPiclists, label_name, index,
                           inputPicdir, category, sess, jpeg_data_tensor,
                           COMPRESSION_tensor)
        with open(COMPRESSION_path, 'r') as COMPRESSION_file:
            COMPRESSION_string = COMPRESSION_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        COMPRESSION_values = [float(x) for x in COMPRESSION_string.split(',')]
    return COMPRESSION_values


def cache_COMPRESSIONs(sess, inputPiclists, inputPicdir, COMPRESSION_dir,
                      jpeg_data_tensor, COMPRESSION_tensor):
  
    how_many_COMPRESSIONs = 0
    ensure_dir_exists(COMPRESSION_dir)
    for label_name, label_lists in inputPiclists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_COMPRESSION(sess, inputPiclists, label_name, index,
                                 inputPicdir, category, COMPRESSION_dir,
                                 jpeg_data_tensor, COMPRESSION_tensor)

                how_many_COMPRESSIONs += 1
                if how_many_COMPRESSIONs % 100 == 0:
                    print(str(how_many_COMPRESSIONs) + ' COMPRESSION files created.')


def get_random_cached_COMPRESSIONs(sess, inputPiclists, how_many, category,
                                  COMPRESSION_dir, inputPicdir, jpeg_data_tensor,
                                  COMPRESSION_tensor):
 
    class_count = len(inputPiclists.keys())
    COMPRESSIONs = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of COMPRESSIONs.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(inputPiclists.keys())[label_index]
            inputPicindex = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            inputPicname = get_inputPicpath(inputPiclists, label_name, inputPicindex,
                                  inputPicdir, category)
            COMPRESSION = get_or_create_COMPRESSION(sess, inputPiclists, label_name,
                                            inputPicindex, inputPicdir, category,
                                            COMPRESSION_dir, jpeg_data_tensor,
                                            COMPRESSION_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            COMPRESSIONs.append(COMPRESSION)
            ground_truths.append(ground_truth)
            filenames.append(inputPicname)
    else:
   
        for label_index, label_name in enumerate(inputPiclists.keys()):
            for inputPicindex, inputPicname in enumerate(
                inputPiclists[label_name][category]):
                inputPicname = get_inputPicpath(inputPiclists, label_name, inputPicindex,
                                    inputPicdir, category)
                COMPRESSION = get_or_create_COMPRESSION(sess, inputPiclists, label_name,
                                              inputPicindex, inputPicdir, category,
                                              COMPRESSION_dir, jpeg_data_tensor,
                                              COMPRESSION_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                COMPRESSIONs.append(COMPRESSION)
                ground_truths.append(ground_truth)
                filenames.append(inputPicname)
    return COMPRESSIONs, ground_truths, filenames


def get_random_distorted_COMPRESSIONs(
    sess, inputPiclists, how_many, category, inputPicdir, input_jpeg_tensor,
    distorted_image, resized_input_tensor, COMPRESSION_tensor):

    class_count = len(inputPiclists.keys())
    COMPRESSIONs = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(inputPiclists.keys())[label_index]
        inputPicindex = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        inputPicpath = get_inputPicpath(inputPiclists, label_name, inputPicindex, inputPicdir,
                                category)
        if not gfile.Exists(inputPicpath):
            tf.logging.fatal('File does not exist %s', inputPicpath)
        jpeg_data = gfile.FastGFile(inputPicpath, 'rb').read()
      
        distorted_inputPicdata = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
        COMPRESSION = run_COMPRESSION_on_image(sess, distorted_inputPicdata,
                                         resized_input_tensor,
                                         COMPRESSION_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        COMPRESSIONs.append(COMPRESSION)
        ground_truths.append(ground_truth)
    return COMPRESSIONs, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
   
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness):
    

    

    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_DEPTH)
    decoded_inputPicas_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_inputPic4d = tf.expand_dims(decoded_inputPicas_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=1.0,
                                         maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, MODEL_WIDTH)
    precrop_height = tf.multiply(scale_value, MODEL_HEIGHT)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_inputPic4d,
                                              precrop_shape_as_int)
    precropped_inputPic3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_inputPic3d,
                                 [MODEL_HEIGHT, MODEL_WIDTH,
                                  MODEL_DEPTH])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                       minval=brightness_min,
                                       maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def variable_summaries(var):
   
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_trainops(class_count, final_tensor_name, COMPRESSION_tensor):
 
    with tf.name_scope('input'):
        COMPRESSION_input = tf.placeholder_with_default(
                COMPRESSION_tensor, shape=[None, COMPRESSION_TENSOR_SIZE],
                name='COMPRESSIONInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  
    layer_name = 'final_trainops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([COMPRESSION_TENSOR_SIZE, class_count],
                                          stddev=0.001)

            layer_weights = tf.Variable(initial_value, name='final_weights')

            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(COMPRESSION_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, COMPRESSION_input, ground_truth_input,
              final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
   
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                    prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def main(_):
   
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)


    maybe_download_and_extract()
    graph, COMPRESSION_tensor, jpeg_data_tensor, resized_inputPictensor = (
            create_inception_graph())

   
    inputPiclists = create_inputPiclists(FLAGS.inputPicdir, FLAGS.testpercentage,
                                   FLAGS.validation_percentage)
    class_count = len(inputPiclists.keys())
    if class_count == 0:
        print('No valid folders of images found at ' + FLAGS.inputPicdir)
        return -1
    if class_count == 1:
        print('Only one valid folder of images found at ' + FLAGS.inputPicdir +
              ' - multiple classes are needed for classification.')
        return -1

   
    do_distort_images = should_distort_images(
            FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
            FLAGS.random_brightness)

    with tf.Session(graph=graph) as sess:

        if do_distort_images:
           
            (distorted_jpeg_data_tensor,
             distorted_inputPictensor) = add_input_distortions(
                     FLAGS.flip_left_right, FLAGS.random_crop,
                     FLAGS.random_scale, FLAGS.random_brightness)
        else:
        
            cache_COMPRESSIONs(sess, inputPiclists, FLAGS.inputPicdir,
                        FLAGS.COMPRESSION_dir, jpeg_data_tensor,
                        COMPRESSION_tensor)


        (train_step, cross_entropy, COMPRESSION_input, ground_truth_input,
         final_tensor) = add_final_trainops(len(inputPiclists.keys()),
                                            FLAGS.final_tensor_name,
                                            COMPRESSION_tensor)

      
        evaluation_step, prediction = add_evaluation_step(
                final_tensor, ground_truth_input)

    
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)

        validation_writer = tf.summary.FileWriter(
                FLAGS.summaries_dir + '/validation')

      
        init = tf.global_variables_initializer()
        sess.run(init)

     
        for i in range(FLAGS.how_many_trainsteps):
            
            if do_distort_images:
                (train_COMPRESSIONs,
                 train_ground_truth) = get_random_distorted_COMPRESSIONs(
                         sess, inputPiclists, FLAGS.train_batch_size, 'training',
                         FLAGS.inputPicdir, distorted_jpeg_data_tensor,
                         distorted_inputPictensor, resized_inputPictensor, COMPRESSION_tensor)
            else:
                (train_COMPRESSIONs,
                 train_ground_truth, _) = get_random_cached_COMPRESSIONs(
                         sess, inputPiclists, FLAGS.train_batch_size, 'training',
                         FLAGS.COMPRESSION_dir, FLAGS.inputPicdir, jpeg_data_tensor,
                         COMPRESSION_tensor)
          

            train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={COMPRESSION_input: train_COMPRESSIONs,
                               ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

          
            is_last_step = (i + 1 == FLAGS.how_many_trainsteps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                        [evaluation_step, cross_entropy],
                        feed_dict={COMPRESSION_input: train_COMPRESSIONs,
                                   ground_truth_input: train_ground_truth})
                validation_COMPRESSIONs, validation_ground_truth, _ = (
                        get_random_cached_COMPRESSIONs(
                                sess, inputPiclists, FLAGS.validation_batch_size, 'validation',
                                FLAGS.COMPRESSION_dir, FLAGS.inputPicdir, jpeg_data_tensor,
                                COMPRESSION_tensor))
                
                validation_summary, validation_accuracy = sess.run(
                        [merged, evaluation_step],
                        feed_dict={COMPRESSION_input: validation_COMPRESSIONs,
                                   ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                print('Step: %d, Train accuracy: %.4f%%, Cross entropy: %f, Validation accuracy: %.1f%% (N=%d)' % (i,
                        train_accuracy * 100, cross_entropy_value, validation_accuracy * 100, len(validation_COMPRESSIONs)))

   
        test_COMPRESSIONs, test_ground_truth, test_filenames = (
                get_random_cached_COMPRESSIONs(sess, inputPiclists, FLAGS.test_batch_size,
                                      'testing', FLAGS.COMPRESSION_dir,
                                      FLAGS.inputPicdir, jpeg_data_tensor,
                                      COMPRESSION_tensor))
        test_accuracy, predictions = sess.run(
                [evaluation_step, prediction],
                feed_dict={COMPRESSION_input: test_COMPRESSIONs,
                           ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%% (N=%d)' % (
                test_accuracy * 100, len(test_COMPRESSIONs)))

        if FLAGS.print_misclassified_test_images:
            print('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    print('%70s  %s' % (test_filename,
                              list(inputPiclists.keys())[predictions[i]]))

        # Write out the trained graph and labels with the weights stored as
        # constants.
        output_graph_def = graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
        with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(inputPiclists.keys()) + '\n')





####




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inputPicdir',
        type=str,
        default='dataset',
        help='Path to folders of labeled images.'
        )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='logs/output_graph.pb',
        help='Where to save the trained graph.'
        )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='logs/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
        )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='logs/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
        )
    parser.add_argument(
        '--how_many_trainsteps',
        type=int,
        default=5000,
        help='How many training steps to run before ending.'
        )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
        )
    parser.add_argument(
        '--testpercentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
        )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
        )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=100,
        help='How often to evaluate the training results.'
        )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='How many images to train on at a time.'
        )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1
        )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100
        )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=False,
        action='store_true'
        )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='logs/imagenet'
        )
    parser.add_argument(
        '--COMPRESSION_dir',
        type=str,
        default='/tmp/COMPRESSION'
        )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result'
        )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        action='store_true'
        )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0
     
        )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0
        )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0
        )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

###




os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def predict(inputPicdata):

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': inputPicdata})

   
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs/output_labels.txt")]

with tf.gfile.FastGFile("logs/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the inputPicdata as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    c = 0

    cap = cv2.VideoCapture(0)

    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''
    
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        
        if ret:
            x1, y1, x2, y2 = 100, 100, 300, 300
            img_cropped = img[y1:y2, x1:x2]

            c += 1
            inputPicdata = cv2.imencode('.jpg', img_cropped)[1].tostring()
            
            a = cv2.waitKey(1) # waits to see if `esc` is pressed
            
            if i == 4:
                res_tmp, score = predict(inputPicdata)
                res = res_tmp
                i = 0
                if mem == res:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive == 2 and res not in ['nothing']:
                    if res == 'space':
                        sequence += ' '
                    elif res == 'del':
                        sequence = sequence[:-1]
                    else:
                        sequence += res
                    consecutive = 0
            i += 1
            cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
            cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            mem = res
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.imshow("img", img)
            img_sequence = np.zeros((200,1200,3), np.uint8)
            cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow('sequence', img_sequence)
            
            if a == 27: 
                break

# IN ORDER TO STOP THE VIDEO, CLICK the ESCAPE KEY
cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()


