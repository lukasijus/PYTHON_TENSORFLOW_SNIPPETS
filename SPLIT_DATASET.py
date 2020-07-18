import collections
import io
import math
import os
import random
from six.moves import urllib
import shutil
import glob
import tensorflow as tf

FLOWERS_DIR = './dataset/flower_photos'
TRAIN_FRACTION = 0.8
RANDOM_SEED = 2018


def download_images():
  """If the images aren't already downloaded, save them to FLOWERS_DIR."""
  if not os.path.exists(FLOWERS_DIR):
    DOWNLOAD_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
    print('Downloading flower images from %s...' % DOWNLOAD_URL)
    urllib.request.urlretrieve(DOWNLOAD_URL, 'flower_photos.tgz')
  print('Flower photos are located in %s' % FLOWERS_DIR)

def extract_all(archives, extract_path):
  for filename in archives:
    shutil.unpack_archive(filename, extract_path)

def archives_names(path, extension):
  glob_path = os.path.join(path, extension)
  names = glob.glob(glob_path)
  return names

path = os.path.abspath('')
extension = '*.tgz'
extract_path = os.path.join(path, 'dataset/natural_images')
ROOT_DIR = os.path.join(extract_path, 'natural_images')

# names = archives_names(path, extension)
# extract_all(names, extract_path)

def make_train_and_test_sets(ROOT_DIR = ROOT_DIR, TRAIN_FRACTION = TRAIN_FRACTION, makefolders = False):
  """Split the data into train and test sets and get the label classes."""
  train_examples, test_examples = [], []
  shuffler = random.Random(RANDOM_SEED)
  is_root = True
  for (dirname, subdirs, filenames) in tf.io.gfile.walk(ROOT_DIR):
    # The root directory gives us the classes
    if is_root:
      subdirs = sorted(subdirs)
      classes = collections.OrderedDict(enumerate(subdirs))
      label_to_class = dict([(x, i) for i, x in enumerate(subdirs)])
      is_root = False
    # The sub directories give us the image files for training.
    else:
      filenames.sort()
      shuffler.shuffle(filenames)
      full_filenames = [os.path.join(dirname, f) for f in filenames]
      label = dirname.split('/')[-1]
      label_class = label_to_class[label]
      # An example is the image file and it's label class.
      examples = list(zip(full_filenames, [label_class] * len(filenames)))

      num_train = int(len(filenames) * TRAIN_FRACTION)
      train_examples.extend(examples[:num_train])
      test_examples.extend(examples[num_train:])


  shuffler.shuffle(train_examples)
  shuffler.shuffle(test_examples)

  copy_train_dir = '/Users/macbook/Software/PYTHON_TENSORFLOW_SNIPPETS/dataset/natural_images/train'
  copy_test_dir = '/Users/macbook/Software/PYTHON_TENSORFLOW_SNIPPETS/dataset/natural_images/test'
  dir_class_train_list = []
  dir_class_test_list = []
  if(makefolders):
    for i in range(len(classes)):
      dir_class_train = os.path.join(copy_train_dir, classes[i])
      dir_class_test = os.path.join(copy_test_dir, classes[i])
      if not os.path.exists(dir_class_train):
        os.makedirs(dir_class_train)
        print('Folder ', dir_class_train, ' created succesfully!')
      else:
        print('Folder ', dir_class_train, ' already exists')
      if not os.path.exists(dir_class_test):
        os.makedirs(dir_class_test)
        print('Folder ', dir_class_test, ' created succesfully!')
      else:
        print('Folder ', dir_class_test, ' already exists')
      tuple_train = (dir_class_train, i)
      tuple_test = (dir_class_test, i)
      dir_class_train_list.append(tuple_train)
      dir_class_test_list.append(tuple_test)
    for example in train_examples:
      file_name = example[0]
      class_number = example[1]
      class_name = classes[class_number]
      dst = os.path.join(copy_train_dir, class_name)
      shutil.copy(file_name, dst)
    for example in test_examples:
      file_name = example[0]
      class_number = example[1]
      class_name = classes[class_number]
      dst = os.path.join(copy_test_dir, class_name)
      shutil.copy(file_name, dst)
  return train_examples, test_examples, classes, copy_train_dir, copy_test_dir


# train_examples, test_examples, classes, train_dir, test_dir = make_train_and_test_sets(makefolders=True)
