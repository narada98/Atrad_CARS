import os

import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

from recommender import Recommender
from user_embedding import UserModel
from item_embedding import ItemModel

# from utils import dataset_to_dataframe
from datetime import datetime

base_loc = r'D:\dev work\recommender systems\ATRAD_CARS'

train_ds = tf.data.Dataset.load("D:/dev work/recommender systems/Atrad_CARS/data/train").cache() #data\ratings_train
test_ds = tf.data.Dataset.load("D:/dev work/recommender systems/Atrad_CARS/data/test").cache()
portfolios = tf.data.Dataset.load("D:/dev work/recommender systems/Atrad_CARS/data/portfolios_tfds").cache()

model = Recommender(
    use_timestamp = True,
    portfolios = portfolios
    )

log_dir = os.path.join(base_loc ,"logs/fit/retriever/" + "retriever" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# log_dir = "../../logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    embeddings_freq = 1,
    write_images = True)

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.3))

train_ds = train_ds.shuffle(1000).batch(128) #.cache()
test_ds = test_ds.batch(128)

history = model.fit(
    train_ds, 
    epochs=3, 
    verbose = 1,
    validation_data=test_ds,
    validation_freq=1,
    callbacks=[tensorboard_callback]
    )

print(history.history.keys())


#save model
base = r'D:\dev work\recommender systems\ATRAD_CARS\model_weights\{}'.format(datetime.now().strftime("%Y_%m_%d"))

if not os.path.exists(base):
    os.makedirs(base)

# model_name = 'tf_retrival_{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M"))
model_name = 'tf_retrival_opt'
save_path = os.path.join(base,model_name)

model.save_weights(save_path)

print()
print("saved model @ : {}".format(save_path))
