import os

import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs

from ranker_recommender import Ranker

# from utils import dataset_to_dataframe
from datetime import datetime

base_loc = r'D:\dev work\recommender systems\ATRAD_CARS'

portfolios = tf.data.Dataset.load(r"D:\dev work\recommender systems\Atrad_CARS\data\portfolios_v2\portfolios").cache()

train_list_ds = tf.data.Dataset.load(r"D:\dev work\recommender systems\Atrad_CARS\data\portfolios_v2\ranker_train").cache()
test_list_ds = tf.data.Dataset.load(r"D:\dev work\recommender systems\Atrad_CARS\data\portfolios_v2\ranker_test").cache()

items_ids = portfolios.batch(10000).map(lambda x: x["STOCKCODE"])
item_names = portfolios.batch(10000).map(lambda x: x["STOCKNAME"])
item_GICS = portfolios.batch(10000).map(lambda x: x["GICS"])

user_ids = portfolios.batch(10000).map(lambda x: x["CDSACCNO"])

unique_item_ids = np.unique(np.concatenate(list(items_ids)))
unique_item_names = np.unique(np.concatenate(list(item_names)))
unique_item_gics = np.unique(np.concatenate(list(item_GICS)))

unique_user_ids = np.unique(np.concatenate(list(user_ids)))

# need these to initialize timestamp embedding layers in future steps

model = Ranker(
    # use_timestamp = True,
    loss = tf.keras.losses.MeanSquaredError(),
    portfolios = portfolios
    )


log_dir = os.path.join(base_loc ,"logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# log_dir = "../../logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    embeddings_freq = 1,
    write_images = True)

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

train_list_ds = train_list_ds.shuffle(10000).batch(256)

model.fit(
    train_list_ds, 
    epochs=20, 
    verbose = 1,
    callbacks=[tensorboard_callback]
    )

#save model
base = r'D:\dev work\recommender systems\ATRAD_CARS\model_weights\{}'.format(datetime.now().strftime("%Y_%m_%d"))

if not os.path.exists(base):
    os.makedirs(base)

model_name = 'tf_listwise_ranking_{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M"))
save_path = os.path.join(base,model_name)

model.save_weights(save_path)

print()
print("saved model @ : {}".format(save_path))

print()
print("*** TESTING ***")
test_list_ds = test_list_ds.batch(128)
model = model.evaluate(test_list_ds, return_dict=True)
print("NDCG of the MSE Model: {:.4f}".format(model["ndcg_metric"]))
