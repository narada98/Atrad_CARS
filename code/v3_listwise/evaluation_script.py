import numpy as np
import tensorflow as tf

from recommender import Recommender

test_list_ds = tf.data.Dataset.load("D:/dev work/recommender systems/Atrad_CARS/data/test_lists_ds").cache()
portfolios = tf.data.Dataset.load("D:/dev work/recommender systems/Atrad_CARS/data/portfolios_tfds").cache()

items_ids = portfolios.batch(10000).map(lambda x: x["STOCKCODE"])
unique_item_ids = np.unique(np.concatenate(list(items_ids)))

item_names = portfolios.batch(10000).map(lambda x: x["STOCKNAME"])
unique_item_names = np.unique(np.concatenate(list(item_names)))

item_GICS = portfolios.batch(10000).map(lambda x: x["GICS"])
unique_item_gics = np.unique(np.concatenate(list(item_GICS)))

user_ids = portfolios.batch(10000).map(lambda x: x["CDSACCNO"])
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

model = Recommender(
    loss = tf.keras.losses.MeanSquaredError(),
    portfolios = portfolios
    )

model.load_weights(r"D:\dev work\recommender systems\Atrad_CARS\model_weights\2024_04_25\tf_listwise_ranking_2024_04_25_12_49")

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.evaluate(test_list_ds.shuffle(100_000).batch(1024), return_dict=True)

