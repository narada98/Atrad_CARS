import numpy as np
import tensorflow as tf

from recommender import Recommender

test_ds = tf.data.Dataset.load("D:/dev work/recommender systems/Atrad_CARS/data/test").cache()

portfolios = tf.data.Dataset.load("D:/dev work/recommender systems/Atrad_CARS/data/portfolios_tfds").cache()

items_ids = portfolios.batch(10000).map(lambda x: x["STOCKCODE"])
item_names = portfolios.batch(10000).map(lambda x: x["STOCKNAME"])
item_GICS = portfolios.batch(10000).map(lambda x: x["GICS"])

user_ids = portfolios.batch(10000).map(lambda x: x["CDSACCNO"])

unique_item_ids = np.unique(np.concatenate(list(items_ids)))
unique_item_names = np.unique(np.concatenate(list(item_names)))
unique_item_gics = np.unique(np.concatenate(list(item_GICS)))

unique_user_ids = np.unique(np.concatenate(list(user_ids)))

# need these to initialize timestamp embedding layers in future steps

timestamps = np.concatenate(list(portfolios.map(lambda x: x["UNIX_TS"]).batch(100)))

max_timestamp = timestamps.max()
min_timestamp = timestamps.min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000,
)

loaded_model = Recommender(
    use_timestamp = True,
    unique_user_ids = unique_user_ids, 
    timestamps = timestamps, 
    timestamp_buckets = timestamp_buckets,
    unique_item_ids = unique_item_ids,
    unique_item_names = unique_item_names,
    unique_item_gics = unique_item_gics
)

loaded_model.load_weights("D:/dev work/recommender systems/ATRAD_CARS/model_weights/2024_04_09_16_07/tf_rating_2024_04_09_16_07")

loaded_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

loaded_model.evaluate(test_ds.shuffle(100_000).batch(1024), return_dict=True)

