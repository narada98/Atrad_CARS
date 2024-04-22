import tensorflow as tf
import numpy as np
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

from typing import Dict, Text

from item_embedding import ItemModel
from user_embedding import UserModel

def generate_embedding(item, item_model):
  
      item_id = item['STOCKCODE'] 
      item_name = item['STOCKNAME']
      item_gics = item['GICS']

      embedding = item_model([item_id, item_name, item_gics])
      return embedding

class Recommender(tfrs.models.Model):

  def __init__(
    self,
    # use_timestamp,
    portfolios
    ):

    super().__init__()

    # self.use_timestamp = use_timestamp
    self.portfolios = portfolios

    self.items_ids = self.portfolios.batch(10000).map(lambda x: x["STOCKCODE"])
    self.item_names = self.portfolios.batch(10000).map(lambda x: x["STOCKNAME"])
    self.item_GICS = self.portfolios.batch(10000).map(lambda x: x["GICS"])

    self.user_ids = self.portfolios.batch(10000).map(lambda x: x["CDSACCNO"])

    self.unique_item_ids = np.unique(np.concatenate(list(self.items_ids)))
    self.unique_item_names = np.unique(np.concatenate(list(self.item_names)))
    self.unique_item_gics = np.unique(np.concatenate(list(self.item_GICS)))

    self.unique_user_ids = np.unique(np.concatenate(list(self.user_ids)))

    # need these to initialize timestamp embedding layers in future steps

    # self.timestamps = np.concatenate(list(self.portfolios.map(lambda x: x["UNIX_TS"]).batch(100)))

    # self.max_timestamp = self.timestamps.max()
    # self.min_timestamp = self.timestamps.min()

    # self.timestamp_buckets = np.linspace(
    #     self.min_timestamp, self.max_timestamp, num=1000,
    # )

    self.item_model = ItemModel(
      unique_item_ids = self.unique_item_ids,
      unique_item_names = self.unique_item_names,
      unique_item_gics = self.unique_item_gics
    )

    self.user_model = UserModel(
      # use_timestamp = self.use_timestamp,
      unique_user_ids = self.unique_user_ids, 
      # timestamps = self.timestamps, 
      # timestamp_buckets = self.timestamp_buckets
    )

    self.score_model = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
    ])

    self.ranking_task = tfrs.tasks.Ranking(
      loss=tfr.keras.losses.ListMLELoss(),
      metrics=[
        tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
        tf.keras.metrics.RootMeanSquaredError()
      ]
    )

  def call(self, features) -> tf.Tensor:
    user_embeddings = self.user_model(
      (
        features['CDSACCNO'],
        # features['UNIX_TS']
       )
    )

    item_embeddings = self.item_model(
      (
        features['STOCKCODE'],
        # features['STOCKNAME'],
        features['GICS']
      )
    )

    # print('**********',features["STOCKCODE"].shape, '**********')

    list_length = features["STOCKCODE"].shape[1] # changed this to 0. was 1

    print('**********',user_embeddings.shape, '**********')

    user_embedding_repeated = tf.repeat(
        tf.expand_dims(user_embeddings, 1), [list_length], axis=1)

    print('**********',user_embedding_repeated.shape, '**********')
    print('**********',item_embeddings.shape, '**********')

    concatenated_embeddings = tf.concat(
        # [user_embedding_repeated, tf.expand_dims(item_embeddings, 0)], 2)
        [user_embedding_repeated, item_embeddings], 2)

    return self.score_model(concatenated_embeddings)
    
  def compute_loss(self, features, training = False):

    labels = features.pop('RATING')

    scores = self(features)

    return self.ranking_task(
      labels = labels,
      predictions = tf.squeeze(scores, axis = 1)
      )