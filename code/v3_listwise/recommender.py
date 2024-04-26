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

class Recommender(tfrs.Model):

  def __init__(self, portfolios, loss):
    super().__init__()

    embedding_dimension = 32,
    self.loss = loss
    
    self.portfolios = portfolios

    self.items_ids = self.portfolios.batch(10000).map(lambda x: x["STOCKCODE"])
    self.unique_item_ids = np.unique(np.concatenate(list(self.items_ids)))

    self.item_GICS = self.portfolios.batch(10000).map(lambda x: x["GICS"])
    self.unique_item_gics = np.unique(np.concatenate(list(self.item_GICS)))

    self.item_names = portfolios.batch(10000).map(lambda x: x["STOCKNAME"])
    self.unique_item_names = np.unique(np.concatenate(list(self.item_names)))
    
    self.user_ids = self.portfolios.batch(10000).map(lambda x: x["CDSACCNO"])
    self.unique_user_ids = np.unique(np.concatenate(list(self.user_ids)))


    # Compute embeddings for users.
    self.user_embeddings = UserModel(
        unique_user_ids = self.unique_user_ids
    )

    # self.item_embeddings = ItemModel(
    #     unique_item_ids = unique_movie_titles
    # )

    self.item_embeddings = ItemModel(
        unique_item_ids = self.unique_item_ids,
        unique_item_gics = self.unique_item_gics,
        unique_item_names = self.unique_item_names
    )

    self.score_model = tf.keras.Sequential([
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(1)
    ])

    self.task = tfrs.tasks.Ranking(
      loss=loss,
      metrics=[
        tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
        tf.keras.metrics.RootMeanSquaredError()
      ]
    )

  def call(self, features):

    user_embeddings = self.user_embeddings(features["CDSACCNO"])

    item_embeddings = self.item_embeddings((features["STOCKCODE"], features['GICS'], features['STOCKNAME'])) #, features['STOCKNAME']

    list_length = features["STOCKCODE"].shape[1]
    user_embedding_repeated = tf.repeat(
        tf.expand_dims(user_embeddings, 1), [list_length], axis=1)

    print(user_embedding_repeated.shape,' | ' ,item_embeddings.shape)
    concatenated_embeddings = tf.concat(
        [user_embedding_repeated, item_embeddings], 2)

    return self.score_model(concatenated_embeddings)

  def compute_loss(self, features, training=False):
    labels = features.pop("RATING")

    scores = self(features)

    return self.task(
        labels=labels,
        predictions=tf.squeeze(scores, axis=-1),
    )