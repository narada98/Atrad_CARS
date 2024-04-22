import tensorflow as tf
import numpy as np
import tensorflow_recommenders as tfrs

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
    use_timestamp,
    portfolios
    ):

    super().__init__()

    self.use_timestamp = use_timestamp
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

    self.timestamps = np.concatenate(list(self.portfolios.map(lambda x: x["UNIX_TS"]).batch(100)))

    self.max_timestamp = self.timestamps.max()
    self.min_timestamp = self.timestamps.min()

    self.timestamp_buckets = np.linspace(
        self.min_timestamp, self.max_timestamp, num=1000,
    )

    self.item_model = ItemModel(
      unique_item_ids = self.unique_item_ids,
      unique_item_names = self.unique_item_names,
      unique_item_gics = self.unique_item_gics
    )

    self.user_model = UserModel(
      use_timestamp = self.use_timestamp,
      unique_user_ids = self.unique_user_ids, 
      timestamps = self.timestamps, 
      timestamp_buckets = self.timestamp_buckets
    )

    self.retrieval_metrics = tfrs.metrics.FactorizedTopK(
      candidates= self.portfolios.batch(128).map(lambda x:
        generate_embedding(x, self.item_model))
    )

    self.task = tfrs.tasks.Retrieval(
      metrics = self.retrieval_metrics
    )

  def compute_loss(self, features: [Text, tf.Tensor], training=False) -> tf.Tensor:
    user_embeddings = self.user_model(
      (
        features['CDSACCNO'],
        features['UNIX_TS']
       )
    )

    item_embeddings = self.item_model(
      (
        features['STOCKCODE'],
        features['STOCKNAME'],
        features['GICS']
      )
    )
    
    return self.task(
      query_embeddings = user_embeddings,
      candidate_embeddings = item_embeddings
      )