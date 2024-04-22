import tensorflow as tf
import tensorflow_recommenders as tfrs

from typing import Dict, Text

from rating_model import RatingModel

class Recommender(tfrs.models.Model):

  def __init__(
    self,
    use_timestamp,
    unique_user_ids, 
    timestamps, 
    timestamp_buckets,
    unique_item_ids,
    unique_item_names,
    unique_item_gics
    ):

    super().__init__()

    self.use_timestamp = use_timestamp
    self.unique_user_ids = unique_user_ids 
    self.timestamps = timestamps
    self.timestamp_buckets = timestamp_buckets
    self.unique_item_ids = unique_item_ids
    self.unique_item_names = unique_item_names
    self.unique_item_gics = unique_item_gics

    self.rating_model: tf.keras.Model = RatingModel(
      use_timestamp = self.use_timestamp,
      unique_user_ids = self.unique_user_ids,
      timestamps = self.timestamps,
      timestamp_buckets = self.timestamp_buckets,
      unique_item_ids = self.unique_item_ids,
      unique_item_names = self.unique_item_names,
      unique_item_gics = self.unique_item_gics
      )

    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    )

  def call(self, features) -> tf.Tensor:
    return self.rating_model(
        (features["CDSACCNO"],
        features['UNIX_TS'],
        features["STOCKCODE"],
        features["STOCKNAME"],
        features["GICS"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("RATING")

    rating_predictions = self(features)

    return self.task(labels=labels, predictions=rating_predictions)