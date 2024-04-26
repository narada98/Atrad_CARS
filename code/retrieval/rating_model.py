import tensorflow as tf

from user_embedding import UserModel
from item_embedding import ItemModel


class RatingModel(tf.keras.Model):

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

    # embedding_dimension = 32
    self.use_timestamp = use_timestamp
    self.unique_user_ids = unique_user_ids 
    self.timestamps = timestamps
    self.timestamp_buckets = timestamp_buckets
    self.unique_item_ids = unique_item_ids
    self.unique_item_names = unique_item_names
    self.unique_item_gics = unique_item_gics

    self.user_embeddings = UserModel(
      use_timestamp = self.use_timestamp,
      unique_user_ids = self.unique_user_ids, 
      timestamps = self.timestamps, 
      timestamp_buckets = self.timestamp_buckets
      )

    self.item_embeddings = ItemModel(
      unique_item_ids = self.unique_item_ids,
      unique_item_names = self.unique_item_names,
      unique_item_gics = self.unique_item_gics
      )

    # Rating Model
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])

  def call(self, inputs):

    user_id, timestamp, item_id, item_name, item_gics = inputs

    user_embedding = self.user_embeddings((user_id,timestamp))
    item_embedding = self.item_embeddings((item_id, item_name, item_gics))

    return self.ratings(tf.concat([user_embedding, item_embedding], axis=1))