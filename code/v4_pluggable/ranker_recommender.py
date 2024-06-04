import tensorflow as tf
import numpy as np
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr


def global_average_mean(x):
  """Custom layer to perform global average mean pooling."""
  axis = -2  # Reduce mean along the last dimension
  return tf.reduce_mean(x, axis=axis)

def reshaper(x):
    shape = (-1,10,1)
    return tf.reshape(x, shape)

class ItemModel(tf.keras.Model):
    def __init__(
        self,
        unique_item_ids,
        unique_item_text_contex_1,
        unique_item_long_text_contex_1

        ):
        super().__init__()

        self.max_tokens = 10000
        self.unique_item_ids = unique_item_ids
        self.unique_item_text_contex_1 = unique_item_text_contex_1
        self.unique_item_long_text_contex_1 = unique_item_long_text_contex_1

        self.embed_item_id = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = self.unique_item_ids,
                mask_token =None
            ),
            tf.keras.layers.Embedding(
                input_dim = len(self.unique_item_ids)+1,
                output_dim = 16 #32
            )
        ])

        self.embed_item_text_contex_1 = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = unique_item_text_contex_1,
                mask_token = None
            ),
            tf.keras.layers.Embedding(
                input_dim = len(unique_item_text_contex_1)+1,
                output_dim = 16 #len(unique_item_text_contex_1)
            )
        ])

        self.long_text_textvectorizer = tf.keras.layers.TextVectorization(
            max_tokens = self.max_tokens,
        )
        self.long_text_textvectorizer.adapt(self.unique_item_long_text_contex_1)

        self.embed_long_text_contex_1 = tf.keras.Sequential([
            tf.keras.layers.Lambda(reshaper),

            self.long_text_textvectorizer,

            tf.keras.layers.Embedding(
                input_dim = self.max_tokens,
                output_dim = 32,
                mask_zero = True
            ),

            tf.keras.layers.Lambda(global_average_mean)
        ])

    def call(self, inputs):

        item_id,  contex_1, contex_2 = inputs

        return tf.concat([
            self.embed_item_id(item_id),
            self.embed_item_text_contex_1(contex_1),
            self.embed_long_text_contex_1(contex_2)
        ],
        axis = 2)


class UserModel(tf.keras.Model):
    def __init__(
        self,
        unique_user_ids, 
        
        ):

        super().__init__()

        self.unique_user_ids = unique_user_ids

        self.embed_user_id = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = self.unique_user_ids,
                mask_token = None
            ),
            tf.keras.layers.Embedding(
                input_dim = len(self.unique_user_ids)+1,
                output_dim = 32
            )
        ])
        
    def call(self, inputs):

        (user_id) = inputs

        return self.embed_user_id(user_id)


class Ranker(tfrs.Model):

  def __init__(self, portfolios, loss, feature_mapper):
    super().__init__()

    embedding_dimension = 32,
    self.loss = loss
    
    self.portfolios = portfolios

    self.items_ids = self.portfolios.batch(10000).map(lambda x: x["STOCKCODE"])
    self.unique_item_ids = np.unique(np.concatenate(list(self.items_ids)))

    self.item_GICS = self.portfolios.batch(10000).map(lambda x: x["GICS"])
    self.unique_item_text_contex_1 = np.unique(np.concatenate(list(self.item_GICS)))

    self.item_names = portfolios.batch(10000).map(lambda x: x["STOCKNAME"])
    self.unique_item_long_text_contex_1 = np.unique(np.concatenate(list(self.item_names)))
    
    self.user_ids = self.portfolios.batch(10000).map(lambda x: x["CDSACCNO"])
    self.unique_user_ids = np.unique(np.concatenate(list(self.user_ids)))


    # Compute embeddings for users.
    self.user_embeddings = UserModel(
        unique_user_ids = self.unique_user_ids
    )

    self.item_embeddings = ItemModel(
        unique_item_ids = self.unique_item_ids,
        unique_item_text_contex_1 = self.unique_item_text_contex_1,
        unique_item_long_text_contex_1 = self.unique_item_long_text_contex_1
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