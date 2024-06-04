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
        unique_item_gics,
        unique_item_names

        ):
        super().__init__()

        self.max_tokens = 10000
        self.unique_item_ids = unique_item_ids
        self.unique_item_gics = unique_item_gics
        self.unique_item_names = unique_item_names

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

        self.embed_items_gics = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = unique_item_gics,
                mask_token = None
            ),
            tf.keras.layers.Embedding(
                input_dim = len(unique_item_gics)+1,
                output_dim = 16 #len(unique_item_gics)
            )
        ])

        self.textvectorizer = tf.keras.layers.TextVectorization(
            max_tokens = self.max_tokens,
            # ragged = True
        )

        self.embed_item_name = tf.keras.Sequential([

            # tf.keras.layers.Reshape((-1,5,1)),
            tf.keras.layers.Lambda(reshaper),

            self.textvectorizer,

            tf.keras.layers.Embedding(
                input_dim = self.max_tokens,
                output_dim = 32,
                mask_zero = True
            ),

            tf.keras.layers.Lambda(global_average_mean)
            # tf.keras.layers.GlobalAveragePooling1D(), # reduces dimensionality to 1d (embedding layer embeddeds each word in a title one by one)
            
            # tf.keras.layers.Flatten() 
            # squeeze_custom_layer()
        ])

        self.textvectorizer.adapt(self.unique_item_names)

    def call(self, inputs):

        item_id,  item_gics, item_name = inputs

        return tf.concat([
            self.embed_item_id(item_id),
            self.embed_items_gics(item_gics),
            self.embed_item_name(item_name)
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


def generate_embedding(item, item_model):
  
      item_id = item['STOCKCODE'] 
      item_name = item['STOCKNAME']
      item_gics = item['GICS']

      embedding = item_model([item_id, item_name, item_gics])
      return embedding

class Ranker(tfrs.Model):

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
