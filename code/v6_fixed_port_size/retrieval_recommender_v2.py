import tensorflow as tf
import numpy as np
import tensorflow_recommenders as tfrs

def generate_embedding(item, item_model):
  
      item_id = item['STOCKCODE'] 
      item_name = item['STOCKNAME']
      item_gics = item['GICS']

      embedding = item_model([item_id, item_name, item_gics])
      return embedding


class ItemModel(tf.keras.Model):
    def __init__(
        self,
        unique_item_ids,
        unique_item_names,
        unique_item_gics,
        # map_ = False

        ):
        super().__init__()

        self.max_tokens = 10000
        self.unique_item_ids = unique_item_ids
        self.unique_item_names = unique_item_names
        self.unique_item_gics = unique_item_gics
        # self.map_ = map_

        

        self.embed_item_id = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = self.unique_item_ids,
                mask_token =None
            ),
            tf.keras.layers.Embedding(
                input_dim = len(self.unique_item_ids)+1,
                output_dim = 8 #32
            )
        ])

        self.embed_item_gics = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = unique_item_gics,
                mask_token = None
            ),
            tf.keras.layers.Embedding(
                input_dim = len(unique_item_gics)+1,
                output_dim = 8 #len(unique_item_gics)
            )
        ])


        self.textvectorizer = tf.keras.layers.TextVectorization(
            max_tokens = self.max_tokens
        )

        self.embed_item_name = tf.keras.Sequential([
            self.textvectorizer,

            tf.keras.layers.Embedding(
                input_dim = self.max_tokens,
                output_dim = 16,
                mask_zero = True
            ),

            tf.keras.layers.GlobalAveragePooling1D() # reduces dimensionality to 1d (embedding layer embeddeds each word in a title one by one)
        ])

        self.textvectorizer.adapt(self.unique_item_names)
    
    def call(self, inputs):

        item_id, item_name, item_gics = inputs['STOCKCODE'], inputs['STOCKNAME'], inputs['GICS']

        return tf.concat([
            self.embed_item_id(item_id),
            self.embed_item_name(item_name),
            self.embed_item_gics(item_gics)
        ],
        axis = 1)



class UserModel(tf.keras.Model):
    def __init__(
        self,
        use_timestamp,
        unique_user_ids, 
        timestamps,
        timestamp_buckets):

        super().__init__()

        self.use_timestamp = use_timestamp
        self.unique_user_ids = unique_user_ids
        self.timestamp_buckets = timestamp_buckets
        self.timestamps = timestamps
        
        self.embed_user_id = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = self.unique_user_ids,
                mask_token = None
            ),
            tf.keras.layers.Embedding(
                input_dim = len(self.unique_user_ids)+1,
                output_dim = 16
            )
        ])

        if self.use_timestamp:
            self.embed_timestamp = tf.keras.Sequential([
                tf.keras.layers.Discretization(
                    bin_boundaries = list(self.timestamp_buckets)
                ),

                tf.keras.layers.Embedding(
                    input_dim = len(list(self.timestamp_buckets))+1 ,
                    output_dim = 15
                )
            ])

            self.normalize_timestamp = tf.keras.layers.Normalization(
                axis = None #calcuate a scaler mean and variance 
            )
            self.normalize_timestamp.adapt(self.timestamps)

    
    def call(self, inputs):

        user_id, timestamp = inputs["CDSACCNO"], inputs["UNIX_TS"]

        if self.use_timestamp:
            user_id_embed = self.embed_user_id(user_id)
            timestamp_embed = self.embed_timestamp(timestamp)
            norm_timestamp = tf.reshape(self.normalize_timestamp(timestamp), (-1,1)) #(-1,1) means first dimension to be infered

            return tf.concat([user_id_embed, timestamp_embed, norm_timestamp], axis = 1) #concatenate vertically
            
        return self.embed_user_id(user_id)



class Retriever(tfrs.models.Model):

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
      candidates= self.portfolios.batch(128).map(lambda x:self.item_model(x)),
      ks = [10]
    )

    self.task = tfrs.tasks.Retrieval(
      metrics = self.retrieval_metrics
    )

  def compute_loss(self, features, training=False) -> tf.Tensor:
    user_embeddings = self.user_model(features)

    item_embeddings = self.item_model(features)
    
    return self.task(
      query_embeddings = user_embeddings,
      candidate_embeddings = item_embeddings
      )