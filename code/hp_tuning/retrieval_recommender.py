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
        id_embedding_dim,
        gics_embedding_dim,
        # name_embedding_dim
        # map_ = False

        ):
        super().__init__()

        self.max_tokens = 10000
        self.unique_item_ids = unique_item_ids
        self.unique_item_names = unique_item_names
        self.unique_item_gics = unique_item_gics
        self.id_embedding_dim = id_embedding_dim
        self.gics_embedding_dim = gics_embedding_dim
        # self.name_embedding_dim = name_embedding_dim
        # self.map_ = map_

        

        self.embed_item_id = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = self.unique_item_ids,
                mask_token =None
            ),
            tf.keras.layers.Embedding(
                input_dim = len(self.unique_item_ids)+1,
                # output_dim = 8 #32
                output_dim = self.id_embedding_dim
            )
        ])

        self.embed_item_gics = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = unique_item_gics,
                mask_token = None
            ),
            tf.keras.layers.Embedding(
                input_dim = len(unique_item_gics)+1,
                # output_dim = 8 #len(unique_item_gics)
                output_dim = self.gics_embedding_dim
            )
        ])


        self.textvectorizer = tf.keras.layers.TextVectorization(
            max_tokens = self.max_tokens
        )

        self.embed_item_name = tf.keras.Sequential([
            self.textvectorizer,

            tf.keras.layers.Embedding(
                input_dim = self.max_tokens,
                # output_dim = 16,
                # output_dim = self.name_embedding_dim,
                output_dim = (32 - (self.id_embedding_dim + self.gics_embedding_dim)),
                mask_zero = True
            ),

            tf.keras.layers.GlobalAveragePooling1D() # reduces dimensionality to 1d (embedding layer embeddeds each word in a title one by one)
        ])

        self.textvectorizer.adapt(self.unique_item_names)
    
    def call(self, inputs, map_ = False):

        if map_ == False:
            item_id, item_name, item_gics = inputs
        else:
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
        timestamp_buckets,
        id_embedding_dim,
        # ts_embedding_dim
        ):

        super().__init__()

        self.use_timestamp = use_timestamp
        self.unique_user_ids = unique_user_ids
        self.timestamp_buckets = timestamp_buckets
        self.timestamps = timestamps
        self.id_embedding_dim = id_embedding_dim
        # self.ts_embedding_dim = ts_embedding_dim

        
        self.embed_user_id = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary = self.unique_user_ids,
                mask_token = None
            ),
            tf.keras.layers.Embedding(
                input_dim = len(self.unique_user_ids)+1,
                # output_dim = 16
                output_dim = self.id_embedding_dim
            )
        ])

        if self.use_timestamp:
            self.embed_timestamp = tf.keras.Sequential([
                tf.keras.layers.Discretization(
                    bin_boundaries = list(self.timestamp_buckets)
                ),

                tf.keras.layers.Embedding(
                    input_dim = len(list(self.timestamp_buckets))+1 ,
                    # output_dim = 15
                    output_dim = (31 - self.id_embedding_dim)
                )
            ])

            self.normalize_timestamp = tf.keras.layers.Normalization(
                axis = None #calcuate a scaler mean and variance 
            )
            self.normalize_timestamp.adapt(self.timestamps)

    
    def call(self, inputs):

        user_id, timestamp = inputs

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
    portfolios,
    item_id_embedding_dim,
    gics_embedding_dim,
    # name_embedding_dim,
    user_id_embedding_dim
    # ts_embedding_dim
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

    self.item_id_embedding_dim= item_id_embedding_dim
    self.gics_embedding_dim= gics_embedding_dim
    # self.name_embedding_dim= name_embedding_dim

    self.user_id_embedding_dim = user_id_embedding_dim
    # self.ts_embedding_dim = ts_embedding_dim

    self.item_model = ItemModel(
      unique_item_ids = self.unique_item_ids,
      unique_item_names = self.unique_item_names,
      unique_item_gics = self.unique_item_gics,
      id_embedding_dim= self.item_id_embedding_dim,
      gics_embedding_dim= self.gics_embedding_dim,
    #   name_embedding_dim= self.id_embedding_dim
    )

    self.user_model = UserModel(
      use_timestamp = self.use_timestamp,
      unique_user_ids = self.unique_user_ids, 
      timestamps = self.timestamps, 
      timestamp_buckets = self.timestamp_buckets,
      id_embedding_dim = self.user_id_embedding_dim,
    #   ts_embedding_dim = self.ts_embedding_dim
    )

    self.retrieval_metrics = tfrs.metrics.FactorizedTopK(
        candidates= self.portfolios.batch(128).map(lambda x:generate_embedding(x, self.item_model)),
        ks = [10]
    )

    self.task = tfrs.tasks.(
      metrics = self.retrieval_metrics
    )

  def compute_loss(self, features, training=False) -> tf.Tensor:
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
