import tensorflow as tf

def global_average_mean(x):
  """Custom layer to perform global average mean pooling."""
  axis = -2  # Reduce mean along the last dimension
  return tf.reduce_mean(x, axis=axis)

def reshaper(x):
    shape = (-1,5,1)
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