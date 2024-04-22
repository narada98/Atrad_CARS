import tensorflow as tf 

'''
this handles embedding item Identifiers and contextual data.
movie title itself is used as the contexual information here.
using timestamp is 
'''

class ItemModel(tf.keras.Model):
    def __init__(
        self,
        unique_item_ids,
        # unique_item_names,
        unique_item_gics

        ):
        super().__init__()

        self.max_tokens = 10000
        self.unique_item_ids = unique_item_ids
        # self.unique_item_names = unique_item_names
        self.unique_item_gics = unique_item_gics

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


        # self.textvectorizer = tf.keras.layers.TextVectorization(
        #     max_tokens = self.max_tokens
        # )

        # self.embed_item_name = tf.keras.Sequential([
        #     self.textvectorizer,

        #     tf.keras.layers.Embedding(
        #         input_dim = self.max_tokens,
        #         output_dim = 16,
        #         mask_zero = True
        #     ),

        #     tf.keras.layers.GlobalAveragePooling1D() # reduces dimensionality to 1d (embedding layer embeddeds each word in a title one by one)
        # ])

        # self.textvectorizer.adapt(self.unique_item_names)
    
    def call(self, inputs):

        item_id,  item_gics = inputs  #item_name,

        return tf.concat([
            self.embed_item_id(item_id),
            self.embed_items_gics(item_gics)
        ],
        axis = 1)
    