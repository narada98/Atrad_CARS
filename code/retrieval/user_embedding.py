import tensorflow as tf

'''
this handles embedding user Identifiers and contextual data.
time stamp is used as the contexual information here.
using timestamp is 
'''

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

        user_id, timestamp = inputs

        if self.use_timestamp:
            user_id_embed = self.embed_user_id(user_id)
            timestamp_embed = self.embed_timestamp(timestamp)
            norm_timestamp = tf.reshape(self.normalize_timestamp(timestamp), (-1,1)) #(-1,1) means first dimension to be infered

            return tf.concat([user_id_embed, timestamp_embed, norm_timestamp], axis = 1) #concatenate vertically
            
        return self.embed_user_id(user_id)

# class UserModel(tf.keras.Model):
#     def __init__(
#         self,
#         unique_user_ids, 
        
#         ):

#         super().__init__()

#         self.unique_user_ids = unique_user_ids

#         self.embed_user_id = tf.keras.Sequential([
#             tf.keras.layers.StringLookup(
#                 vocabulary = self.unique_user_ids,
#                 mask_token = None
#             ),
#             tf.keras.layers.Embedding(
#                 input_dim = len(self.unique_user_ids)+1,
#                 output_dim = 32
#             )
#         ])
        
#     def call(self, inputs):

#         (user_id) = inputs

#         return self.embed_user_id(user_id)