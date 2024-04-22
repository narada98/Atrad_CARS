import tensorflow as tf

'''
this handles embedding user Identifiers and contextual data.
time stamp is used as the contexual information here.
using timestamp is 
'''

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

        (user_id) = inputs  #, timestamp
        return self.embed_user_id(user_id)