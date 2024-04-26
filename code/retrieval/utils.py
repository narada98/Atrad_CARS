import tensorflow as tf
import pandas as pd

def dataset_to_dataframe(dataset):
  
  data = []
  for element in dataset.as_numpy_iterator():
    data.append(element)

  return pd.DataFrame(data)