import numpy as np
import pandas as pd
import tensorflow as tf

def load_point(config, train=False):
    '''
    Point regression or classification
    '''
    
    if train:
        df = pd.read_csv(config['train'])
    else:
        df = pd.read_csv(config['val'])
    
    numeric_features = df[''] #specify the numeric columns
    cateorical_features = df[''] #specify the categorical columns
    all_features = numerical_features + categorical_features
    df = df[all_features] 
    
    df[numeric_features] = normalize(df[numeric_features])
    df = one_hot_encode(df, catgorical_features)
    
    y = df.pop('target')
    x = tf.convert_to_tensor(df)
    
    return x, y
    
def one_hot_code (df, cols):
    for col in cols:
        for i in range(np.max[col] + 1):
            df[f'{col} {i}'] = 0
        df = df.apply(lambda x:encode(i, col), axis=1)
        df = df.drop([f'{col}_{np.max(df[col])}',col], axis=1)
       
    return df

def encode(df, column):
    df[f'{column}_{df[column]}'] = 1
    return df

def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

def tf_point(x, y, config, train=True):
    
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.shuffle(buffer_size=1000) #shuffle the data for each epoch
    dataset = dataset.repeat(config['epochs']).batch(config['batch'], drop_remainder=True) #if data size is too big you cannot push the whole dataset
    dataset = dataset.prefetch(buffer_size=tf.data_experimental.AUTOTUNE)
    options = tf.data.options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    return dataset.with_options(options)

def get_dataset(train_paths, val_paths, batch, epochs):
    
    train_samples, train_targets = load_point(train_paths)
    val_samples, val_targets = load_point(val_paths)
    
    print(f'Samples: {len(train_samples)} - targets: {len(train_targets)}')
    print(f'Samples: {len(val_samples)} - targets: {len(val_targets)}')
    
    train_dataset = tf_point(train_samples, train_targets, batch, epochs)
    val_dataset = tf_point(val_samples, val_targets, batch, epochs)
    
    num_train = len(train_samples)
    num_val  = len(val_samples)
    
    steps_per_epoch = num_train // batch
    val_steps_per_epoch = num_val // batch
    
    return train_dataset, val_dataset, steps_per_epoch, val_steps_per_epoc