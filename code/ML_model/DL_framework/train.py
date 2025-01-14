import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import json

from data import *
from models import *

### Exec code with this command ###
### python train.py config.json ###

def get_dataset(config):
    
    train_samples, train_targets = eval(config['load_func'])(config, train=True)
    val_samples, val_targets = eval(config['load_func'])(config, train=False)
    
    print(f'Samples: {len(train_samples)} - targets: {len(train_targets)}')
    print(f'Samples: {len(val_samples)} - targets: {len(val_targets)}')
    
    train_dataset = eval(config['dataset_func'])(train_samples, train_targets, config, train=True)
    val_dataset = eval(config['dataset_func'])(train_samples, train_targets, config, train=False)
    
    num_train = len(train_samples)
    num_val  = len(val_samples)
    
    steps_per_epoch = num_train // config['batch']
    val_steps_per_epoch = num_val // config['batch']
    
    return train_dataset, val_dataset, steps_per_epoch, val_steps_per_epoch

def get_complied_model(config):
    
    if config['optimizer'] == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
    if config['optimizer'] == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])    
    
    if config['restore']:
        print("restoring model")
        model = eval(config['model'])
        model.summary()
        return model
    else:
        print('compiling and returning model')
        model = eval(config['model'])
        model.summary()
        model.compile(
            optimizers=opt,
            loss=eeval(config['loss_func']),
            metrics=[eval(config['metric'])]
        )
        return model

def run_training(config):
    
    output_dir = os.getcwd()
    
    print(f'loading data')
    train_dataset, val_dataset, steps_per_epoch, val_steps_per_epoch = get_dataset(config)
    model = get_complied_model(config)
    
    logs_dir = f"{config['output']}/model/logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    model_output = f"{output_dir}/{model}.h5"
    callbakcs = [tf.keras.callbacks.ModelCheckpoint(model_output, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False)]
    
    print(f'Training model: {model}')
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=config['epochs'],
        validation_data=val_dataset,
        validation_steps=val_steps_per_epoch,
        callbacks=callbacks
    )
    
    df = dataFrame(history.history)
    print("saving model metrics")
    df.to_csv(f"{output_dir}/{model}_history.csv")

def get_parser():
    parser = argparse.ArgumentParser(description='Train Neural Network')
    parser.add_argument('config', help='config_file[.json]', metavar='config', type=str)
    
    return parser

def command_line_runner():
    
    parser = get_parser()
    args = vars(parser.parse_args())
    
    config_file = args['config']
    
    return config_file

if __name__ == '__main__':
    
    config_file = command_line_runner()
    with open(config_file) as f:
        config = json.load(f)
        
    run_training(config)