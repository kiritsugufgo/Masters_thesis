import os
import re
from time import sleep
from tqdm import tqdm

from mpi4py import MPI

import torch
import numpy as np
import pandas as pd
from functools import wraps
from collections import Counter
import matplotlib.pyplot as plt

'''
### Exec code with this command ###
### mpirun -n 4 python3 SGD_MPI.py ###
'''

mpi_world = MPI.COMM_WORLD
mpi_world.barrier() # synchronization

def user_defined(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(**(dict(zip(list(inspect.signature(func).parameters.keys())[:len(args)], args)) 
                       | {key:kwargs[key] for key in list(inspect.signature(func).parameters.keys()) if key in kwargs}))
    return wrapper

#Chunking the data and sending it to the workers
def chunk_data(X, y, chunk_size):
    num_samples = X.shape[0]
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        yield X[start_idx:end_idx], y[start_idx:end_idx]
        
def chunkfunc(X, y, chunk_size=1_000, **kwargs):
    total_chunks = (X.shape[0] + chunk_size - 1) //chunk_size
    chunks = chunk_data(X, y, chunk_size)
    return total_chunks, chunks

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#SGD optimization on each chunk
def workfunc(chunk, weights, bias, learning_rate):
    X_chunk, y_chunk = chunk
    num_samples = X_chunk.shape[0]
    for i in range(num_samples):
        xi = X_chunk[i]
        yi = y_chunk[i]
        linear_output = np.ot(xi, weights) + bias
        prediction = sigmoid(linear_output)
        
        #Compute gradients
        error = prediction - yi
        weights -= learning_rate * error * xi
        bias -= learning_rate * error
   
#collect and aggregate the results     
def resultfunc(result, weights, bias):
    weights_chunk, bias_chunk = result
    weights += weights_chunk
    bias += bias_chunk
    return weights, bias

#initialize data and parameters
def initfunc(X, y):
    num_features = X.shape[1]
    weights = np.zeros(num_features)
    bias = 0
    return {'weights': weights, 'bias': bias}

#Finalize the results
def finalfunc(weights, bias, num_chunks):
    weights /= num_chunks
    bias /= num_chunks
    return weights, bias

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

#Build vocabulary based 
def build_vocab(texts, vocab_size):
    counter = Counter()
    for text in texts:
        words = preprocess(text)
        counter.update(words)
        
    most_common = counter.most_common(vocab_size)
    vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
    return {word: i for i, word in enumerate(vocab)}

#convert text to BoW 
def text_to_Bow(text, vocab):
    bow = np.zeros(len(vocab))
    word_counts = Counter(preprocess(text))
    for word, count in word_counts.items():
        if word in vocab:
            bow[vocab[word]] = count
    return bow

class MPI_PROCESS:
    def __init__(self, chunkfunc, workfunc, resultfunc, initfunc=None, finalfunc=None, **kwargs):
        self.cfunc = chunkfunc
        self.wfunc = workfunc
        self.rfunc = resultfunc
        self.ifunc = initfunc
        self.ffunc = finalfunc
        self.kwargs = kwargs

    def __call__(self):
        self.master(mpi_world, **self.kwargs) if (mpi_world.rank == 0) else self.worker(mpi_world, **self.kwargs)
        mpi_world.barrier()
        results = self.final(**self.kwargs) if (mpi_world.rank == 0) else None
        mpi_world.barrier()
        return results
    

    def master(self, mpi_world, **kwargs):
        print('parrallel processing begins:')
        print(f'Number of Workers: {mpi_world.size}')

        mpi_status = MPI.Status()
            
        # Filter kwargs for initfunc
        init_kwargs = {k: v for k, v in kwargs.items() if k in ['X', 'y']}
        kwargs.update(self.ifunc(**init_kwargs)) if self.ifunc is not None else print('no initialization required')

        total_chunks, chunks = self.cfunc(**kwargs)

        # give out starting chunks to each worker
        for worker in range(1, mpi_world.size):
            mpi_world.send(next(chunks, None), dest=worker)

        # listen for results and handle them then send more work
        for _ in tqdm(range(total_chunks)):
            result, chunk = mpi_world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=mpi_status)
            result_kwargs = {k: v for k, v in kwargs.items() if k in ['weights', 'bias']}
            _ = self.rfunc(result, **({'chunk': chunk} | result_kwargs))
            # send a new chunk if there are any remaining
            mpi_world.send(next(chunks, None), dest=mpi_status.Get_source())
            
    def worker(self, mpi_world, **kwargs):
        chunk = mpi_world.recv(source=0)
        
        while chunk is not None:
            work_kwargs = {k: v for k, v in kwargs.items() if k in ['weights', 'bias', 'learning_rate']}
            work_kwargs['chunk'] = chunk  # Ensure chunk is included
            result = self.wfunc(**work_kwargs)
            mpi_world.send((result, chunk), dest=0)
            chunk = mpi_world.recv(source=0)
        
def mpi4py_main():
    mpi_world = MPI.COMM_WORLD
    mpi_master = (mpi_world.rank == 0)
    
    # Example text data and labels
    text_data = ["I love this movie", "I hate this movie", "This movie is great", "This movie is terrible"]
    labels = np.array([1, 0, 1, 0])
    
    # Preprocess data
    vocab = build_vocab(text_data, vocab_size=10000)
    X = np.array([text_to_Bow(text, vocab) for text in text_data])
    y = labels
    
    # Initialize parameters
    learning_rate = 0.01
    chunk_size = 2
    
    # Run parallel processing
    run_mpi = MPI_PROCESS(chunkfunc, workfunc, resultfunc, initfunc, finalfunc, X=X, y=y, learning_rate=learning_rate, chunk_size=chunk_size)
    weights, bias = run_mpi()
    
    if mpi_master:
        print(f"Final Weights: {weights}")
        print(f"Final Bias: {bias}")

if __name__ == '__main__':
    mpi4py_main()
    