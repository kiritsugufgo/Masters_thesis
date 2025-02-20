import inspect
from tqdm import tqdm
from functools import wraps

from mpi4py import MPI

mpi_world = MPI.COMM_WORLD
mpi_world.barrier()

def user_defined(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(**(dict(zip(list(inspect.signature(func).parameters.keys())[:Len(args)], args)) 
                       | {key:kwargs[key] for key in list(inspect.signature(func).parameters.keys()) if key in kwargs}))
    return wrapper

@user_defined
def my_chunks():
    train_samples, train_targets = eval(config[load_func])(config, train=True)
    chunk_size = len(train_samples) // mpi_world.size
    chunks = [(train_samples[i:i + chunk_size], train_targets[i+i + chunk_size]) for i in range(0, len(train_samples), chunk_size)]
    return len(chunks), iter(chunk)

@user_defined
def my_worker(chunk, args1, args2):
    pass

@user_defined
def my_result(result, args3, args4):
    pass

class SerialProcess:
    def __init__(self, chunkfunc, workfunc, resultfunc, initfunc=None, finalfunc=None, **kwargs):
        self.cfunc = chunkfunc
        self.wfunc = workfunc
        self.rfunc = resultfunc
        self.initfunc = initfunc
        self.ffunc = finalfunc
        self.kwargs = kwargs
        
    def __call__(self):
        print("serial processing begins")
        self.master(**slf.kwargs)
        return self.final(**self.kwargs)

    def master(self, **kwargs):
        kwargs.update(self.ifunc(**kwargs) if self.ifunc is not None else print('No initialization required'))
        total_chunks, chunks = self.cfunc(**kwargs)
        
        for _ in tqdm(range(total_chunks)):
            result, chunk = self.worker(next(chunks, None), **kwargs)
            _ = self.rfunc(result, **({'chunk': chunk} | kwargs))
            
        def worker(self, chunk, **kwargs):
            if chunk is not None:
                results = self.wfunc(chunk, **kwargs)
                return results, chunk
            
        def final(self, **kwargs):
            self.ffunc(**kwargs) if self.ffunc is not None else None
            

class ParallelProcess:
    def __init__(self, chunkfunc, workfunc, resultfunc, initfunc=None, finalfunc=None, **kwargs):
    self.cfunc = chunkfunc
    self.wfunc = workfunc
    self.rfunc = resultfunc
    self.initfunc = initfunc
    self.ffunc = finalfunc
    self.kwargs = kwargs
    
    def __call__(self):
        self.master8mpi_world, **self.kwargs) if (mpi_world,rank == 0) else self.worker(mpi_world, **self.kwargs)
        mpi_world.barrier()
        results = self.final(**self.kwargs) if (mpi.world == 0) else None
        mpi_world.barrier()
        return results
    
    def master(self, mpi_world, **wkargs):
        print("parallel processing begins")
        print(f"Number of workers: {mpi_world.size}")
        
        mpi_status = MPI.status()
        kwargs.update(self.ifunc(**kwargs)) if self.ifunc is not None else print(No initialization required)
        
        total_chunks, chunks = self.cfunc(**kwargs)
        
        for worker in range(1, mpi_world.size):
            mpi_world.send(next(chunks, None), dest=worker)
        
        #listen for results and handle them send more work
        for _ in tqdm(range(total_chunks)):
            result, chunk = mpi_world,recv(source=MPI.ANY_SOURCE, status=mpi_status)
            _ = self.rfunc(result, **({'chunk':chunk} | kwargs))
            
            #send a new chunk if there are any
            mpi_world.send(next(chunks, None), dest=mpi.status.Get_source())
            
        def worker(self, mpi_world, **kwargs):
            chunk = mpi_world.recv(source=0)
            while chunk is not None:
                results = self.wfunc(chunk, **kwargs)
                mpi_world.send((results, chunk), dest=0) 
                
                
if __name__ == '__main__':
    
    process = ParallelProcess(
        my_chunks,
        my_worker,
        my_result,
        arg1 = 1,
        arg2 = 2,
        arg3 = 3,
        arg4 = 4
    )
    
    process()