from tqdm import tqdm

### Exec code with this command ###
### mpirun -n 4 python3 MPI_process.py ###

class MPI_PROCESS:
    def __init__(self, func, inputs, output):
        
        self.func= func
        self.inputs = inputs
        self.output = output
        
    def __call__(self):
        from mpi4py import MPI
        
        mpi_world = MPI.COMM_WORLD
        mpi_world.barrier()
        mpi_master = (mpi_world.rank == 0)
        
        if mpi_master:
            self.master(self.inputs)
        else:
            self.workers(self.func)
            
        mpi_world.barrier()
        
    def master(self, inputs):
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size
        
        if(mpi_size <= 1):
            print(f"No workers available, exiting...")
            raise SystemExit(1)
    
        n_processes = len(inputs)
        print(f"Number of processes:{n_processes}")
        
        sent = 0
        finished = 0
        # send continue to worker if n_process > n_workers
        # if not send don't continue
        for worker in range(1, mpi_size):
            if worker > n_processes:
                cont = False 
                mpi_world.send(cont, dest=worker)
            else:
                cont = True
                mpi_world.send(cont, dest=worker)
                data = inputs[sent]
                mpi_world.send(data, dest=worker)
                sent += 1
        
        pbar = tqdm(total=n_processes) #progressbar
        while finished < n_processes:
            status = MPI.Status()
            results = mpi_world.recv(source = MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            requesting_rank = status.Get_source()
            
            try:
                with open(self.output, "a+") as file:
                    file.write(f"{results['output']} \n")
            except KeyError:
                print(f"results not found")
            del results

            finished += 1
            pbar.update()
            
            if sent == n_processes:
                data = None
            else: 
                data = inputs[sent]
            if data is None:
                b_continue = False
                mpi_world.send(b_continue, dest=requesting_rank)
            else:
                b_continue = True
                mpi_world.send(b_continue, dest=requesting_rank)
                mpi_world.send(data, dest=requesting_rank)
                sent += 1
        return True
    
    def workers(self, func):
        from mpi4py import MPI
        mpi_world = MPI.COMM_WORLD
        
        b_continue = mpi_world.recv(source=0)
        
        while b_continue:
            data = mpi_world.recv(source=0)
            results = self.worker(data, func)
            
            mpi_world.send(results, dest=0)
            b_continue = mpi_world.recv(source=0)
            del results
        return True
                      
    def worker(self, data, func):
        return func(data)          
    

from time import sleep
def some_func(data):
    sleep(1)
    return {'output': data**2}

def mpi4py_main():
    from mpi4py import MPI
    mpi_world = MPI.COMM_WORLD
    
    inputs = [0,1,2,3,4,5,6,7,8,9]
    output = "output.txt"
    run_mpi = MPI_PROCESS(some_func, inputs, output)
    
    run_mpi()
### MPI4py ###

if __name__ == '__main__':
    
    mpi4py_main()