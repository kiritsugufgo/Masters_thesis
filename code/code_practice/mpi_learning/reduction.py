from mpi4py import MPI

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
size = comm.Get_size()

def reduce_func(a,b):
    return a+b

data = comm.reduce(worker, op=MPI.SUM, root=0)

print(f"{worker} : {worker}")

if worker == 0:
    print(f"final rsult: {data}")