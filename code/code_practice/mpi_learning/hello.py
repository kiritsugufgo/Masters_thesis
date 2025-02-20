from mpi4py import MPI

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
size = comm.Get_size()

print(f"hello world from worker {worker} of {size}")