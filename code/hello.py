from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get:rank()
print(f"hello world from process {rank}")