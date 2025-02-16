from mpi4py import MPI

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
size = comm.Get_size()

data = comm.gather(worker**3, root=0)

print(f"{worker} : {worker**3}")

if worker == 0:
    print(f"final result: {data}")