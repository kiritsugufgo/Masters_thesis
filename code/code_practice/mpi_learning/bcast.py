from mpi4py import MPI

comm  = MPI.COMM_WORLD
size = comm.Get_size()
worker = comm.Get_rank()

if worker == 0:
    data = {f"data for everyone: [1,2,3,4,5]"}
else:
    data = None
    
data = comm.bcast(data, root=0)
print(f"{worker} : {data}")