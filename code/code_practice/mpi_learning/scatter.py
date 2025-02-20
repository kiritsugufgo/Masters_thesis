from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
worker = comm.Get_rank()

if worker == 0:
    data = [{"data for: "+str(i): i } for i in range(0, size)]
    print(data)
else:
    data = None
    
data = comm.scatter(data, root=0)
print(f"{worker} : {data}")

