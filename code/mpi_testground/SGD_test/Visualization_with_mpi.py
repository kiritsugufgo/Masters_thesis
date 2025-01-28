import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

from mpi4py import MPI

from time import sleep

# python3 SDG_visualization.py

### Exec code with this command ###
### mpirun -n 4 python3 MPI_process.py ###

def z_function(x, y):
    return np.sin(5*x) * np.cos(5*y) / 5

def calc_gradient(x, y):
    return np.cos(5*x) * np.cos(5*y), -np.sin(5*x) * np.sin(5*y)

class MPI_PROCESS:
    def __init__(self, func, inputs):
        self.func = func
        self.inputs = inputs

    def __call__(self):
        mpi_world = MPI.COMM_WORLD
        mpi_world.barrier()
        mpi_master = (mpi_world.rank == 0)

        if mpi_master:
            return self.master(self.inputs)
        else:
            self.workers(self.func)

        mpi_world.barrier()

    def master(self, inputs):
        mpi_world = MPI.COMM_WORLD
        mpi_size = mpi_world.size

        if mpi_size <= 1:
            print(f"No workers available, exiting...")
            raise SystemExit(1)

        n_processes = len(inputs)
        print(f"Number of processes: {n_processes}")

        sent = 0
        finished = 0
        results = []

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

        while finished < n_processes:
            status = MPI.Status()
            result = mpi_world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            requesting_rank = status.Get_source()

            results.append(result)
            finished += 1

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

        return results

    def workers(self, func):
        mpi_world = MPI.COMM_WORLD

        b_continue = mpi_world.recv(source=0)

        while b_continue:
            data = mpi_world.recv(source=0)
            result = self.worker(data, func)

            mpi_world.send(result, dest=0)
            b_continue = mpi_world.recv(source=0)

    def worker(self, data, func):
        return func(data)

def worker_calc_gradient(data):
    current_pos, learning_rate = data
    for _ in range(1000):
        noise_x, noise_y = np.random.normal(0, 0.1, 2)
        noisy_pos = (current_pos[0] + noise_x, current_pos[1] + noise_y)
        X_deriv, Y_deriv = calc_gradient(noisy_pos[0], noisy_pos[1])
        X_new, Y_new = current_pos[0] - learning_rate * X_deriv, current_pos[1] - learning_rate * Y_deriv
        current_pos = (X_new, Y_new, z_function(X_new, Y_new))
    return {'output': current_pos}

def mpi4py_main():
    mpi_world = MPI.COMM_WORLD
    ax = plt.subplot(projection='3d', computed_zorder=False)

    x = np.arange(-1, 1, 0.05)
    y = np.arange(-1, 1, 0.05)
    X, Y = np.meshgrid(x, y)
    Z = z_function(X, Y)

    initial_positions = [
        (0.0, 0.5, z_function(0.0, 0.5)),
        (0.3, 0.9, z_function(0.3, 0.9)),
        (-0.2, 0.5, z_function(-0.2, 0.5))
    ]
    learning_rate = 0.01
    inputs = [(pos, learning_rate) for pos in initial_positions]

    run_mpi = MPI_PROCESS(worker_calc_gradient, inputs)
    results = run_mpi()

    for result in results:
        current_pos = result['output']

        ax.plot_surface(X, Y, Z, cmap='viridis', zorder=0)
        ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='cyan', zorder=1)
        plt.pause(0.001)
        ax.clear()

if __name__ == '__main__':
    mpi4py_main()