import os
from time import sleep
from tqdm import tqdm
from mpi4py import MPI

import torch
import numpy as np
import matplotlib.pyplot as plt

### Exec code with this command ###
### mpirun -n 4 python3 MPI_process.py ###


def regression(my_x, my_m, my_b):
    return my_m*my_x + my_b

def mse(my_yhat, my_y):
    sigma = torch.sum((my_yhat - my_y)**2)
    return sigma / len(my_y)

def SGD_calc(data):
    x, y, m, b, lr, batch_size = data
    batch_indices = np.random.choice(len(x), size=batch_size, replace=False)
    yhat = regression(x[batch_indices], m, b)
    C = mse(yhat, y[batch_indices])
    C.backward()
    return {'m_grad': m.grad.item(), 'b_grad': b.grad.item(), 'cost': C.item()}

def plot_figure():
    torch.manual_seed(42)
    np.random.seed(42)
    n = 8000000
    x = torch.linspace(0., 8., n)
    y = -0.5 * x + 2 + torch.normal(mean=torch.zeros(n), std=1)
    indices = np.random.choice(n, size=2000, replace=False)
    
    fig, ax = plt.subplots()
    plt.title("Clinical Trial")
    plt.xlabel("Drug dosage (mL)")
    plt.ylabel("Forgetfulness")
    _ = ax.scatter(x[indices], y[indices], alpha=0.1)
    
    m = torch.tensor([0.9]).requires_grad_()
    b = torch.tensor([0.1]).requires_grad_()
    
    x_min, x_max = ax.get_xlim()
    y_min = regression(x_min, m, b).detach().numpy()
    y_max = regression(x_max, m, b).detach().numpy()
    
    plt.ylabel('b = {}'.format('%.3g' % b.item()))
    plt.xlabel('m = {}'.format('%.3g' % m.item()))
    
    ax.set_xlim([x_min, x_max])
    _ = ax.plot([x_min, x_max], [y_min, y_max], c='C01')
    
    return fig, ax, indices


class MPI_PROCESS:
    def __init__(self, func, inputs, output):
        self.func = func
        self.inputs = inputs
        self.output = output

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
        m_grad_sum = 0
        b_grad_sum = 0
        total_cost = 0

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

            m_grad_sum += result['m_grad']
            # print(f'''m_grad: {result['m_grad']}''')
            # print(f'''m_grad_sum: {m_grad_sum}''')
            b_grad_sum += result['b_grad']
            # print(f'''b_grad: {result['b_grad']}''')
            # print(f'''b_grad_sum: {b_grad_sum}''')
            total_cost += result['cost']
            finished += 1

            # Write worker results to output file
            with open(self.output, "a") as f:
                f.write(f"Worker {requesting_rank}: m_grad = {result['m_grad']}, b_grad = {result['b_grad']}, cost = {result['cost']}\n")
                
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

        m_grad_avg = m_grad_sum / n_processes
        b_grad_avg = b_grad_sum / n_processes
        avg_cost = total_cost / n_processes

        return m_grad_avg, b_grad_avg, avg_cost

    def workers(self, func):
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

def mpi4py_main():
    mpi_world = MPI.COMM_WORLD
    mpi_master = (mpi_world.rank == 0)
    
    torch.manual_seed(42)
    np.random.seed(42)
    n = 8000000
    x = torch.linspace(0., 8., n)
    y = -0.5 * x + 2 + torch.normal(mean=torch.zeros(n), std=1)
    m = torch.tensor([0.9]).requires_grad_()
    b = torch.tensor([0.1]).requires_grad_()
    lr = 0.001
    batch_size = 32
    rounds = 32
    inputs = [(x, y, m, b, lr, batch_size) for _ in range(rounds)]
    output = "output_1.txt"
    
    if mpi_master:
        if os.path.exists(output):
            os.remove(output)
    
    run_mpi = MPI_PROCESS(SGD_calc, inputs, output)
    m_grad_avg, b_grad_avg, avg_cost = run_mpi()

    if mpi_master:
        print(f"Average Cost: {avg_cost}")
        print(f"Average m Gradient: {m_grad_avg}")
        print(f"Average b Gradient: {b_grad_avg}")
        
        # Get the figure and axis from plot_figure
        fig, ax, indices = plot_figure()
        
        # Initialize m and b
        m_opt = m.clone().detach()
        b_opt = b.clone().detach()
        
        # Accumulate gradients over all rounds
        for _ in range(rounds):
            m_opt -= lr * m_grad_avg
            b_opt -= lr * b_grad_avg
        
        # Plot the optimized line
        fig, ax = plt.subplots()
        ax.scatter(x[indices], y[indices], alpha=0.1)
        
        x_min, x_max = ax.get_xlim()
        y_min_opt = regression(x_min, m_opt, b_opt).detach().numpy()
        y_max_opt = regression(x_max, m_opt, b_opt).detach().numpy()
        
        ax.plot([x_min, x_max], [y_min_opt, y_max_opt], c='purple', label='Optimized Line')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    mpi4py_main()
    
    # mpi_world = MPI.COMM_WORLD
    # mpi_master = (mpi_world.rank == 0)
    
    # if mpi_master:
    #     plot_figure()