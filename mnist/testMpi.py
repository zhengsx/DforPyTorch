from mpi4py import MPI
import numpy as np

def mpi_list_sum(x,y):
    return [a+b for a,b in zip(x,y)]

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
 

data = np.eye(3)
eye_arr=[np.eye(1),np.eye(2),np.eye(3)]

all_all_sum = [sum_np / 4 for sum_np in comm.allreduce(eye_arr, op=mpi_list_sum)]
print(all_all_sum)
