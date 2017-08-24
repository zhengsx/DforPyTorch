from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
 

data = [comm_rank] * 4
all_sum = comm.allreduce(data, op=MPI.SUM)
print(all_sum)

x=[data] * 4
all_all_sum = comm.allreduce(x, op=MPI.SUM)
print(all_all_sum)