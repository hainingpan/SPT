#!/bin/bash
mpirun -np 4 python -m mpi4py.futures MI_position.py --es 1000
mpirun -np 4 python -m mpi4py.futures MI_position.py --es 2000
mpirun -np 4 python -m mpi4py.futures MI_position.py --es 3000
mpirun -np 4 python -m mpi4py.futures MI_position.py --es 5000
mpirun -np 4 python -m mpi4py.futures MI_position.py --es 10000