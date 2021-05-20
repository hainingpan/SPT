#!/bin/bash
# mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 100 --type=onsite --timing=True
# mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 500 --type=link --timing=True
# mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 100 --type=onsite --timing=True --Bp=True
# mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 500 --type=link --timing=True --Bp=True

mpirun -np 4 python -m mpi4py.futures MI_LN_inf_SSH.py --es 100 --min 8 --max 100 --timing True --type onsite
mpirun -np 4 python -m mpi4py.futures MI_LN_inf_SSH.py --es 1000 --min 8 --max 100 --timing True --type link