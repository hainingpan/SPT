#!/bin/bash
# mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 100 --type=onsite --timing=True
# mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 500 --type=link --timing=True
# mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 100 --type=onsite --timing=True --Bp=True
# mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 500 --type=link --timing=True --Bp=True
# mpirun -np 4 python -m mpi4py.futures MI_LN_inf_SSH.py --es 100 --min 8 --max 100 --timing True --type onsite 
# mpirun -np 4 python -m mpi4py.futures MI_LN_inf_SSH.py --es 1000 --min 8 --max 100 --timing True --type link
# mpirun -np 4 python -m mpi4py.futures MI_LN_inf_SSH.py --es 100 --min 8 --max 100 --timing True --type onsite --prob True --size 32
# mpirun -np 4 python -m mpi4py.futures MI_LN_inf_SSH.py --es 1000 --min 8 --max 100 --timing True --type link 
# mpirun -np 4 python -m mpi4py.futures Born_CI.py --es 100 --timing True --Lx 16 --Ly 16 --num 20
# mpirun -np 4 python -m mpi4py.futures MI_LN_CI.py --es 10 --timing True --Lx 16 --Ly 16 --pts 20

# mpirun -n 64 python -m mpi4py.futures SSH_scaling.py --ps 1000 --L 128 --es 1
# mpirun -n 64 python -m mpi4py.futures SSH_scaling.py --ps 1000 --L 128 --es 100 --type onsite
# mpirun -n 4 python -m mpi4py.futures SSH_scaling.py --ps 1000 --L 64 --es 50 --type link

# mpirun -n 64 python -m mpi4py.futures Majorana_scaling.py --ps 1000 --L 128 --es 1
# mpirun -n 64 python -m mpi4py.futures Majorana_scaling.py --ps 1000 --L 128 --es 100 --type onsite

mpirun -n 4 python -m mpi4py.futures Born_CI_geo.py --es 1 --num 11 --Lx 16 --Ly 16 --Born -1 --geo 1
mpirun -n 4 python -m mpi4py.futures Born_CI_geo.py --es 1 --num 11 --Lx 16 --Ly 16 --Born 0 --geo 1
mpirun -n 4 python -m mpi4py.futures Born_CI_geo.py --es 20 --num 11 --Lx 16 --Ly 16 --Born 1 --geo 1

(16,16),(24,24),(32,32)

mpirun -n 4 python -m mpi4py.futures Born_CI_geo.py --es 1 --num 11 --Lx 32 --Ly 16 --Born -1 --geo 2
mpirun -n 4 python -m mpi4py.futures Born_CI_geo.py --es 1 --num 11 --Lx 32 --Ly 16 --Born 0 --geo 2
mpirun -n 4 python -m mpi4py.futures Born_CI_geo.py --es 20 --num 11 --Lx 32 --Ly 16 --Born 1 --geo 2

(32,16),(32,24),(32,32)