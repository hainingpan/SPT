#!/bin/bash
# mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 100 --type=onsite --timing=True
mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 500 --type=link --timing=True
# mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 100 --type=onsite --timing=True --Bp=True
mpirun -n 4 python MI_LN_SSH.py --es 100 --pt 500 --type=link --timing=True --Bp=True