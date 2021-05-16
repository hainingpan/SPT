#!/bin/bash
mpirun -n 4 python MI_LN_SSH.py --es 1000 --pt 100 --type=onsite --timing=True
mpirun -n 4 python MI_LN_SSH.py --es 1000 --pt 100 --type=link --timing=True