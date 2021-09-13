import os

template='''
#!/bin/bash
#SBATCH --mail-user=hnpan@terpmail.umd.edu
#SBATCH --mail-type=ALL
#SBATCH --share
#SBATCH -t 2:30:00
#SBATCH --ntasks 101
##SBATCH --mem=200G

. ~/.profile

cd $PWD

module load python
module load openmpi
export OMP_NUM_THREADS=1
export OMP_DYNAMIC="FALSE"

mpirun -np $SLURM_NTASKS python -m mpi4py.futures Born_CI_geo.py --es esx --num 11 --Lx lx --Ly ly --Born Bx --geo Gx
'''

input1=[(1,Lx,Ly,-1,1) for Lx,Ly in zip([16,24,32],[16,24,32])]
input2=[(1,Lx,Ly,0,1) for Lx,Ly in zip([16,24,32],[16,24,32])]
input3=[(20,Lx,Ly,1,1) for Lx,Ly in zip([16,24,32],[16,24,32])]
input4=[(1,Lx,Ly,-1,2) for Lx,Ly in zip([32,32,32],[16,24,32])]
input5=[(1,Lx,Ly,0,2) for Lx,Ly in zip([32,32,32],[16,24,32])]
input6=[(20,Lx,Ly,1,2) for Lx,Ly in zip([32,32,32],[16,24,32])]

inputs=(input1+input2+input3+input4+input5+input6)

for i in inputs:
    fn='CI_g{:d}_es{:}_Lx{:}_Ly{:}_B{:}.sh'.format(*i)
    with open(fn,'w') as f:
        # print()
        f.write(template.replace('esx',str(i[0])).replace('lx',str(i[1])).replace('ly',str(i[2])).replace('Bx',str(i[3])).replace('Gx',str(i[4])))
        os.chmod(fn,0o777)