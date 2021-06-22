from Chern_insulator import *
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
import pickle
import argparse

def run(p):
    m,Lx,Ly=p
    params=Params(m=m,Lx=Lx,Ly=Ly,bcx=-1,bcy=1)
    proj_range=params.linearize_index([np.arange(params.Lx//4,params.Lx//2),np.arange(0,params.Ly)],4,proj=True,k=2)
    prob_list=[1-(1-(-1)**i)/2 for i,x in enumerate(proj_range)]
    params.measure_all_Born(proj_range,type='onsite',prob=prob_list,linear=True)
    MI=params.mutual_information_m([np.arange(Lx//4),np.arange(Ly)],[np.arange(Lx//4)+Lx//2,np.arange(Ly)])
    LN=(params.log_neg([np.arange(params.Lx//4),np.arange(params.Ly)],[np.arange(params.Lx//4)+params.Lx//2,np.arange(params.Ly)]))
    return MI,LN

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--Lx',default=64,type=int)
    parser.add_argument('--Lymax',default=32,type=int)
    parser.add_argument('--m',default=1,type=int)

    args=parser.parse_args()

    Ly_list=np.arange(2,args.Lymax)
    LN_Born_scaling_list=[]
    MI_Born_scaling_list=[]
    Lx=args.Lx
    executor=MPIPoolExecutor()
    ensemble_list_pool=[]

    inputs=[(args.m,Lx,Ly) for Ly in Ly_list]
    async_result=(executor.map(run,inputs))

    for index,results in enumerate(async_result):
        MI_Born_scaling_list.append(results[0])
        LN_Born_scaling_list.append(results[1])

    with open('CI_scaling_m{:d}_Lx{:d}.pickle'.format(args.m,args.Lx),'wb') as f:
        pickle.dump([Ly_list,MI_Born_scaling_list,LN_Born_scaling_list],f)