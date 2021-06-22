from pSC import *
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
import pickle
import argparse

def run(p):
    m,Lx,Ly=p
    params=Params(m=m,Lx=Lx,Ly=Ly,bcx=-1,bcy=1)
    # total=params.linearize_index([np.arange(params.Lx),np.arange(params.Ly)],2)
    subA=params.linearize_index([np.arange(params.Lx//4),np.arange(params.Ly)],2)
    proj_range=params.linearize_index([np.arange(params.Lx//4)+params.Lx//4,np.arange(params.Ly)],2,proj=True)
    subB=params.linearize_index([np.arange(params.Lx//4)+params.Lx//2,np.arange(params.Ly)],2)

    params.measure_all_Born(proj_range,linear=True)
    LN=params.log_neg(subA,subB,linear=True)
    MI=params.mutual_information_m(subA,subB,linear=True)
    return MI,LN

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--Lx',default=32,type=int)
    parser.add_argument('--Lymax',default=32,type=int)
    parser.add_argument('--es',default=20,type=int)
    parser.add_argument('--m',default=2,type=int)

    args=parser.parse_args()

    Ly_list=np.arange(2,args.Lymax)
    es=args.es
    LN_Born_scaling_list=[]
    MI_Born_scaling_list=[]
    Lx=args.Lx
    executor=MPIPoolExecutor()
    ensemble_list_pool=[]
    for Ly_i,Ly in enumerate(Ly_list):
        inputs=[(args.m,Lx,Ly) for _ in range(es)]
        ensemble_list_pool.append(executor.map(run,inputs))
    for Ly_i,Ly in enumerate(Ly_list):
        print("Ly_i={:d}:".format(Ly_i))
        st=time.time()
        MI_ensemble_list=[]
        LN_ensemble_list=[]
        for result in ensemble_list_pool[Ly_i]:
            MI,LN=result
            MI_ensemble_list.append(MI)
            LN_ensemble_list.append(LN)
        print('gather all Ly_i={:d}:{:.1f}'.format(Ly_i,time.time()-st))
        MI_Born_scaling_list.append(MI_ensemble_list)
        LN_Born_scaling_list.append(LN_ensemble_list)
    executor.shutdown()
    MI_Born_scaling_list=np.array(MI_Born_scaling_list)
    LN_Born_scaling_list=np.array(LN_Born_scaling_list)
    
    with open('scaling_m{:d}_En{:d}_Lx{:d}.pickle'.format(args.m,args.es,args.Lx),'wb') as f:
        pickle.dump([Ly_list,MI_Born_scaling_list,LN_Born_scaling_list],f)