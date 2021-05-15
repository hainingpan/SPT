from Majorana_chain import *
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor


def mutual_info_run_MPI(L=512,es=100,Bp=False):
    eta_pos_list=[]
    MI_pos_list=[]
    executor=MPIPoolExecutor()
    inputs=[(L,Bp) for _ in range(es)]
    executor_pool=executor.starmap(MI_pool,inputs)
    executor.shutdown()
    for result in executor_pool:
        eta,MI=result
        eta_pos_list.append(eta)
        MI_pos_list.append(MI)
    return eta_pos_list,MI_pos_list
    
def MI_pool(L,Bp):
    params=Params(L=L,bc=-1,basis='m',T=0)
    x=2*np.random.choice(np.arange(params.L),4,replace=False)
    x.sort()
    subregionAp=np.arange(x[1],x[2],2)
    params.measure_all(1,subregionAp)
    if Bp:
        subregionBp=np.concatenate([np.arange(x[3],2*params.L,2),np.arange(0,x[0],2)])
        params.measure_all(1,subregionBp)
    eta,MI=params.CFT_correlator(x)
    return eta,MI

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    parser.add_argument('--L',default=128,type=int)
    parser.add_argument('--Bp',default=False,type=bool)
    args=parser.parse_args()

    eta_pos_list,MI_pos_list=mutual_info_run_MPI(args.L,args.es,args.Bp)

    with open('MI_pos_Ap{}_L{:d}_es{:d}.pickle'.format('Bp'*args.Bp,args.L,args.es),'wb') as f:
        pickle.dump([eta_pos_list,MI_pos_list],f)