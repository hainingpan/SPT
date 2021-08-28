from Majorana_chain import *
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor

def parallel_func_inf_L(TL):
    T,dmax=TL
    # all projections to s=-1
    params=Params(delta=1,L=np.inf,bc=-1,T=T,dmax=dmax,basis='f')
    subA=np.arange(16)
    subB=np.arange(dmax-16,dmax)
    subAp=np.arange(16,dmax-16)
    params.measure_all(0,proj_range=subAp)
    return params.mutual_information_m(subA,subB)

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--Lmin',default=6,type=int)
    parser.add_argument('--Lmax',default=10,type=int)
    args=parser.parse_args()
    executor=MPIPoolExecutor()

    dmax_list=2**np.arange(args.Lmin,args.Lmax)
    MI_neg_infL=np.zeros((dmax_list.shape[0],50))
    T_list=np.zeros((dmax_list.shape[0],50))
    inflection_infL_list=np.zeros(dmax_list.shape[0])
    Tmax=6e-1
    for dmax_index,dmax in enumerate(dmax_list):
        print(dmax)
        T_list[dmax_index]=np.linspace(0,Tmax,50)
        input_list=[(T,dmax) for T in T_list[dmax_index]]
        pool=map(parallel_func_inf_L,input_list)
        for r_i,result in enumerate(pool):
            MI_neg_infL[dmax_index,r_i]=result
        inflection_infL_list[dmax_index]=find_inflection(T_list[dmax_index],MI_neg_infL[dmax_index]/np.log(2))
        Tmax=inflection_infL_list[dmax_index]
        
    executor.shutdown()

    with open('Maj_infL_occ_T.pickle'.format(),'wb') as f:
        pickle.dump([dmax_list,T_list,MI_neg_infL,inflection_infL_list],f)
