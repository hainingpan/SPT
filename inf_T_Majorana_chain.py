from Majorana_chain import *
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor

def run(p):
    L,ty=p
    params=Params(delta=0,L=L,T=np.inf,E0=0,dE=1,basis='f')
    if ty=='onsite':
        params.measure_all_Born()
    LN=params.log_neg(np.arange(L//4),np.arange(L//4)+L//2)
    MI=params.mutual_information_m(np.arange(L//4),np.arange(L//4)+L//2)
    return MI,LN,params.E_mean

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=6,type=int)
    # parser.add_argument('--Lmin',default=16,type=int)
    # parser.add_argument('--Lmax',default=128,type=int)
    parser.add_argument('--type',default='no',type=str)
    args=parser.parse_args()
    executor=MPIPoolExecutor()
    L_list=[16,32,64,96,128,128+64,256]
    LN_dict={}
    MI_dict={}
    E_mean_dict={}
    for L in L_list:
        inputs=inputs=[(L,args.type) for _ in range(args.es)]
        pool=executor.map(run,inputs)
        LN_dict[L]=[]
        MI_dict[L]=[]
        E_mean_dict[L]=[]
        for result in pool:
            MI,LN,E_mean=result
            LN_dict[L].append(LN)
            MI_dict[L].append(MI)
            E_mean_dict[L].append(E_mean)
        
    executor.shutdown()

    with open('inf_T_Maj_{:s}.pickle'.format(args.type),'wb') as f:
        pickle.dump([MI_dict,LN_dict,E_mean_dict],f)
