from Majorana_chain import *
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
from copy import copy

def run_Born(p):
    delta,r,LA,es,delta_i,L_i=p
    params=Params(delta=delta,L=np.inf,bc=-1,dmax=(2*LA+r)*2,basis='f')
    x=np.array([0,LA,LA+r,2*LA+r])
    subA=np.arange(x[0],x[1])
    subB=np.arange(x[2],x[3])
    subAp=np.arange(x[1],x[2])
    params.measure_all_Born(proj_range=subAp)
    MI=params.mutual_information_m(subA,subB)
    LN=params.log_neg(subA,subB)
    return MI,LN,L_i,delta_i,es

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=200,type=int)
    parser.add_argument('--delta_min',default=-.1,type=float)
    parser.add_argument('--delta_max',default=.1,type=float)
    parser.add_argument('--delta_num',default=101,type=int)
    parser.add_argument('--L_min',default=16,type=int)
    parser.add_argument('--L_max',default=64,type=int)
    args=parser.parse_args()

    
    delta_list=np.linspace(args.delta_min,args.delta_max,args.delta_num)
    L_list=np.arange(args.L_min,args.L_max+1,16)
    MI_list=np.zeros((L_list.shape[0],delta_list.shape[0],args.es))
    LN_list=np.zeros((L_list.shape[0],delta_list.shape[0],args.es))
    inputs=[(delta,L,L,es,delta_i,L_i) for delta_i,delta in enumerate(delta_list) for (L_i,L) in enumerate(L_list) for es in range(args.es)]

    executor=MPIPoolExecutor()
    pool=executor.map(run_Born,inputs)
    for result in pool:
        MI,LN,L_i,delta_i,es=result
        MI_list[L_i,delta_i,es]=MI
        LN_list[L_i,delta_i,es]=LN

    executor.shutdown()

    with open('Majorana_Born_es{:d}.pickle'.format(args.es),'wb') as f:
        pickle.dump([delta_list,L_list,MI_list],f)

