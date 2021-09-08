from Majorana_chain import *
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor
from copy import copy

def run(p):
    L,x,type=p
    subregionA=np.arange(x[0],x[1])
    subregionB=np.arange(x[2],x[3])
    subregionAp=np.arange(x[1],x[2])
    params=Params(delta=0,L=L,bc=-1,basis='f')
    if type=='onsite':
        params.measure_all_Born(subregionAp)
    MI=params.mutual_information_m(subregionA,subregionB)
    LN=params.log_neg(subregionA,subregionB)
    return MI,LN

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=10,type=int)
    parser.add_argument('--L',default=64,type=int)
    parser.add_argument('--ps',default=500,type=int)
    parser.add_argument('--type',default='',type=str)

    args=parser.parse_args()
    executor=MPIPoolExecutor()


    es=args.es
    ps=args.ps
    L=args.L
    type=args.type

    eta_list=np.zeros(ps)
    MI_list=np.zeros((ps,es))
    LN_list=np.zeros((ps,es))
    
    for ps_i in range(ps):
        x=sorted(np.random.choice(np.arange(1,L),3,replace=False))
        x=[0]+x
        
        eta=cross_ratio(x,L)
        eta_list[ps_i]=(eta)
        inputs=[(L,x,type) for _ in range(es)]
        pool=executor.map(run,inputs)
        for es_i,result in enumerate(pool):
            MI,LN=result
            MI_list[ps_i,es_i]=MI
            LN_list[ps_i,es_i]=LN
        
    executor.shutdown()

    with open('Majarona_scaling_L{:d}_es{:d}_ps{:d}{:s}.pickle'.format(L,es,ps,'_'+type),'wb') as f:
        pickle.dump([LN_list,MI_list,eta_list,L,es,ps],f)