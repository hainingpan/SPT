
from SSH import *
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor
from copy import copy

def run(p):
    L0,params=p
    subA=(np.arange(0,params.L//2)+L0)%(2*params.L)
    subAp_proj=(np.arange(0,params.L//2,2)+L0+params.L//2)%(2*params.L)
    subB=(np.arange(0,params.L//2)+L0+params.L)%(2*params.L)
    params_Born=copy(params)
    params_Born.measure_all_Born(type='link',proj_range=subAp_proj)
    MI=(params_Born.mutual_information_m(subA,subB))
    LN=(params_Born.log_neg(subA,subB))
    return MI,LN

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=10,type=int)
    parser.add_argument('--L',default=64,type=int)
    parser.add_argument('--ds',default=0,type=int)
    parser.add_argument('--var',default=0,type=float)

    args=parser.parse_args()
    executor=MPIPoolExecutor()
        
    delta_list=np.linspace(-1,1,51)**3
    MI_Born_map=[]
    LN_Born_map=[]
    bandstructure_map=[]
    var=args.var
    L=args.L
    es=args.es
    ds=args.ds
    disorder_map=[]
    for disorder_index in range(ds):
        disorder=(np.random.normal(loc=0,scale=var,size=L*2))
        disorder=disorder-disorder.mean()
        disorder_map.append(disorder)
        MI_Born_list=[]
        LN_Born_list=[]
        bandstructure_list=[]
        for delta in delta_list:
            params=Params(delta=delta,L=L,bc=-1,disorder=disorder)
            params.bandstructure()
            bandstructure_list.append(params.val)

            MI_Born_ensemble=np.zeros(es*L)
            LN_Born_ensemble=np.zeros(es*L)
            input=[(L0,params) for L0 in range(0,2*L,2) for _ in range(es)]
            pool=executor.map(run,input)
            MI_list=[]
            LN_list=[]
            for r_i,result in enumerate(pool):
                MI,LN=result
                MI_list.append(MI)
                LN_list.append(LN)
            MI_Born_list.append(MI_list)
            LN_Born_list.append(LN_list)
        MI_Born_map.append(MI_Born_list)
        LN_Born_map.append(LN_Born_list)
        bandstructure_map.append(bandstructure_list)
            
    executor.shutdown()
    MI_Born_map=np.array(MI_Born_map)
    LN_Born_map=np.array(LN_Born_map)
    bandstructure_map=np.array(bandstructure_map)

    with open('SSH_disorder_L{:d}_var{:.1f}_es{:d}_ds{:d}.pickle'.format(L,var,es,ds),'wb') as f:
        pickle.dump([delta_list,MI_Born_map,LN_Born_map,disorder_map,bandstructure_map],f)

