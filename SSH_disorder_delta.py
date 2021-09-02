from SSH import *
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor
from copy import copy 

def run(p):
    subregionA,subregionB,subregionAp,L,disorder=p
    params=Params(delta=0,L=L,bc=-1,disorder=disorder)
    params.measure_all_Born(subregionAp,type='link')
    return params.log_neg(subregionA,subregionB)

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=10,type=int)
    parser.add_argument('--L',default=64,type=int)
    parser.add_argument('--ds',default=1,type=int)
    parser.add_argument('--ps',default=100,type=int)
    parser.add_argument('--var',default=0,type=float)

    args=parser.parse_args()
    executor=MPIPoolExecutor()
        
    bandstructure_list=[]
    disorder_map=[]
    LN_Born_map=[]
    eta_Born_map=[]
    es=args.es
    ps=args.ps
    ds=args.ds
    L=args.L
    var=args.var
    for disorder_index in range(ds):
        disorder=(np.random.normal(loc=0,scale=var,size=L*2))
        disorder=disorder-disorder.mean()
        disorder_map.append(disorder)

        eta_Born_list=[]
        LN_Born_list=[]
        for _ in range(ps):
            x=sorted(2*np.random.choice(np.arange(0,L),4,replace=False))
            subregion=[]
            subregion.append(np.arange(x[0],x[1]))
            subregion.append(np.arange(x[2],x[3]))
            subregion.append(np.arange(x[1],x[2]))
            subregion.append(np.hstack([np.arange(x[3],2*L),np.arange(0,x[0])]))
            proj_start=np.random.choice(np.arange(4),1).item()
            proj_index=np.arange(proj_start,proj_start+4)%4

            eta=cross_ratio([x[index] for index in proj_index],2*L)
            inputs=[(subregion[proj_index[0]],subregion[proj_index[2]],subregion[proj_index[1]][::2],L,disorder) for _ in range(es)]
            pool=executor.map(run,inputs)
            LN_ensemble_list=[]
            for _,result in enumerate(pool):
                LN_ensemble_list.append(result)

            eta_Born_list.append(eta)
            LN_Born_list.append(LN_ensemble_list)

        LN_Born_map.append(LN_Born_list)
        eta_Born_map.append(eta_Born_list)
        eta_Born_map=np.array(eta_Born_map)
        LN_Born_map=np.array(LN_Born_map)

        with open('SSH_disorder_delta0_L{:d}_var{:.1f}_es{:d}_ds{:d}_ps{:d}.pickle'.format(L,var,es,ds,ps),'wb') as f:
            pickle.dump([eta_Born_map,LN_Born_map,disorder_map],f)