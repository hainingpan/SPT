from SSH import *
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor
from copy import copy 
import time

def run(p):
    subregionA,subregionB,subregionAp,L,disorder,disorder_J,delta=p
    params=Params(delta=delta,L=L,bc=-1,disorder=disorder,disorder_J=disorder_J)
    LN0=params.log_neg(subregionA,subregionB)
    params.measure_all_Born(subregionAp,type='link')
    LN1=params.log_neg(subregionA,subregionB)
    return LN0,LN1

if __name__=="__main__":  
    st=time.time()
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=10,type=int)
    parser.add_argument('--L',default=64,type=int)
    parser.add_argument('--ds',default=1,type=int)
    parser.add_argument('--ps',default=100,type=int)
    parser.add_argument('--var',default=0,type=float)
    parser.add_argument('--disorder_type',default='disorder_J',type=str)
    

    args=parser.parse_args()
    executor=MPIPoolExecutor()
        
    bandstructure_list=[]
    disorder_map=[]
    disorder_J_map=[]
    LN0_Born_map=[]
    LN1_Born_map=[]
    eta_Born_map=[]
    es=args.es
    ps=args.ps
    ds=args.ds
    L=args.L
    disorder_type=args.disorder_type
    var=args.var
    for disorder_index in range(ds):
        if disorder_type=='disorder':
            disorder=(np.random.normal(loc=0,scale=var,size=L*2))
            disorder=disorder-disorder.mean()
            disorder_J=1
        elif disorder_type=='disorder_J':
            disorder=0
            disorder_J=randomize_J(2*L,var)
        else:
            raise ValueError('disorder type ({:s}) not recognized'.format(disorder_type))

        disorder_map.append(disorder)
        disorder_J_map.append(disorder_J)

        gap_list=[]
        delta_list=np.linspace(-1,1,101)
        for delta in delta_list:
            params=Params(delta=delta,L=L,bc=-1,disorder=disorder,disorder_J=disorder_J)
            params.bandstructure()
            gap_list.append(params.val[L]-params.val[L-1])

        delta=delta_list[np.argmin(gap_list)]

        eta_Born_list=[]
        LN0_Born_list=[]
        LN1_Born_list=[]
        for _ in range(ps):
            x=sorted(2*np.random.choice(np.arange(0,L),4,replace=False))
            # x[0]=0
            subregion=[]
            subregion.append(np.arange(x[0],x[1]))
            subregion.append(np.arange(x[1],x[2]))
            subregion.append(np.arange(x[2],x[3]))
            subregion.append(np.hstack([np.arange(x[3],2*L),np.arange(0,x[0])]))
            proj_start=np.random.choice(np.arange(4),1).item()
            proj_index=np.arange(proj_start,proj_start+4)%4
            subregion1=[subregion[i] for i in proj_index]
            eta=cross_ratio(subregion1,2*L)
            inputs=[(subregion1[0],subregion1[2],subregion1[1][::2],L,disorder,disorder_J,delta) for _ in range(es)]
            pool=executor.map(run,inputs)
            LN0_ensemble_list=[]
            LN1_ensemble_list=[]
            for _,result in enumerate(pool):
                LN0,LN1=result
                LN0_ensemble_list.append(LN0)
                LN1_ensemble_list.append(LN1)

            eta_Born_list.append(eta)
            LN0_Born_list.append(LN0_ensemble_list)
            LN1_Born_list.append(LN1_ensemble_list)

        LN0_Born_map.append(LN0_Born_list)
        LN1_Born_map.append(LN1_Born_list)
        eta_Born_map.append(eta_Born_list)

    eta_Born_map=np.array(eta_Born_map)
    LN0_Born_map=np.array(LN0_Born_map)
    LN1_Born_map=np.array(LN1_Born_map)

    with open('SSH_disorder_delta0_L{:d}_var{:.1f}_es{:d}_ds{:d}_ps{:d}{:s}.pickle'.format(L,var,es,ds,ps,'_'+disorder_type),'wb') as f:
        pickle.dump([eta_Born_map,LN0_Born_map,LN1_Born_map,disorder_map,disorder_J_map,delta],f)

    print('{:f}'.format(time.time()-st))