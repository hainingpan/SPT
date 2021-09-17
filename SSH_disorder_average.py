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
    subregionA,subregionB,subregionAp,L,disorder,disorder_J,delta,size_A_i,dis_i,es_i=p
    params=Params(delta=delta,L=L,bc=-1,disorder=disorder,disorder_J=disorder_J)
    LN0=params.log_neg(subregionA,subregionB)
    params.measure_all_Born(subregionAp,type='link')
    LN1=params.log_neg(subregionA,subregionB)
    return LN0,LN1,size_A_i,dis_i,es_i

if __name__=="__main__":  
    st=time.time()
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=10,type=int)
    parser.add_argument('--L',default=64,type=int)
    parser.add_argument('--ds',default=10,type=int)
    parser.add_argument('--var',default=0,type=float)
    parser.add_argument('--disorder_type',default='disorder_J',type=str)
  
    args=parser.parse_args()
    executor=MPIPoolExecutor()
        
    disorder_map=[]
    disorder_J_map=[]
    es=args.es
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


    inputs=[]
    eta_list=[]
    for size_A_i,size_A in enumerate(range(2,L,2)):
        x=[0,size_A,L,L+size_A]
        subregion=[]
        subregion.append(np.arange(x[0],x[1]))
        subregion.append(np.arange(x[1],x[2]))
        subregion.append(np.arange(x[2],x[3]))
        subregion.append(np.hstack([np.arange(x[3],2*L),np.arange(0,x[0])]))
        eta=cross_ratio(subregion,2*L)
        eta_list.append(eta)
        delta=0
        for dis_i,(disorder,disorder_J) in enumerate(zip(disorder_map,disorder_J_map)):
            for es_i in range(es):
                inputs.append((subregion[0],subregion[2],subregion[1][::2],L,disorder,disorder_J,delta,size_A_i,dis_i,es_i))

    eta_list=np.array(eta_list)
    pool=executor.map(run,inputs)
    LN0_list=np.zeros((L//2-1,len(disorder_map),es))
    LN1_list=np.zeros((L//2-1,len(disorder_map),es))
    for _,result in enumerate(pool):
        LN0,LN1,size_A_i,dis_i,es_i=result
        LN0_list[size_A_i,dis_i,es_i]=(LN0)
        LN1_list[size_A_i,dis_i,es_i]=(LN1)

    executor.shutdown()

    with open('SSH_disorder_delta0_L{:d}_var{:.1f}_es{:d}_ds{:d}_{:s}.pickle'.format(L,var,es,ds,'_'+disorder_type),'wb') as f:
        pickle.dump([eta_list,LN0_list,LN1_list,disorder_map,disorder_J_map,delta],f)

    print('{:f}'.format(time.time()-st))