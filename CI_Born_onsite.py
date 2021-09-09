from Chern_insulator import *
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor
from copy import copy
import itertools

def run(p):
    Lx,Ly,m,prob=p
    params=Params(m=m,Lx=Lx,Ly=Ly)
    # prob=list(np.random.permutation([1,0]*(Lx//4*Ly)))
    params.measure_all_Born([np.arange(Lx//4,Lx//2),np.arange(Ly)],prob=prob)
    LN=params.log_neg([np.arange(Lx//4),np.arange(Ly)],[np.arange(Lx//4)+Lx//2,np.arange(Ly)])
    fn=params.fermion_number([np.arange(Lx//4,Lx//2),np.arange(Ly)])
    total_prob=params.snap_prob([np.arange(Lx//4,Lx//2),np.arange(Ly)],occ=np.array(prob)).prod()
    return LN,fn,total_prob


if __name__=="__main__":   
    st=time.time()
    parser=argparse.ArgumentParser()
    parser.add_argument('--kmax',default=2,type=int)
    parser.add_argument('--Lx',default=32,type=int)
    parser.add_argument('--Ly',default=4,type=int)
    parser.add_argument('--m',default=1,type=int)

    args=parser.parse_args()
    executor=MPIPoolExecutor()
    kmax=args.kmax
    Lx=args.Lx
    Ly=args.Ly
    m=args.m
    prob=([1,0]*(Lx//4*Ly))
    prob1_list=[[prob,]]
    for k in range(1,kmax):
        prob_k_list=[]
        nchoosek=list(itertools.combinations(np.arange(Lx*Ly//4),k))
        for i1 in (nchoosek):
            for i2 in (nchoosek):
                prob_k=copy(prob)
                prob_k=np.array(prob_k)
                prob_k[2*np.array(list(i1))]=0
                prob_k[2*np.array(list(i2))+1]=1
                prob_k_list.append(list(prob_k))
        prob1_list.append(prob_k_list)
    
    LN_map=[]
    fn_map=[]
    total_prob_map=[]
    for k,prob_list in enumerate(prob1_list):
        st=time.time()
        inputs=[(Lx,Ly,m,prob) for prob in prob_list]
        LN_list=np.zeros(len(prob_list))
        fn_list=np.zeros(len(prob_list))
        total_prob_list=np.zeros(len(prob_list))
        pool=executor.map(run, inputs)
        # pool=map(run, inputs)
        for es_i,result in enumerate(pool):
            LN,fn,total_prob=result
            LN_list[es_i]=LN
            fn_list[es_i]=fn
            total_prob_list[es_i]=total_prob
        LN_map.append(LN_list)
        fn_map.append(fn_list)
        total_prob_map.append(total_prob_list)
        
        with open('CI_kmax{:d}_Lx{:d}_Ly{:d}_m{:.1f}.pickle'.format(kmax,Lx,Ly,m),'wb') as f:
            pickle.dump([LN_map,fn_map,total_prob_map],f)

        print('k={:d}, len={:d}, {:.1f}'.format(k,len(prob_list),time.time()-st))

