from HOTI import *
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor
from copy import copy

def run(p):
    m,Lx,Ly=p
    params=Params(m=m,t=.5,l=.5,Delta=0.25,Lx=Lx,Ly=Ly,bcx=-1,bcy=-1)
    subA=[np.arange(params.Lx//4),np.arange(params.Ly//4)]
    subAp=[np.arange(params.Lx//4)+params.Lx//4,np.arange(params.Ly//4)]
    subB=[np.arange(params.Lx//4)+params.Lx//2,np.arange(params.Ly//4)]

    params.measure_all_Born(subAp)
    MI=params.mutual_information_m(subA,subB)
    LN=params.log_neg(subA,subB)
    fn=params.fermion_number(subAp)
    return MI,LN,fn

if __name__=="__main__":   
    st=time.time()
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=50,type=int)
    parser.add_argument('--Lx',default=24,type=int)
    parser.add_argument('--Ly',default=24,type=int)
    parser.add_argument('--mN',default=11,type=int)

    args=parser.parse_args()
    executor=MPIPoolExecutor()

    es=args.es
    Lx=args.Lx
    Ly=args.Ly
    mN=args.mN

    m_list=np.linspace(1,3,mN)
    MI_list=np.zeros((mN,es))
    LN_list=np.zeros((mN,es))
    fn_list=np.zeros((mN,es))
    for m_i,m in enumerate(m_list):
        st=time.time()
        inputs=[(m,Lx,Ly) for _ in range(es)]
        pool=executor.map(run,inputs)
        for es_i,result in enumerate(pool):
            MI,LN,fn=result
            MI_list[m_i,es_i]=MI
            LN_list[m_i,es_i]=LN
            fn_list[m_i,es_i]=fn

        print('m={:.2f} {:.2f}'.format(m,time.time()-st))
    
    with open('HOTI_Born_es{:d}_Lx{:d}_Ly{:d}_mN{:d}.pickle'.format(es,Lx,Ly,mN),'wb') as f:
        pickle.dump([m_list,MI_list,LN_list,fn_list],f)