from pSC import *
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
import pickle
import argparse

def run(p):
    m,Lx,Ly=p
    params=Params(m=m,Lx=Lx,Ly=Ly,bcx=-1,bcy=1)
    # total=params.linearize_index([np.arange(params.Lx),np.arange(params.Ly)],2)
    subA=params.linearize_index([np.arange(params.Lx//4),np.arange(params.Ly)],2)
    proj_range=params.linearize_index([np.arange(params.Lx//4)+params.Lx//4,np.arange(params.Ly)],2,proj=True)
    subB=params.linearize_index([np.arange(params.Lx//4)+params.Lx//2,np.arange(params.Ly)],2)

    params.measure_all_Born(proj_range,linear=True)
    LN=params.log_neg(subA,subB,linear=True)
    MI=params.mutual_information_m(subA,subB,linear=True)
    return MI,LN

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--Lx',default=128,type=int)
    parser.add_argument('--Ly',default=4,type=int)
    parser.add_argument('--es',default=20,type=int)

    args=parser.parse_args()

    m_list=(lambda x:(x-2)**3+2 )(np.linspace(1,3,25))
    es=args.es
    LN_Born_list=[]
    MI_Born_list=[]
    Lx,Ly=args.Lx,args.Ly
    executor=MPIPoolExecutor()
    ensemble_list_pool=[]
    for m_i,m in enumerate(m_list):

        inputs=[(m,Lx,Ly) for _ in range(es)]
        ensemble_list_pool.append(executor.map(run,inputs))
    for m_i,m in enumerate(m_list):
        print("m_i={:d}:".format(m_i))
        st=time.time()
        MI_ensemble_list=[]
        LN_ensemble_list=[]
        for result in ensemble_list_pool[m_i]:
            MI,LN=result
            MI_ensemble_list.append(MI)
            LN_ensemble_list.append(LN)
        print('gather all m_i={:d}:{:.1f}'.format(m_i,time.time()-st))
        MI_Born_list.append(MI_ensemble_list)
        LN_Born_list.append(LN_ensemble_list)
    executor.shutdown()
    MI_Born_list=np.array(MI_Born_list)
    LN_Born_list=np.array(LN_Born_list)
    
    with open('MI_LN_pSC_Born_En{:d}_Lx{:d}_Ly{:d}.pickle'.format(args.es,args.Lx,args.Ly),'wb') as f:
        pickle.dump([m_list,MI_Born_list,LN_Born_list],f)