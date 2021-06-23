from Chern_insulator import *
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
import pickle
import argparse

def run(p):
    m,Lx,Ly=p
    params=Params(m=m,Lx=Lx,Ly=Ly,bcx=-1,bcy=1,history=True)
    params.measure_all_Born([np.arange(Lx//4,Lx//2),np.arange(Ly)],type='link')
    LN=params.log_neg([np.arange(Lx//4),np.arange(Ly)],[np.arange(Lx//4)+Lx//2,np.arange(Ly)])
    MI=params.mutual_information_m([np.arange(Lx//4),np.arange(Ly)],[np.arange(Lx//4)+Lx//2,np.arange(Ly)])
    outcome=[np.sum(np.array(params.s_history)==s) for s in ['o+','o-','e+','e-']]
    return MI,LN,outcome

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--Lx',default=128,type=int)
    parser.add_argument('--Ly',default=4,type=int)
    parser.add_argument('--es',default=1000,type=int)
    parser.add_argument('--num',default=10,type=int)

    args=parser.parse_args()

    m_list=np.linspace(1,3,args.num)
    es=args.es
    LN_Born_link_list=[]
    MI_Born_link_list=[]
    outcome_Born_link_list=[]
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
        outcome_ensemble_list=[]
        for result in ensemble_list_pool[m_i]:
            MI,LN,outcome=result
            MI_ensemble_list.append(MI)
            LN_ensemble_list.append(LN)
            outcome_ensemble_list.append(outcome)
        print('gather all m_i={:d}:{:.1f}'.format(m_i,time.time()-st))
        MI_Born_link_list.append(MI_ensemble_list)
        LN_Born_link_list.append(LN_ensemble_list)
        outcome_Born_link_list.append(outcome_ensemble_list)
    executor.shutdown()
    MI_Born_link_list=np.array(MI_Born_link_list)
    LN_Born_link_list=np.array(LN_Born_link_list)
    outcome_Born_link_list=np.array(outcome_Born_link_list)
    
    with open('MI_LN_CI_Born_link_En{:d}_Lx{:d}_Ly{:d}.pickle'.format(args.es,args.Lx,args.Ly),'wb') as f:
        pickle.dump([m_list,MI_Born_link_list,LN_Born_link_list,outcome_Born_link_list],f)