from SSH import *
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
from copy import copy
# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size= comm.Get_size()

def run(es,pt,L,Bp,type):
    eta_Born_list=[]
    MI_Born_list=[]
    LN_Born_list=[]
    step=2 if type=='onsite' else 4
    for _ in range(pt):
        MI_ensemble_list=[]
        LN_ensemble_list=[]
        x=sorted(4*np.random.choice(np.arange(1,L),3,replace=False))
        x=[0]+x
        subregionA=np.arange(x[0],x[1])
        subregionB=np.arange(x[2],x[3])
        subregionAp=np.arange(x[1],x[2],step)
        if Bp:
            subregionBp=np.concatenate([np.arange(x[3],2*L,step),np.arange(0,x[0],step)])
        eta=cross_ratio(x,4*L)
        for _ in range(es):
            if Bp:
                params=Params(delta=0,L=L,bc=-1).measure_all_Born(subregionAp,type=type).measure_all_Born(subregionBp,type=type)
            else:
                params=Params(delta=0,L=L,bc=-1).measure_all_Born(subregionAp,type=type)
            MI_ensemble_list.append(params.mutual_information_m(subregionA,subregionB))
            LN_ensemble_list.append(params.log_neg(subregionA,subregionB))
        eta_Born_list.append(eta)
        MI_Born_list.append(MI_ensemble_list)
        LN_Born_list.append(LN_ensemble_list)
    return np.array(eta_Born_list),np.array(MI_Born_list),np.array(LN_Born_list)

def run(params_init,subregionA,subregionB,subregionAp,type):
    params=copy(params_init)
    params.measure_all_Born(subregionAp,type=type)
    MI=params.mutual_information_m(subregionA,subregionB)
    LN=params.log_neg(subregionA,subregionB)
    return MI,LN

if __name__=="__main__":
    # if rank==0:
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    parser.add_argument('--min',default=8,type=int)
    parser.add_argument('--max',default=100,type=int)
    parser.add_argument('--timing',default=False,type=bool)
    parser.add_argument('--type',type=str)
    args=parser.parse_args()
    if args.timing:
        st=time.time()
    dist_list=range(args.min,args.max)
    L=np.inf
    eta_inf_Born_Ap_list=[]
    MI_inf_Born_Ap_list=[]
    LN_inf_Born_Ap_list=[]
    params_init=Params(delta=0,L=L,bc=-1,dmax=args.max+32)
    for d in dist_list:
        MI_ensemble_list=[]
        LN_ensemble_list=[]
        x=np.array([0,16,16+d,32+d])*4
        subregionA=np.arange(x[0],x[1])
        subregionB=np.arange(x[2],x[3])
        subregionAp=np.arange(x[1],x[2],4)
        eta=cross_ratio(x,L)
        executor=MPIPoolExecutor()
        inputs=[(params_init,subregionA,subregionB,subregionAp,args.type) for _ in range(args.es)]
        mutual_info_ensemble_list_pool=executor.starmap(run,inputs)
        executor.shutdown()
        for result in mutual_info_ensemble_list_pool:
            MI,LN=result
            MI_ensemble_list.append(MI)
            LN_ensemble_list.append(LN)

        eta_inf_Born_Ap_list.append(eta)
        MI_inf_Born_Ap_list.append(MI_ensemble_list)
        LN_inf_Born_Ap_list.append(LN_ensemble_list)
        
    eta_inf_Born_Ap_list=np.array(eta_inf_Born_Ap_list)
    MI_inf_Born_Ap_list=np.array(MI_inf_Born_Ap_list)
    LN_inf_Born_Ap_list=np.array(LN_inf_Born_Ap_list)

    with open('MI_LN_SSH_inf_Born_En{:d}_{:s}_({:d},{:d}).pickle'.format(args.es,args.type,args.min,args.max),'wb') as f:
        pickle.dump([dist_list,eta_inf_Born_Ap_list,MI_inf_Born_Ap_list,LN_inf_Born_Ap_list],f)


    if args.timing:
            print('Elapsed:{:.1f}'.format(time.time()-st))

        
