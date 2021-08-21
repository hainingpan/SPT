from Majorana_chain import *
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor


def mutual_info_run_MPI(T,es,L,mtype):
    delta_list=np.linspace(-1,1,50)
    mutual_info_dis_list=[]
    log_neg_dis_list=[]
    ensemblesize=es

    for delta in delta_list:
        log_neg_ensemble_list=[]
        mutual_info_ensemble_list=[]
        mutual_info_ensemble_list_pool=[]
        executor=MPIPoolExecutor()
        inputs=[(delta,T,L,mtype) for _ in range(ensemblesize)]
        mutual_info_ensemble_list_pool=executor.map(MI_pool,inputs)
        executor.shutdown()
        for result in mutual_info_ensemble_list_pool:
            MI,LN=result
            mutual_info_ensemble_list.append(MI)
            log_neg_ensemble_list.append(LN)
        mutual_info_dis_list.append(mutual_info_ensemble_list)
        log_neg_dis_list.append(log_neg_ensemble_list)

    return delta_list,log_neg_dis_list
def MI_pool(p):
    delta,T,L,mtype=p
    params=Params(delta=delta,L=L,bc=-1,basis='f',T=T)
    if mtype=='Born':
        params.measure_all_Born()
    elif mtype=='1':
        params.measure_all(0)
    
    MI=params.mutual_information_m(np.arange(params.L//4),np.arange(params.L//4)+params.L//2)
    LN=params.log_neg(np.arange(params.L//4),np.arange(params.L//4)+params.L//2)
    return MI,LN

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=6,type=int)
    parser.add_argument('--L',default=64,type=int)
    parser.add_argument('--mtype',default='',type=str)
    args=parser.parse_args()

    delta_dict={}
    mutual_info_dis_dict={}
    log_neg_dis_dict={}
    s_history_dis_dict={}
    T_list=np.linspace(0,8e-1,50)
    for T in T_list:
        st=time.time()
        delta_dict[T],log_neg_dis_dict[T]=mutual_info_run_MPI(T,args.es,args.L,args.mtype)
        print("Time elapsed for {:.4f}: {:.4f}".format(T,time.time()-st))


    with open('MI_Born_En{:d}_L{:d}_{:s}_T.pickle'.format(args.es,args.L,args.mtype),'wb') as f:
        pickle.dump([delta_dict,log_neg_dis_dict,T_list],f)
    
   