from Majorana_chain import *
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor


def mutual_info_run_MPI(T,es):
    delta_list=np.linspace(-1,1,50)**3
    mutual_info_dis_list=[]
    log_neg_dis_list=[]
    s_history_dis_list=[]
    ensemblesize=es

    for delta in delta_list:
        mutual_info_ensemble_list=[]
        log_neg_ensemble_list=[]
        mutual_info_ensemble_list_pool=[]
        # s_history_list=[]
        executor=MPIPoolExecutor()
        inputs=[(delta,T) for _ in range(ensemblesize)]
        mutual_info_ensemble_list_pool=executor.starmap(MI_pool,inputs)
        executor.shutdown()
        for result in mutual_info_ensemble_list_pool:
            # MI,LN,s_history=result
            LN=result
            # mutual_info_ensemble_list.append(MI)
            log_neg_ensemble_list.append(LN)
            # s_history_list.append(s_history)
        # mutual_info_dis_list.append(mutual_info_ensemble_list)
        log_neg_dis_list.append(log_neg_ensemble_list)
        # s_history_dis_list.append(s_history_list)

    # return delta_list,mutual_info_dis_list,log_neg_dis_list,s_history_dis_list
    return delta_list,log_neg_dis_list
def MI_pool(delta,T):
    params=Params(delta=delta,L=64,bc=-1,basis='m',T=T)
    params.measure_all_Born()
    # MI=params.mutual_information_m(np.arange(params.L//2),np.arange(params.L//2)+params.L)
    LN=params.log_neg(np.arange(params.L//2),np.arange(params.L//2)+params.L)
    # return MI,LN,params.s_history
    return LN

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=6,type=int)
    args=parser.parse_args()

    delta_dict={}
    mutual_info_dis_dict={}
    log_neg_dis_dict={}
    s_history_dis_dict={}
    # T_list=[0]
    T_list=np.linspace(0,8e-1,50)
    # T_list=(0,0.01,0.02,0.05,0.1,0.2,0.3)
    for T in T_list:
        st=time.time()
        # delta_dict[T],mutual_info_dis_dict[T],log_neg_dis_dict[T],s_history_dis_dict[T]=mutual_info_run_MPI(T,args.es)
        delta_dict[T],log_neg_dis_dict[T]=mutual_info_run_MPI(T,args.es)
        print("Time elapsed for {:.4f}: {:.4f}".format(T,time.time()-st))


    with open('MI_Born_En{:d}_T.pickle'.format(args.es),'wb') as f:
        # pickle.dump([delta_dict,mutual_info_dis_dict,log_neg_dis_dict,s_history_dis_dict],f)
        pickle.dump([delta_dict,log_neg_dis_dict],f)
    
   