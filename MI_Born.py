from Majorana_chain import *
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor


def mutual_info_run_MPI(T,es=100):
    delta_list=np.linspace(-1,1,100)**3
    mutual_info_dis_list=[]
    s_history_dis_list=[]
    ensemblesize=es

    for delta in delta_list:
        mutual_info_ensemble_list=[]
        mutual_info_ensemble_list_pool=[]
        s_history_list=[]
        executor=MPIPoolExecutor()
        inputs=[(delta,T) for _ in range(ensemblesize)]
        mutual_info_ensemble_list_pool=executor.starmap(MI_pool,inputs)
        executor.shutdown()
        for result in mutual_info_ensemble_list_pool:
            MI,s_history=result
            mutual_info_ensemble_list.append(MI)
            s_history_list.append(s_history)
        mutual_info_dis_list.append(mutual_info_ensemble_list)
        s_history_dis_list.append(s_history_list)

    return delta_list,mutual_info_dis_list,s_history_dis_list

def MI_pool(delta,T):
    params=Params(delta=delta,L=64,bc=-1,basis='m',T=T)
    params.measure_all_born()
    return params.mutual_information_m(np.arange(params.L//2),np.arange(params.L//2)+params.L),params.s_history

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=6,type=int)
    args=parser.parse_args()


    delta_dict={}
    mutual_info_dis_dict={}
    s_history_dis_dict={}
    # T_list=[0]
    T_list=np.linspace(0,6e-1,50)
    # T_list=(0,0.01,0.02,0.05,0.1,0.2,0.3)
    for T in T_list:
        st=time.time()
        delta_dict[T],mutual_info_dis_dict[T],s_history_dis_dict[T]=mutual_info_run_MPI(T,args.es)
        print("Time elapsed for {:.4f}: {:.4f}".format(T,time.time()-st))


    with open('mutual_info_Born_En{:d}_alternating.pickle'.format(args.es),'wb') as f:
        pickle.dump([delta_dict,mutual_info_dis_dict,s_history_dis_dict],f)
    
    # fig,ax=plt.subplots()
    # for i in s_prob_list:

    # ax.plot(delta_dict[i],np.array(mutual_info_dis_dict[i]).mean(axis=1)/np.log(2),label='Born rule')

    # ax.legend()
    # ax.set_xlabel(r'$\delta$')
    # ax.set_ylabel(r'Mutual information between A and B [$\log2$]')

    # fig.savefig('mutual_info_Born_En{:d}.pdf'.format(args.es),bbox_inches='tight')