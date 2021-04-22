from Majorana_chain import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor


def mutual_info_run_MPI(es=100):
    delta_list=np.linspace(-1,1,100)**3
    mutual_info_dis_list=[]
    s_history_dis_list=[]
    ensemblesize=es

    for delta in delta_list:
        mutual_info_ensemble_list=[]
        mutual_info_ensemble_list_pool=[]
        s_history_list=[]
        st=time.time()
        executor=MPIPoolExecutor()
        inputs=[delta for _ in range(ensemblesize)]
        mutual_info_ensemble_list_pool=executor.map(MI_pool,inputs)
        executor.shutdown()
        for result in mutual_info_ensemble_list_pool:
            MI,s_history=result
            mutual_info_ensemble_list.append(MI)
            s_history_list.append(s_history)
        mutual_info_dis_list.append(mutual_info_ensemble_list)
        s_history_dis_list.append(s_history_list)
        print("Time elapsed for {:.4f}: {:.4f}".format(delta,time.time()-st))

    return delta_list,mutual_info_dis_list,s_history_dis_list

def MI_pool(delta):
    params=Params(delta=delta,L=64,bc=-1,basis='m')
    params.measure_all_born()
    return params.mutual_information_m(np.arange(int(params.L/2)),np.arange(params.L/2)+params.L),params.s_history

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    args=parser.parse_args()


    delta_dict={}
    mutual_info_dis_dict={}
    s_history_dis_dict={}

    i=0  
    delta_dict[i],mutual_info_dis_dict[i],s_history_dis_dict[i]=mutual_info_run_MPI(args.es)

    with open('mutual_info_Born_En{:d}.pickle'.format(args.es),'wb') as f:
        pickle.dump([delta_dict,mutual_info_dis_dict,s_history_dis_dict],f)
    
    fig,ax=plt.subplots()
    # for i in s_prob_list:

    ax.plot(delta_dict[i],np.array(mutual_info_dis_dict[i]).mean(axis=1)/np.log(2),label='Born rule')

    ax.legend()
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'Mutual information between A and B [$\log2$]')

    fig.savefig('mutual_info_Born_En{:d}.pdf'.format(args.es),bbox_inches='tight')