from Majorana_chain import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor


def mutual_info_run_MPI(s_prob,es=100):
    delta_list=np.linspace(-1,1,100)**3
    mutual_info_dis_list=[]
    if s_prob==0 or s_prob ==1:
        ensemblesize=1
    else:
        ensemblesize=es

    for delta in delta_list:
        mutual_info_ensemble_list=[]
        mutual_info_ensemble_list_pool=[]
        executor=MPIPoolExecutor()
        inputs=[(delta,s_prob) for _ in range(ensemblesize)]
        mutual_info_ensemble_list_pool=executor.starmap(MI_pool,inputs)
        executor.shutdown()
        for result in mutual_info_ensemble_list_pool:
            mutual_info_ensemble_list.append(result)
        mutual_info_dis_list.append(mutual_info_ensemble_list)
    
    return delta_list,mutual_info_dis_list

def MI_pool(delta,s_prob):
    params=Params(delta=delta,L=64,bc=-1,basis='m')
    params.measure_all_position(s_prob)
    return params.mutual_information_m(np.arange(int(params.L/2)),np.arange(params.L/2)+params.L)

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    args=parser.parse_args()


    delta_dict={}
    mutual_info_dis_dict={}
    s_prob_list=np.linspace(0,16,5)/16
    
    for i in s_prob_list:
        print(i)
        st=time.time()
        delta_dict[i],mutual_info_dis_dict[i]=mutual_info_run_MPI(i,args.es)
        print(time.time()-st)

    with open('mutual_info_position_En{:d}.pickle'.format(args.es),'wb') as f:
        pickle.dump([delta_dict,mutual_info_dis_dict],f)
    
    fig,ax=plt.subplots()
    for i in s_prob_list:
        ax.plot(delta_dict[i],np.array(mutual_info_dis_dict[i]).mean(axis=1)/np.log(2),label='Prob(s=0): {:.2f}'.format(i))

    ax.legend()
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'Mutual information between A and B [$\log2$]')

    fig.savefig('mutual_info_position_En{:d}.pdf'.format(args.es),bbox_inches='tight')