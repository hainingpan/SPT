from Majorana_chain import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor


def mutual_info_run_MPI(s_prob,L):
    delta_list=np.linspace(-1,1,100)**3
    mutual_info_dis_list=[]
 
    params=Params(delta=0,L=L,bc=-1,basis='m')
    proj_range=np.arange(int(params.L/2),params.L,2)
    s_list_list=params.generate_position_list(np.arange(int(params.L/2),params.L,2),s_prob)
    for delta in delta_list:
        mutual_info_ensemble_list=[]
        mutual_info_ensemble_list_pool=[]        
        executor=MPIPoolExecutor()
        inputs=[(delta,proj_range,s_list,L) for s_list in (s_list_list)]
        mutual_info_ensemble_list_pool=executor.starmap(MI_pool,inputs)
        executor.shutdown()
        for result in mutual_info_ensemble_list_pool:
            mutual_info_ensemble_list.append(result)
        mutual_info_dis_list.append(mutual_info_ensemble_list)
    
    return delta_list,mutual_info_dis_list

def MI_pool(delta,proj_range,s_list,L):
    params=Params(delta=delta,L=L,bc=-1,basis='m')
    params.measure_list(proj_range,s_list)
    return params.mutual_information_m(np.arange(int(params.L/2)),np.arange(params.L/2)+params.L)

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--L',default=64,type=int)
    args=parser.parse_args()


    delta_dict={}
    mutual_info_dis_dict={}
    s_prob_list=np.linspace(0,16,5)/16
    
    for i in s_prob_list:
        print(i)
        st=time.time()
        delta_dict[i],mutual_info_dis_dict[i]=mutual_info_run_MPI(i,args.L)
        print(time.time()-st)

    with open('mutual_info_position_exhaust_L{:d}.pickle'.format(args.L),'wb') as f:
        pickle.dump([delta_dict,mutual_info_dis_dict],f)
    
    fig,ax=plt.subplots()
    for i in s_prob_list:
        ax.plot(delta_dict[i],np.array(mutual_info_dis_dict[i]).mean(axis=1)/np.log(2),label='Prob(s=0): {:.2f}'.format(i))

    ax.legend()
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'Mutual information between A and B [$\log2$]')

    fig.savefig('mutual_info_position_exhaust_L{:d}.pdf'.format(args.L),bbox_inches='tight')