import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy.matlib
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la
import numpy.matlib
import argparse
import pickle
import multiprocessing
from mpi4py.futures import MPIPoolExecutor

from Majorana_chain import *

        

def mutual_info_run(batchsize,es=100):
    delta_list=np.linspace(-1,1,100)**3
    mutual_info_dis_list=[]
    if batchsize==0:
        ensemblesize=1
    else:
        ensemblesize=es

    for delta in delta_list:
        mutual_info_ensemble_list=[]
        for ensemble in range(ensemblesize):
            params=Params(delta=delta,L=64,bc=-1)
            params.measure_all_random_even(batchsize,(int(params.L/2),params.L))
            mutual_info_ensemble_list.append(params.mutual_information_m(np.arange(int(params.L/2)),np.arange(int(params.L/2))+params.L))
        mutual_info_dis_list.append(mutual_info_ensemble_list)
    return delta_list,mutual_info_dis_list

def MI_pool(delta,batchsize):
    params=Params(delta=delta,L=64,bc=-1)
    params.measure_all_random_even(batchsize,(int(params.L/2),params.L))
    return params.mutual_information_m(np.arange(int(params.L/2)),np.arange(int(params.L/2))+params.L)
    

def mutual_info_run_pool(batchsize,es=100):
    delta_list=np.linspace(-1,1,100)**3
    mutual_info_dis_list=[]
    if batchsize==0:
        ensemblesize=1
    else:
        ensemblesize=es

    for delta in delta_list:
        mutual_info_ensemble_list=[]
        mutual_info_ensemble_list_pool=[]
        with multiprocessing.Pool(4) as pool:
            inputs=[(delta,batchsize) for _ in range(ensemblesize)]
            mutual_info_ensemble_list=pool.starmap(MI_pool,inputs)
            # for ensemble in range(ensemblesize):
            #     mutual_info_ensemble_list_pool.append(pool.apply_async(MI,(delta,batchsize)))            
            # for r in mutual_info_ensemble_list_pool:
            #     mutual_info_ensemble_list.append(r.get())
        mutual_info_dis_list.append(mutual_info_ensemble_list)
    return delta_list,mutual_info_dis_list

def mutual_info_run_MPI(batchsize,es=100):
    delta_list=np.linspace(-1,1,100)**3
    mutual_info_dis_list=[]
    if batchsize==0:
        ensemblesize=1
    else:
        ensemblesize=es

    for delta in delta_list:
        mutual_info_ensemble_list=[]
        mutual_info_ensemble_list_pool=[]
        executor=MPIPoolExecutor()
        inputs=[(delta,batchsize) for _ in range(ensemblesize)]
        mutual_info_ensemble_list_pool=executor.starmap(MI_pool,inputs)
        executor.shutdown()
        for result in mutual_info_ensemble_list_pool:
            mutual_info_ensemble_list.append(result)
        mutual_info_dis_list.append(mutual_info_ensemble_list)
    return delta_list,mutual_info_dis_list


if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    args=parser.parse_args()


    delta_dict={}
    mutual_info_dis_dict={}
    density_list=(0,12,14,16)
    
    for i in density_list:
        print(i)
        st=time.time()
        delta_dict[i],mutual_info_dis_dict[i]=mutual_info_run_MPI(i,args.es)
        print(time.time()-st)

    with open('mutual_info_Ap_En{:d}.pickle'.format(args.es),'wb') as f:
        pickle.dump([delta_dict,mutual_info_dis_dict],f)
    
    fig,ax=plt.subplots()
    for i in density_list:
        ax.plot(delta_dict[i],np.array(mutual_info_dis_dict[i]).mean(axis=1)/np.log(2),label='Number of gates: {}'.format(i))

    ax.legend()
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'Mutual information between A and B [$\log2$]')

    fig.savefig('mutual_info_Ap_En{:d}.pdf'.format(args.es),bbox_inches='tight')
