from SSH import *
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor


def run(p):
    delta,p_rate,L=p
    params=Params(L=L,delta=delta)
    subA=np.arange(0,L//2,2)
    subAp=np.arange(0,L//2,2)+L//2
    subB=np.arange(0,L//2,2)+L
    subBp=np.arange(0,L//2,2)+L//2*3
    proj_range_bool=np.random.choice([0,1],size=subAp.shape[0],p=[1-p_rate,p_rate])
    proj_range=subAp[proj_range_bool==1]   
    params.measure_all_Born(proj_range=proj_range,type='link')
    MI=params.mutual_information_m(np.arange(0,L//2),np.arange(0,L//2)+L)
    LN=params.log_neg(np.arange(0,L//2),np.arange(0,L//2)+L)
    return MI,LN

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=10,type=int)
    parser.add_argument('--L',default=128,type=int)
    parser.add_argument('--Nd',default=50,type=int)
    parser.add_argument('--type',default='link',type=str)
    parser.add_argument('--Bp',default=False,type=bool)
    parser.add_argument('--random',default=False,type=bool)
    args=parser.parse_args()
    executor=MPIPoolExecutor()
    
    p_rate_list=np.array([0,.8,.9,.95,.97,1])
    delta_list=np.linspace(-1,1,args.Nd)**3

    MI_link_rate_list=np.zeros((p_rate_list.shape[0],delta_list.shape[0],args.es))
    LN_link_rate_list=np.zeros((p_rate_list.shape[0],delta_list.shape[0],args.es))

    for p_rate_i,p_rate in enumerate(p_rate_list):
        print(p_rate)
        for delta_i,delta in enumerate(delta_list):
            sync_results=executor.map(run,[(delta,p_rate,args.L)]*args.es)
            for es_i,result in enumerate(sync_results):
                MI,LN=result
                MI_link_rate_list[p_rate_i,delta_i,es_i]=MI
                LN_link_rate_list[p_rate_i,delta_i,es_i]=LN

    executor.shutdown()
    with open('SSH_p_rate_{:s}_L{:d}_es{:d}.pickle'.format(args.type,args.L,args.es),'wb') as f:
        pickle.dump([MI_link_rate_list,LN_link_rate_list,p_rate_list,delta_list,args],f)