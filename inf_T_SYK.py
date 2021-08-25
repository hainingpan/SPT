from SYK import *
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor

def run(p):
    L,ty,Bp,random=p
    params=Params(L=L)

    LN_0=params.log_neg(np.arange(L//4),np.arange(L//4)+L//2)
    MI_0=params.mutual_information_m(np.arange(L//4),np.arange(L//4)+L//2)

    if ty!='no':
        step=1*(ty=='onsite')+2*(ty=='link')
        if random:
            assert not Bp, 'Bp cannot be true while random is True'
            subA=np.arange(0,L//4,step)
            subAp=np.arange(0,L//4,step)+L//4
            subB=np.arange(0,L//4,step)+L//4*2
            subBp=np.arange(0,L//4,step)+L//4*3
            proj_range_all=np.hstack([subAp,subBp])
            proj_range=np.sort(np.random.choice(proj_range_all,proj_range_all.shape[0],replace=False))
            params.measure_all_Born(type=ty,proj_range=proj_range)
        else:
            params.measure_all_Born(type=ty)
            if Bp:
                params.measure_all_Born(proj_range=np.arange(L*3//2,L*2,step),type=ty)
        
    LN=params.log_neg(np.arange(L//4),np.arange(L//4)+L//2)
    MI=params.mutual_information_m(np.arange(L//4),np.arange(L//4)+L//2)
    return MI_0,LN_0,MI,LN,params.E_mean

if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=4,type=int)
    parser.add_argument('--type',default='no',type=str)
    parser.add_argument('--Bp',default=False,type=bool)
    parser.add_argument('--random',default=False,type=bool)
    args=parser.parse_args()
    executor=MPIPoolExecutor()
    # L_list=[16,32,64,96,128,128+64,256]
    L_list=[128,256]
    LN_0_dict={}
    MI_0_dict={}
    LN_dict={}
    MI_dict={}
    E_mean_dict={}
    for L in L_list:
        inputs=inputs=[(L,args.type,args.Bp,args.random) for _ in range(args.es)]
        pool=executor.map(run,inputs)
        LN_0_dict[L]=[]
        MI_0_dict[L]=[]
        LN_dict[L]=[]
        MI_dict[L]=[]
        E_mean_dict[L]=[]
        for result in pool:
            MI_0,LN_0,MI,LN,E_mean=result
            LN_0_dict[L].append(LN_0)
            MI_0_dict[L].append(MI_0)
            LN_dict[L].append(LN)
            MI_dict[L].append(MI)
            E_mean_dict[L].append(E_mean)
        
    executor.shutdown()

    with open('inf_T_SYK_{:s}_Ap{:s}{:s}_sametype.pickle'.format(args.type,args.Bp*'Bp',args.random*'_R'),'wb') as f:
        pickle.dump([MI_0_dict,LN_0_dict,MI_dict,LN_dict,E_mean_dict],f)
