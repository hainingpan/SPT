from Chern_insulator import *
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
from copy import copy

def run(params_init,subregionA,subregionB,subregionAp,subregionBp,Bp):
    params=copy(params_init)
    params.measure_all_Born(subregionAp)
    if Bp:
        params.measure_all_Born(subregionBp)
    MI=params.mutual_information_m(subregionA,subregionB)
    LN=params.log_neg(subregionA,subregionB)
    return MI,LN

if __name__=="__main__":
    # if rank==0:
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    parser.add_argument('--timing',default=False,type=bool)
    parser.add_argument('--Lx',default=32,type=int)
    parser.add_argument('--Ly',default=16,type=int)
    parser.add_argument('--pts',default=100,type=int)
    parser.add_argument('--Bp',default=False,type=bool)

    args=parser.parse_args()
    if args.timing:
        st=time.time()

    eta_Born_list=[]
    MI_Born_list=[]
    LN_Born_list=[]
    params_init=Params(m=2,Lx=args.Lx,Ly=args.Ly)
    executor=MPIPoolExecutor()
    mutual_info_ensemble_list_pool=[]
    for pt in range(args.pts):
        MI_ensemble_list=[]
        LN_ensemble_list=[]
        inputs=[]
        x=sorted(np.random.choice(np.arange(1,args.Lx),3,replace=False))
        x=[0]+x
        eta=cross_ratio(x,args.Lx)
        eta_Born_list.append(eta)
        subregionA=[np.arange(x[0],x[1]),np.arange(params_init.Ly)]
        subregionB=[np.arange(x[2],x[3]),np.arange(params_init.Ly)]
        subregionAp=[np.arange(x[1],x[2]),np.arange(params_init.Ly)]
        subregionBp=[np.arange(x[3],args.Lx),np.arange(params_init.Ly)]
        inputs=[(params_init,subregionA,subregionB,subregionAp,subregionBp,args.Bp) for _ in range(args.es)]
        mutual_info_ensemble_list_pool.append(executor.starmap(run,inputs))
        
    for pt in range(args.pts):
        print("{:d}:".format(pt),end='')   
        st=time.time()
        MI_ensemble_list=[]
        LN_ensemble_list=[]
        for result in mutual_info_ensemble_list_pool[pt]:
            MI,LN=result
            MI_ensemble_list.append(MI)
            LN_ensemble_list.append(LN)
        MI_Born_list.append(MI_ensemble_list)
        LN_Born_list.append(LN_ensemble_list)
        print("{:.1f}".format(time.time()-st))
    
    executor.shutdown()    


    eta_Born_list=np.array(eta_Born_list)
    MI_Born_list=np.array(MI_Born_list)
    LN_Born_list=np.array(LN_Born_list)

    
    with open('MI_LN_CI_Born_En{:d}_pts{:d}_Lx{:d}_Ly{:d}_Ap{:s}'.format(args.es,args.pts,args.Lx,args.Ly,args.Bp*'Bp'),'wb') as f:
        pickle.dump([eta_Born_list,MI_Born_list,LN_Born_list],f)


    if args.timing:
            print('Elapsed:{:.1f}'.format(time.time()-st))
