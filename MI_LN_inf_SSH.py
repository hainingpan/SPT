from SSH import *
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
from copy import copy

def run(params_init,subregionA,subregionB,subregionAp,type):
    params=copy(params_init)
    params.measure_all_Born(subregionAp,type=type)
    MI=params.mutual_information_m(subregionA,subregionB)
    LN=params.log_neg(subregionA,subregionB)
    return MI,LN

if __name__=="__main__":
    # if rank==0:
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    parser.add_argument('--min',default=8,type=int)
    parser.add_argument('--max',default=100,type=int)
    parser.add_argument('--timing',default=False,type=bool)
    parser.add_argument('--type',type=str)
    parser.add_argument('--density_numerator',type=int,default=1)
    parser.add_argument('--density_denominator',type=int,default=1)
    args=parser.parse_args()
    if args.timing:
        st=time.time()
    dist_list=np.arange(args.min,args.max)
    dist_list=dist_list[dist_list%args.density_denominator==0]
    # print(dist_list)
    L=np.inf
    eta_inf_Born_Ap_list=[]
    MI_inf_Born_Ap_list=[]
    LN_inf_Born_Ap_list=[]
    step=2 if args.type=='onsite' else 4
    params_init=Params(delta=0,L=L,bc=-1,dmax=args.max+32,history=False)
    for d in dist_list:
        print("d={:d}:".format(d),end='')
        st=time.time()
        MI_ensemble_list=[]
        LN_ensemble_list=[]
        x=np.array([0,16,16+d,32+d])*4
        subregionA=np.arange(x[0],x[1])
        subregionB=np.arange(x[2],x[3])
        subregionAp=np.arange(x[1],x[2],step)
        # print(subregionAp)
        measured=args.density_numerator*subregionAp.shape[0]//args.density_denominator
        # print(measured)
        subregionAp_list=[sorted(np.random.choice(subregionAp,measured,replace=False)) for _ in range(args.es)]
        # print(subregionAp_list)
        eta=cross_ratio(x,L)
        executor=MPIPoolExecutor()
        inputs=[(params_init,subregionA,subregionB,subregionAp,args.type) for subregionAp in subregionAp_list]
        mutual_info_ensemble_list_pool=executor.starmap(run,inputs)
        executor.shutdown()
        for result in mutual_info_ensemble_list_pool:
            MI,LN=result
            MI_ensemble_list.append(MI)
            LN_ensemble_list.append(LN)

        eta_inf_Born_Ap_list.append(eta)
        MI_inf_Born_Ap_list.append(MI_ensemble_list)
        LN_inf_Born_Ap_list.append(LN_ensemble_list)
        print("{:.1f}".format(time.time()-st))
        
    eta_inf_Born_Ap_list=np.array(eta_inf_Born_Ap_list)
    MI_inf_Born_Ap_list=np.array(MI_inf_Born_Ap_list)
    LN_inf_Born_Ap_list=np.array(LN_inf_Born_Ap_list)

    with open('MI_LN_SSH_inf_Born_En{:d}_{:s}_den({:d},{:d})_dist({:d},{:d}).pickle'.format(args.es,args.type,args.density_numerator,args.density_denominator,args.min,args.max),'wb') as f:
        pickle.dump([dist_list,eta_inf_Born_Ap_list,MI_inf_Born_Ap_list,LN_inf_Born_Ap_list],f)


    if args.timing:
            print('Elapsed:{:.1f}'.format(time.time()-st))
