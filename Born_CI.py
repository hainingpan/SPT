from Chern_insulator import *
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
from copy import copy

def run(params_init,subregionA,subregionB,subregionAp):
    params=copy(params_init)
    params.measure_all_Born(subregionAp)
    MI=params.mutual_information_m(subregionA,subregionB)
    LN=params.log_neg(subregionA,subregionB)
    return MI,LN

if __name__=="__main__":
    # if rank==0:
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    parser.add_argument('--min',default=1,type=float)
    parser.add_argument('--max',default=3,type=float)
    parser.add_argument('--num',default=50,type=int)
    parser.add_argument('--Lx',default=32,type=int)
    parser.add_argument('--Ly',default=16,type=int)
    parser.add_argument('--timing',default=False,type=bool)
    args=parser.parse_args()
    if args.timing:
        st=time.time()
    m_list=np.linspace(args.min,args.max,args.num)
    MI_Born_list=[]
    LN_Born_list=[]
    MI_no_list=[]
    LN_no_list=[]
    st0=time.time()
    for m_i,m in enumerate(m_list):
        params_init=Params(m=m,Lx=args.Lx,Ly=args.Ly,bcx=1,bcy=-1,history=False)
        print("m_i={:d}:".format(m_i),end='')
        st=time.time()
        # Born rule
        MI_ensemble_list=[]
        LN_ensemble_list=[]
        subregionA=[np.arange(args.Lx//4),np.arange(args.Ly)]
        subregionB=[np.arange(args.Lx//4)+args.Lx//2,np.arange(args.Ly)]
        subregionAp=[np.arange(args.Lx//4)+args.Lx//4,np.arange(args.Ly)]
        executor=MPIPoolExecutor()
        inputs=[(params_init,subregionA,subregionB,subregionAp) for _ in range(args.es)]
        ensemble_list_pool=executor.starmap(run,inputs)
        executor.shutdown()
        for result in ensemble_list_pool:
            MI,LN=result
            MI_ensemble_list.append(MI)
            LN_ensemble_list.append(LN)

        MI_Born_list.append(MI_ensemble_list)
        LN_Born_list.append(LN_ensemble_list)

        # no measurement
        MI_no_list.append(params_init.mutual_information_m(subregionA,subregionB))
        LN_no_list.append(params_init.log_neg(subregionA,subregionB))

        print("{:.1f}".format(time.time()-st))

    MI_Born_list=np.array(MI_Born_list)
    LN_Born_list=np.array(LN_Born_list)
    MI_no_list=np.array(MI_no_list)
    LN_no_list=np.array(LN_no_list)

    with open('MI_LN_CI_Born_En{:d}_Lx{:d}_Ly{:d}.pickle'.format(args.es,args.Lx,args.Ly),'wb') as f:
        pickle.dump([m_list,MI_Born_list,LN_Born_list,MI_no_list,LN_no_list],f)

    if args.timing:
            print('Elapsed:{:.1f}'.format(time.time()-st0))