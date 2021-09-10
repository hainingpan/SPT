from Chern_insulator import *
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
from copy import copy

def run(p):
    params_init,subregionA,subregionB,subregionAp,type=p
    params=copy(params_init)
    if not type=='':
        params.measure_all_Born(subregionAp,type=type)
    MI=params.mutual_information_m(subregionA,subregionB)
    LN=params.log_neg(subregionA,subregionB)
    fn=params.fermion_number(subregionAp)
    return MI,LN,fn

if __name__=="__main__":
    # if rank==0:
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    parser.add_argument('--min',default=1,type=float)
    parser.add_argument('--max',default=3,type=float)
    parser.add_argument('--num',default=21,type=int)
    parser.add_argument('--Lx',default=32,type=int)
    parser.add_argument('--Ly',default=4,type=int)
    parser.add_argument('--timing',default=True,type=bool)
    parser.add_argument('--type',default='',type=str)
    args=parser.parse_args()
    if args.timing:
        st=time.time()
    m_list=np.linspace(args.min,args.max,args.num)
    MI_Born_list=np.zeros((args.num,args.es))
    LN_Born_list=np.zeros((args.num,args.es))
    fn_Born_list=np.zeros((args.num,args.es))
    subregionA=[np.arange(args.Lx//4),np.arange(args.Ly)]
    subregionB=[np.arange(args.Lx//4)+args.Lx//2,np.arange(args.Ly)]
    subregionAp=[np.arange(args.Lx//4)+args.Lx//4,np.arange(args.Ly)]
    st=time.time()
    executor=MPIPoolExecutor()
    ensemble_list_pool=[]
    for m_i,m in enumerate(m_list):
        st0=time.time()
        params_init=(Params(m=m,Lx=args.Lx,Ly=args.Ly))
        # Born rule
        inputs=[(params_init,subregionA,subregionB,subregionAp,args.type) for _ in range(args.es)]
        pool=executor.map(run,inputs)
        for es_i,result in enumerate(pool):
            MI,LN,fn=result
            MI_Born_list[m_i,es_i]=MI
            LN_Born_list[m_i,es_i]=LN
            fn_Born_list[m_i,es_i]=fn
        print('m={:.2f} {:.1f}'.format(m,time.time()-st0),flush=True)
    executor.shutdown()

    with open('CI_Born_En{:d}_Lx{:d}_Ly{:d}_{:s}.pickle'.format(args.es,args.Lx,args.Ly,args.type),'wb') as f:
        pickle.dump([m_list,MI_Born_list,LN_Born_list,fn_Born_list],f)

    if args.timing:
            print('Elapsed:{:.1f}'.format(time.time()-st))