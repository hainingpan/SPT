from HOTI import *
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
from copy import copy

def run(p):
    m,Lx,Ly,Born,m_i,es_i=p
    params=Params(m=m,t=.5,l=.5,Delta=0.25,Lx=Lx,Ly=Ly,bcx=1,bcy=1)
    subA=[np.arange(params.Lx//4),np.arange(params.Ly//4)]
    subAp=[np.arange(params.Lx//4)+params.Lx//4,np.arange(params.Ly//4)]
    subB=[np.arange(params.Lx//4)+params.Lx//2,np.arange(params.Ly//4)]
        
    
    if Born==1:
        params.measure_all_Born(subAp,type='link',pool=2)
    elif Born==0:
        params.measure_all_Born(subAp,type='link',prob=[1,0,0,0])
    elif Born==-1:
        params.measure_all_Born(subAp,type='link',prob=[0,1,0,0])
        
    LN=params.log_neg(subA,subB)
    MI=params.mutual_information_m(subA,subB)
    return MI,LN,m_i,es_i

if __name__=="__main__":
    # if rank==0:
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    parser.add_argument('--min',default=1,type=float)
    parser.add_argument('--max',default=3,type=float)
    parser.add_argument('--num',default=21,type=int)
    parser.add_argument('--Lx',default=32,type=int)
    parser.add_argument('--Ly',default=4,type=int)
    parser.add_argument('--Born',default=1,type=int) # 1: Born measurement; 0: Force measurement to |10>+|01>; 0: Force measurement to |10>-|01>; other: no measurement
    parser.add_argument('--timing',default=True,type=bool)
    args=parser.parse_args()
    if args.timing:
        st=time.time()

    if args.Born==1:
        assert args.es>1, 'Born measurement but with single ensemble size {:d}'.format(args.es)
    else:
        assert args.es==1, 'Not need to have ensemble size of {:d}'.format(args.es)

    m_list=np.linspace(args.min,args.max,args.num)
    MI_Born_list=np.zeros((args.num,args.es))
    LN_Born_list=np.zeros((args.num,args.es))
    st=time.time()
    executor=MPIPoolExecutor()
    ensemble_list_pool=[]
    inputs=[(m,args.Lx,args.Ly,args.Born,m_i,es_i) for m_i,m in enumerate(m_list) for es_i in range(args.es)]
    pool=executor.map(run,inputs)
    for result in pool:
        MI,LN,m_i,es_i=result
        MI_Born_list[m_i,es_i]=MI
        LN_Born_list[m_i,es_i]=LN

    executor.shutdown()
    if args.Born==1:
        Born_str='Born'
    elif args.Born==0:
        Born_str='Force+'
    elif args.Born==-1:
        Born_str='Force-'
    else:
        Born_str=''
    with open('HOTI_Born_En{:d}_Lx{:d}_Ly{:d}_{:s}.pickle'.format(args.es,args.Lx,args.Ly,Born_str),'wb') as f:
        pickle.dump([m_list,MI_Born_list,LN_Born_list],f)

    if args.timing:
            print('Elapsed:{:.1f}'.format(time.time()-st))