from Chern_insulator import *
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
from copy import copy

def run(inputs):
    Lx,Ly,m,Born,geo,m_i,es_i=inputs
    # Lx,Ly,m,bcx,bcy=inputs
    bcx=1
    bcy=1
    params=Params(m=m,Lx=Lx,Ly=Ly,bcx=bcx,bcy=bcy)

    if geo==1:
        proj_total_A=params.linearize_index([np.arange(params.Lx//8*3,params.Lx//8*5),np.arange(params.Ly//8*3,params.Ly//8*5)],4,proj=True,k=4)
        proj_total_A_Ap=params.linearize_index([np.arange(params.Lx//4,params.Lx//4*3),np.arange(params.Ly//4,params.Ly//4*3)],4,proj=True,k=4)
        proj_total_Ap=np.setdiff1d(proj_total_A_Ap,proj_total_A)

        sub_A=params.linearize_index([np.arange(params.Lx//8*3,params.Lx//8*5),np.arange(params.Ly//8*3,params.Ly//8*5)],4)
        sub_A_Ap=params.linearize_index([np.arange(params.Lx//4,params.Lx//4*3),np.arange(params.Ly//4,params.Ly//4*3)],4)
        sub_A_Ap_B=params.linearize_index([np.arange(params.Lx//8,params.Lx//8*7),np.arange(params.Ly//8,params.Ly//8*7)],4)
        sub_B=np.setdiff1d(sub_A_Ap_B,sub_A_Ap)
    elif geo==2:
        proj_total_A=params.linearize_index([np.arange(params.Lx//8,params.Lx//8*3),np.arange(params.Ly//4,params.Ly//2)],4,proj=True,k=4)
        proj_total_B=params.linearize_index([np.arange(params.Lx//8*5,params.Lx//8*7),np.arange(params.Ly//4,params.Ly//2)],4,proj=True,k=4)
        proj_total_Ap_A_B=params.linearize_index([np.arange(0,params.Lx),np.arange(0,params.Ly//4*3)],4,proj=True,k=4)
        proj_total_Ap=np.setdiff1d(proj_total_Ap_A_B,np.hstack((proj_total_A,proj_total_B)))

        sub_A=params.linearize_index([np.arange(params.Lx//8,params.Lx//8*3),np.arange(params.Ly//4,params.Ly//2)],4)
        sub_B=params.linearize_index([np.arange(params.Lx//8*5,params.Lx//8*7),np.arange(params.Ly//4,params.Ly//2)],4)
        
    
    if Born==1:
        params.measure_all_Born(proj_total_Ap,linear=True,type='correlated')
    elif Born==0:
        params.measure_all_Born(proj_total_Ap,linear=True,type='correlated',prob=[1,0])
        
    LN=params.log_neg(sub_A,sub_B,linear=True)
    MI=params.mutual_information_m(sub_A,sub_B,linear=True)
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
    parser.add_argument('--Born',default=1,type=int) # 1: Born measurement; 0: Force measurement; other: no measurement
    parser.add_argument('--geo',default=1,type=int) # 1: A inside A' inside B; 2: A//B inside A'
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
    inputs=[(args.Lx,args.Ly,m,args.Born,args.geo,m_i,es_i) for m_i,m in enumerate(m_list) for es_i in range(args.es)]
    pool=executor.map(run,inputs)
    for result in pool:
        MI,LN,m_i,es_i=result
        MI_Born_list[m_i,es_i]=MI
        LN_Born_list[m_i,es_i]=LN

    executor.shutdown()
    if args.Born==1:
        Born_str='Born'
    elif args.Born==0:
        Born_str='Force'
    else:
        Born_str=''
    with open('CI_Born_En{:d}_Lx{:d}_Ly{:d}_geo{:d}_{:s}.pickle'.format(args.es,args.Lx,args.Ly,args.geo,Born_str),'wb') as f:
        pickle.dump([m_list,MI_Born_list,LN_Born_list],f)

    if args.timing:
            print('Elapsed:{:.1f}'.format(time.time()-st))