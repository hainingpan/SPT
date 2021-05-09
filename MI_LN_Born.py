from Majorana_chain import *
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size= comm.Get_size()

def run(es,pt,L,Bp):
    eta_Born_list=[]
    MI_Born_list=[]
    LN_Born_list=[]
    for _ in range(pt):
        MI_ensemble_list=[]
        LN_ensemble_list=[]
        x=sorted(2*np.random.choice(np.arange(1,L),3,replace=False))
        x=[0]+x
        subregionA=np.arange(x[0],x[1])
        subregionB=np.arange(x[2],x[3])
        subregionAp=np.arange(x[1],x[2],2)
        if Bp:
            subregionBp=np.concatenate([np.arange(x[3],2*L,2),np.arange(0,x[0],2)])
        eta=cross_ratio(x,L)
        for _ in range(es):
            if Bp:
                params=Params(delta=0,L=L,bc=-1,basis='m').measure_all_Born(subregionAp).measure_all_Born(subregionBp)
            else:
                params=Params(delta=0,L=L,bc=-1,basis='m').measure_all_Born(subregionAp)
            MI_ensemble_list.append(params.mutual_information_m(subregionA,subregionB))
            LN_ensemble_list.append(params.log_neg(subregionA,subregionB))
        eta_Born_list.append(eta)
        MI_Born_list.append(MI_ensemble_list)
        LN_Born_list.append(LN_ensemble_list)
    return np.array(eta_Born_list),np.array(MI_Born_list),np.array(LN_Born_list)
        

        

if __name__=="__main__":
    # if rank==0:
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    parser.add_argument('--pt',default=100,type=int)
    parser.add_argument('--L',default=128,type=int)
    parser.add_argument('--Bp',default=False,type=bool)
    parser.add_argument('--timing',default=False,type=bool)
    args=parser.parse_args()
    assert args.pt%size==0, "point size {} mod {} is not zero".format(args.pt,size)
    per=args.pt//size
    if rank==0 and args.timing:
        st=time.time()
        
    eta_Born_list,MI_Born_list,LN_Born_list=run(args.es,per,args.L,args.Bp)
    if rank==0:
        eta_recv=np.empty(args.pt)
        MI_recv=np.empty((args.pt,args.es))
        LN_recv=np.empty((args.pt,args.es))
    else:
        eta_recv=None
        MI_recv=None
        LN_recv=None
    # print("rank{}:{}".format(rank,eta_Born_list))
    # print(eta_Born_list)
    comm.Gather(eta_Born_list,eta_recv,root=0)
    comm.Gather(MI_Born_list,MI_recv,root=0)
    comm.Gather(LN_Born_list,LN_recv,root=0)

    if rank==0:
        with open('MI_LN_Born_es{}_pt{}_Ap{}.pickle'.format(args.es,args.pt,args.Bp*'Bp'),'wb') as f:
            pickle.dump([eta_recv,MI_recv,LN_recv],f)
        if args.timing:
            print('Elapsed:{:.1f}'.format(time.time()-st))

