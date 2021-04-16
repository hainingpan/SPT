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

class Params:
    '''
    example: params=Params(mu=2)
    '''
    def __init__(self,
    delta=0,    
    L=100,
    T=0,
    bc=1    # 0: open boundary condition; >0: PBC; <0: APBC
    ):
        self.delta=delta
        self.mu=2*(1-delta)
        self.t=-(1+delta)
        self.Delta=-(1+delta)
        self.L=L
        self.tau_z=sp.dia_matrix(np.diag([1,-1]))
        self.tau_y=sp.dia_matrix(np.array([[0,-1j],[1j,0]]))
        self.bc=bc
        self.T=T
        self.band1sm=sp.diags([1],[1],(L,L)).tocsr()
        self.bandm1sm=sp.diags([1],[-1],(L,L)).tocsr()
        self.band1sm[-1,0]=1*(2*np.heaviside(bc,1/2)-1)
        self.bandm1sm[0,-1]=1*(2*np.heaviside(bc,1/2)-1)
        # Hamiltonian in the ferimion basis
        self.Hamiltonian_f=-self.mu*sp.kron(self.tau_z,sp.identity(self.L))-sp.kron(self.t*self.tau_z+1j*self.Delta*self.tau_y,self.band1sm)-sp.kron(self.t*self.tau_z-1j*self.Delta*self.tau_y,self.bandm1sm)
        # BdG Hamiltonian back to original
        
        self.Hamiltonian_f/=2
        # Hamiltonian in the Majorana basis
        band=np.vstack([np.ones(L)*(1-delta)*1j,np.ones(L)*(1+delta)*1j]).flatten('F')
        Ham=sp.diags(np.array([band[:-1],band[:-1].conj()]),np.array([-1,1]),shape=(2*L,2*L)).tocsr()
        Ham[0,-1]=(1+delta)*1j*bc
        Ham[-1,0]=-(1+delta)*1j*bc
        self.Hamiltonian_m=Ham

        


    def bandstructure(self,H_type='f'):    
        if H_type=='f':    
            val,vec=la.eigh(self.Hamiltonian_f.toarray())
            sortindex=np.argsort(val)
            self.val_f=val[sortindex]
            self.vec_f=vec[:,sortindex]
        elif H_type=='m':
            val,vec=la.eigh(self.Hamiltonian_m.toarray()) 
            sortindex=np.argsort(val)
            self.val_m=val[sortindex]
            self.vec_m=vec[:,sortindex]
        else:
            raise ValueError('type of Hamiltonian ({}) not found'.format(H_type))



    def fermi_dist(self,energy,E_F):      
        if self.T==0:
            return np.heaviside(E_F-energy,0)
        else:
            return 1/(1+np.exp((energy-E_F)/self.T)) 


    def covariance_matrix_f(self,E_F=0):
        if not (hasattr(self,'val_f') and hasattr(self,'vec_f')):
            self.bandstructure('f')
        occupancy=self.fermi_dist(self.val_f,E_F)
        occupancy_mat=np.matlib.repmat(occupancy,self.vec_f.shape[0],1)
        self.C_f=np.real((occupancy_mat*self.vec_f)@self.vec_f.T.conj())

    def covariance_matrix_m(self,E_F=0):
        if not (hasattr(self,'val_m') and hasattr(self,'vec_m')):
            self.bandstructure('m')
        occupancy=self.fermi_dist(self.val_m,E_F)
        occupancy_mat=np.matlib.repmat(occupancy,self.vec_m.shape[0],1)
        self.C_m=(1j*2*(occupancy_mat*self.vec_m)@self.vec_m.T.conj())-1j*np.eye(self.L*2)
        assert np.abs(np.imag(self.C_m)).max()<1e-10, "Covariance matrix not real"
        self.C_m=np.real(self.C_m)
        self.C_m_history=[self.C_m]
    
    def projection_obs(self,s):
        '''
        s= 0,1 occupancy number
        i,j: adjacent pair of Majorana
        flow is from alpha_{i,j} to gamma_{i,j}

        return: the basis are ordered as gamma_i,gamma_j,alpha_j,alpha_i
        '''
        assert (s==0 or s==1),"s={} is either 0 or 1".format(s)
        blkmat=(np.array([[0,-(-1)**s],[(-1)**s,0]]))
        return sp.bmat([[blkmat,None],[None,blkmat.T]]).toarray()

    def projection(self,s):
        '''
        s= 0,1 occupancy number
        i,j: adjacent pair of Majorana
        flow is from alpha_{i,j} to gamma_{i,j}

        return: the basis are ordered as gamma_i,gamma_j,alpha_j,alpha_i
        '''
        assert (s==0 or s==1),"s={} is either 0 or 1".format(s)
        # blkmat=(np.array([[0,-(-1)**s],[(-1)**s,0]]))
        # zero=np.zeros((2,2))
        blkmat=np.array([[0,-(-1)**s,0,0],
                        [(-1)**s,0,0,0],
                        [0,0,0,(-1)**s],
                        [0,0,-(-1)**s,0]])
        # return np.block([[blkmat,zero],[zero,blkmat.T]])
        return blkmat

    def measure_obs(self,s,i,j):
        permutation_mat=sp.diags([1],[0],(self.L*2,self.L*2)).tocsr()
        # i <-> -2
        permutation_mat[i,i]=0
        permutation_mat[-2,-2]=0
        permutation_mat[i,-2]=1
        permutation_mat[-2,i]=1
        # j <-> -1
        permutation_mat[j,j]=0
        permutation_mat[-1,-1]=0
        permutation_mat[j,-1]=1
        permutation_mat[-1,j]=1
        if not hasattr(self,'C_m'):
            self.covariance_matrix_m()

        # m=np.arange(64).reshape((8,8))
        # C_m_perm=permutation_mat.T@m@permutation_mat.T

        C_m_perm=permutation_mat.T@self.C_m_history[-1]@permutation_mat.T

        self.m=C_m_perm
        Gamma_LL=C_m_perm[:-2,:-2]
        Gamma_LR=C_m_perm[:-2,-2:]
        Gamma_RR=C_m_perm[-2:,-2:]

        proj=self.projection(s)
        Upsilon_LL=proj[:-2,:-2]
        Upsilon_LR=proj[:-2,-2:]
        Upsilon_RR=proj[-2:,-2:]
        Upsilon_RL=proj[-2:,:-2]
        zero=np.zeros((self.L*2-2,2))
        zero0=np.zeros((2,2))
        mat1=np.block([[Gamma_LL,zero],[zero.T,Upsilon_RR]])
        mat2=np.block([[Gamma_LR,zero],[zero0,Upsilon_RL]])
        mat3=np.block([[Gamma_RR,np.eye(2)],[-np.eye(2),Upsilon_LL]])
        self.mat2=mat2
        if np.count_nonzero(mat2):
            Psi=mat1+mat2@(la.solve(mat3,mat2.T))
        else:
            Psi=mat1
        Psi=permutation_mat.T@Psi@permutation_mat
        
        
        self.C_m_history.append(Psi)

    def measure(self,s,i,j):
        if not hasattr(self,'C_m'):
            self.covariance_matrix_m()
        
        # m=np.arange(64).reshape((8,8))
        
        m=self.C_m_history[-1].copy()
        m[i:,:]=np.roll(m[i:,:],-1,0)
        m[:,i:]=np.roll(m[:,i:],-1,1)

        if j>i:
            j-=1    #the position of j is rotated by 1 ahead
        m[j:,:]=np.roll(m[j:,:],-1,0)
        m[:,j:]=np.roll(m[:,j:],-1,1)
        self.m=m

        Gamma_LL=m[:-2,:-2]
        Gamma_LR=m[:-2,-2:]
        Gamma_RR=m[-2:,-2:]       

        proj=self.projection(s)
        Upsilon_LL=proj[:-2,:-2]
        Upsilon_LR=proj[:-2,-2:]
        Upsilon_RR=proj[-2:,-2:]
        Upsilon_RL=proj[-2:,:-2]
        zero=np.zeros((self.L*2-2,2))
        zero0=np.zeros((2,2))
        mat1=np.block([[Gamma_LL,zero],[zero.T,Upsilon_RR]])
        mat2=np.block([[Gamma_LR,zero],[zero0,Upsilon_RL]])
        mat3=np.block([[Gamma_RR,np.eye(2)],[-np.eye(2),Upsilon_LL]])
        self.mat2=mat2
        if np.count_nonzero(mat2):
            Psi=mat1+mat2@(la.solve(mat3,mat2.T))
        else:
            Psi=mat1
        
        Psi[j:,:]=np.roll(Psi[j:,:],1,0)
        Psi[:,j:]=np.roll(Psi[:,j:],1,1)

        Psi[i:,:]=np.roll(Psi[i:,:],1,0)
        Psi[:,i:]=np.roll(Psi[:,i:],1,1)
        # Psi=permutation_mat.T@Psi@permutation_mat        
        
        self.C_m_history.append(Psi) 


    def c_subregion_f_obs(self,subregion):
        if not hasattr(self,'C'):
            self.covariance_matrix_f()
        try:
            subregion=np.array(subregion)
        except:
            raise ValueError("The subregion is ill-defined"+subregion)
        XX,YY=np.meshgrid(np.arange(2*self.L),np.arange(2*self.L))
        mask_hh=np.isin(XX,subregion)*np.isin(YY,subregion)
        mask_hp=np.isin(XX,subregion)*np.isin(YY,subregion+self.L)
        mask_ph=np.isin(XX,subregion+self.L)*np.isin(YY,subregion)
        mask_pp=np.isin(XX,subregion+self.L)*np.isin(YY,subregion+self.L)
        mask=mask_hh+mask_hp+mask_ph+mask_pp
        return self.C_f[mask].reshape((2*subregion.shape[0],2*subregion.shape[0]))

    def c_subregion_f(self,subregion):
        if not hasattr(self,'C'):
            self.covariance_matrix_f()
        try:
            subregion=np.array(subregion)
        except:
            raise ValueError("The subregion is ill-defined"+subregion)
        subregion_ph=np.concatenate([subregion,subregion+self.L])
        return self.C_f[np.ix_(subregion_ph,subregion_ph)]

    def von_Neumann_entropy(self,subregion):
        c_A=self.c_subregion(subregion)
        val,vec=la.eigh(c_A)
        self.val_sh=val
        val=np.sort(val)[:subregion.shape[0]]
        return np.real(-np.sum(val*np.log(val+1e-18j))-np.sum((1-val)*np.log(1-val+1e-18j)))

    def c_subregion_m_obs(self,subregion,Gamma=None):
        if not hasattr(self,'C_m'):
            self.covariance_matrix_m()

        if Gamma==None:
            Gamma=self.C_m_history[-1]
        try:
            subregion=np.array(subregion)
        except:
            raise ValueError("The subregion is ill-defined"+subregion)
        XX,YY=np.meshgrid(np.arange(2*self.L),np.arange(2*self.L))
        mask=np.isin(XX,subregion)*np.isin(YY,subregion)  
        return Gamma[mask].reshape((subregion.shape[0],subregion.shape[0]))

    def c_subregion_m(self,subregion,Gamma=None):
        if not hasattr(self,'C_m'):
            self.covariance_matrix_m()
        if Gamma==None:
            Gamma=self.C_m_history[-1]
        try:
            subregion=np.array(subregion)
        except:
            raise ValueError("The subregion is ill-defined"+subregion)
        return Gamma[np.ix_(subregion,subregion)]
        


    def von_Neumann_entropy_m(self,subregion):
        # c_A=self.c_subregion_m(subregion)
        c_A=self.c_subregion_m_obs(subregion)
        val,vec=la.eigh(1j*c_A)
        self.val_sh=val
        val=np.sort(val)
        val=(1-val)/2   #\lambda=(1-\xi)/2
        return np.real(-np.sum(val*np.log(val+1e-18j))-np.sum((1-val)*np.log(1-val+1e-18j)))/2

    def mutual_information(self,subregion_A,subregion_B):
        s_A=self.von_Neumann_entropy(subregion_A)
        s_B=self.von_Neumann_entropy(subregion_B)
        assert np.intersect1d(subregion_A,subregion_B).size==0 , "Subregion A and B overlap"
        subregion_AB=np.concatenate([subregion_A,subregion_B])
        s_AB=self.von_Neumann_entropy(subregion_AB)
        return s_A+s_B-s_AB

    def mutual_information_m(self,subregion_A,subregion_B):
        s_A=self.von_Neumann_entropy_m(subregion_A)
        s_B=self.von_Neumann_entropy_m(subregion_B)
        assert np.intersect1d(subregion_A,subregion_B).size==0 , "Subregion A and B overlap"
        subregion_AB=np.concatenate([subregion_A,subregion_B])
        s_AB=self.von_Neumann_entropy_m(subregion_AB)
        return s_A+s_B-s_AB

    def measure_batch(self,batchsize,proj_range):
        self.i_history=[]
        self.s_history=[]
        for _ in range(batchsize):
            i=np.random.randint(*proj_range)
            s=np.random.randint(0,2)
            self.i_history.append(i)
            self.s_history.append(s)
            self.measure(s,i,i+1)

    def measure_all(self,s):
        self.i_history=[]
        self.s_history=[]
        # proj_range=np.hstack([np.arange(int(self.L/2),self.L,2),np.arange(int(self.L/2),self.L,2)+self.L])
        proj_range=np.hstack([np.arange(int(self.L/2),self.L,2)]) 
        # proj_range=np.hstack([np.arange(int(self.L/2),self.L)])
        # proj_range=np.hstack([np.arange(int(self.L/2),self.L),np.arange(int(self.L/2),int(self.L/2)+2)+self.L])
        for i in proj_range:
            self.i_history.append(i)
            self.s_history.append(s)
            self.measure(s,i,i+1)

    def measure_all_random(self,batchsize,proj_range):
        self.i_history=[]
        self.s_history=[]        
        # if batchsize>proj_range.shape[0]:
        #     raise ValueError("The batchsize {} cannot be larger than the proj_range {}".format(batchsize,proj_range.shape[0]))
        choice=np.random.choice(range(*proj_range),batchsize,replace=False)
        for i in choice:
            s=np.random.randint(0,2)
            self.i_history.append(i)  
            self.s_history.append(s)
            self.measure(s,i,i+1)      


    def measure_all_random_even(self,batchsize,proj_range):
        '''
        proj_range: (start,end) tuple
        '''
        self.i_history=[]
        self.s_history=[]        
        proj_range_even=[i//2 for i in proj_range]
        choice=np.random.choice(range(*proj_range_even),batchsize,replace=False)
        for i in choice:
            s=np.random.randint(0,2)
            self.i_history.append(2*i)  #only even is accepted 
            self.s_history.append(s)
            self.measure(s,2*i,2*i+1)  

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


if __name__=="__main__":   
    parser=argparse.ArgumentParser()
    parser.add_argument('--es',default=100,type=int)
    args=parser.parse_args()


    delta_dict={}
    mutual_info_dis_dict={}
    
    for i in (0,12,13,14,15,16):
        print(i)
        st=time.time()
        delta_dict[i],mutual_info_dis_dict[i]=mutual_info_run(i,args.es)
        print(time.time()-st)

    with open('mutual_info_Ap_En{:d}.pickle'.format(args.es),'wb') as f:
        pickle.dump([delta_dict,mutual_info_dis_dict],f)
    
    fig,ax=plt.subplots()
    for i in (0,12,14,15,16):
        ax.plot(delta_dict[i],np.array(mutual_info_dis_dict[i]).mean(axis=1)/np.log(2),label='Number of gates: {}'.format(i))

    ax.legend()
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'Mutual information between A and B [$\log2$]')

    fig.savefig('mutual_info_Ap_En{:d}.pdf'.format(args.es),bbox_inches='tight')
