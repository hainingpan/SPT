import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la
import numpy.matlib
import itertools

class Params:
    '''
    example: params=Params(mu=2)
    '''
    def __init__(self,
    delta=0,    
    L=100,
    T=0,
    bc=1,    # 0: open boundary condition; >0: PBC; <0: APBC
    basis='mf'    # 'generate Hamiltonian of fermionic ('f') and Majorana basis ('m') or both ('mf')
    ):
        self.delta=delta
        self.mu=2*(1-delta)
        self.t=-(1+delta)
        self.Delta=-(1+delta)
        self.L=L
        self.tau_z=np.array([[1,0],[0,-1]])
        self.tau_y=np.array([[0,-1j],[1j,0]])
        self.bc=bc
        self.T=T
        if 'f' in basis:
            band1sm=np.diag(np.ones(L-1),1)
            band1sm[-1,0]=1*(2*np.heaviside(bc,1/2)-1)
            id_mat=np.eye(L)
            # Hamiltonian in the ferimion basis
            self.Hamiltonian_f=np.block([[-self.mu*id_mat-self.t*(band1sm+band1sm.T),-self.Delta*(band1sm-band1sm.T)],
                                        [self.Delta*(band1sm-band1sm.T),self.mu*id_mat+self.t*(band1sm+band1sm.T)]])
            # BdG Hamiltonian back to original        
            self.Hamiltonian_f/=2

        if 'm' in basis:    
            # Hamiltonian in the Majorana basis
            band=np.vstack([np.ones(L)*(1-delta)*1j,np.ones(L)*(1+delta)*1j]).flatten('F')
            Ham=np.diag(band[:-1],-1)
            Ham[0,-1]=(1+delta)*1j*bc
            Ham=Ham+Ham.conj().T
            self.Hamiltonian_m=Ham

    # def __init_obs__(self,
    # delta=0,    
    # L=100,
    # T=0,
    # bc=1    # 0: open boundary condition; >0: PBC; <0: APBC
    # ):
    #     self.delta=delta
    #     self.mu=2*(1-delta)
    #     self.t=-(1+delta)
    #     self.Delta=-(1+delta)
    #     self.L=L
    #     self.tau_z=sp.dia_matrix(np.diag([1,-1]))
    #     self.tau_y=sp.dia_matrix(np.array([[0,-1j],[1j,0]]))
    #     self.bc=bc
    #     self.T=T
    #     self.band1sm=sp.diags([1],[1],(L,L)).tocsr()
    #     self.bandm1sm=sp.diags([1],[-1],(L,L)).tocsr()
    #     self.band1sm[-1,0]=1*(2*np.heaviside(bc,1/2)-1)
    #     self.bandm1sm[0,-1]=1*(2*np.heaviside(bc,1/2)-1)
    #     # Hamiltonian in the ferimion basis
    #     self.Hamiltonian_f=-self.mu*sp.kron(self.tau_z,sp.identity(self.L))-sp.kron(self.t*self.tau_z+1j*self.Delta*self.tau_y,self.band1sm)-sp.kron(self.t*self.tau_z-1j*self.Delta*self.tau_y,self.bandm1sm)
    #     # BdG Hamiltonian back to original
        
    #     self.Hamiltonian_f/=2
    #     # Hamiltonian in the Majorana basis
    #     band=np.vstack([np.ones(L)*(1-delta)*1j,np.ones(L)*(1+delta)*1j]).flatten('F')
    #     Ham=sp.diags(np.array([band[:-1],band[:-1].conj()]),np.array([-1,1]),shape=(2*L,2*L)).tocsr()
    #     Ham[0,-1]=(1+delta)*1j*bc
    #     Ham[-1,0]=-(1+delta)*1j*bc
    #     self.Hamiltonian_m=Ham

    # def bandstructure_obs(self,H_type='f'):    
    #     if H_type=='f':    
    #         val,vec=la.eigh(self.Hamiltonian_f)
    #         sortindex=np.argsort(val)
    #         self.val_f=val[sortindex]
    #         self.vec_f=vec[:,sortindex]
    #     elif H_type=='m':
    #         val,vec=la.eigh(self.Hamiltonian_m) 
    #         sortindex=np.argsort(val)
    #         self.val_m=val[sortindex]
    #         self.vec_m=vec[:,sortindex]
    #     else:
    #         raise ValueError('type of Hamiltonian ({}) not found'.format(H_type))

    def bandstructure(self,basis='mf'):
        if 'f' in basis:    
            val,vec=la.eigh(self.Hamiltonian_f)
            sortindex=np.argsort(val)
            self.val_f=val[sortindex]
            self.vec_f=vec[:,sortindex]
        if 'm' in basis:
            val,vec=np.linalg.eigh(self.Hamiltonian_m) 
            sortindex=np.argsort(val)
            self.val_m=val[sortindex]
            self.vec_m=vec[:,sortindex]       

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
    
    # def projection_obs(self,s):
    #     '''
    #     s= 0,1 occupancy number
    #     '''
    #     assert (s==0 or s==1),"s={} is either 0 or 1".format(s)
    #     blkmat=(np.array([[0,-(-1)**s],[(-1)**s,0]]))
    #     return sp.bmat([[blkmat,None],[None,blkmat.T]]).toarray()

    def projection(self,s):
        '''
        occupancy number: s= 0,1 
        (-1)^0 even parity, (-1)^1 odd parity

        '''
        assert (s==0 or s==1),"s={} is either 0 or 1".format(s)
        blkmat=np.array([[0,-(-1)**s,0,0],
                        [(-1)**s,0,0,0],
                        [0,0,0,(-1)**s],
                        [0,0,-(-1)**s,0]])
        return blkmat

    # Slower than measure() 8.3x times 
    # def measure_obs(self,s,i,j):
    #     permutation_mat=sp.diags([1],[0],(self.L*2,self.L*2)).tocsr()
    #     # i <-> -2
    #     permutation_mat[i,i]=0
    #     permutation_mat[-2,-2]=0
    #     permutation_mat[i,-2]=1
    #     permutation_mat[-2,i]=1
    #     # j <-> -1
    #     permutation_mat[j,j]=0
    #     permutation_mat[-1,-1]=0
    #     permutation_mat[j,-1]=1
    #     permutation_mat[-1,j]=1
    #     if not hasattr(self,'C_m'):
    #         self.covariance_matrix_m()

    #     # m=np.arange(64).reshape((8,8))
    #     # C_m_perm=permutation_mat.T@m@permutation_mat.T

    #     C_m_perm=permutation_mat.T@self.C_m_history[-1]@permutation_mat.T

    #     self.m=C_m_perm
    #     Gamma_LL=C_m_perm[:-2,:-2]
    #     Gamma_LR=C_m_perm[:-2,-2:]
    #     Gamma_RR=C_m_perm[-2:,-2:]

    #     proj=self.projection(s)
    #     Upsilon_LL=proj[:-2,:-2]
    #     Upsilon_LR=proj[:-2,-2:]
    #     Upsilon_RR=proj[-2:,-2:]
    #     Upsilon_RL=proj[-2:,:-2]
    #     zero=np.zeros((self.L*2-2,2))
    #     zero0=np.zeros((2,2))
    #     mat1=np.block([[Gamma_LL,zero],[zero.T,Upsilon_RR]])
    #     mat2=np.block([[Gamma_LR,zero],[zero0,Upsilon_RL]])
    #     mat3=np.block([[Gamma_RR,np.eye(2)],[-np.eye(2),Upsilon_LL]])
    #     self.mat2=mat2
    #     if np.count_nonzero(mat2):
    #         Psi=mat1+mat2@(la.solve(mat3,mat2.T))
    #     else:
    #         Psi=mat1
    #     Psi=permutation_mat.T@Psi@permutation_mat
        
        
    #     self.C_m_history.append(Psi)

    # Slower than measure() 1.7x times 
    # def measure_roll(self,s,i,j):
    #     if not hasattr(self,'C_m'):
    #         self.covariance_matrix_m()
        
    #     # m=np.arange(64).reshape((8,8))
        
    #     m=self.C_m_history[-1].copy()
    #     m[i:,:]=np.roll(m[i:,:],-1,0)
    #     m[:,i:]=np.roll(m[:,i:],-1,1)

    #     if j>i:
    #         j-=1    #the position of j is rotated by 1 ahead
    #     m[j:,:]=np.roll(m[j:,:],-1,0)
    #     m[:,j:]=np.roll(m[:,j:],-1,1)
    #     self.m=m

    #     Gamma_LL=m[:-2,:-2]
    #     Gamma_LR=m[:-2,-2:]
    #     Gamma_RR=m[-2:,-2:]       

    #     proj=self.projection(s)
    #     Upsilon_LL=proj[:-2,:-2]
    #     Upsilon_LR=proj[:-2,-2:]
    #     Upsilon_RR=proj[-2:,-2:]
    #     Upsilon_RL=proj[-2:,:-2]
    #     zero=np.zeros((self.L*2-2,2))
    #     zero0=np.zeros((2,2))
    #     mat1=np.block([[Gamma_LL,zero],[zero.T,Upsilon_RR]])
    #     mat2=np.block([[Gamma_LR,zero],[zero0,Upsilon_RL]])
    #     mat3=np.block([[Gamma_RR,np.eye(2)],[-np.eye(2),Upsilon_LL]])
    #     self.mat2=mat2
    #     if np.count_nonzero(mat2):
    #         Psi=mat1+mat2@(la.solve(mat3,mat2.T))
    #     else:
    #         Psi=mat1
        
    #     Psi[j:,:]=np.roll(Psi[j:,:],1,0)
    #     Psi[:,j:]=np.roll(Psi[:,j:],1,1)

    #     Psi[i:,:]=np.roll(Psi[i:,:],1,0)
    #     Psi[:,i:]=np.roll(Psi[:,i:],1,1)
    #     # Psi=permutation_mat.T@Psi@permutation_mat        
        
    #     self.C_m_history.append(Psi) 

    def measure(self,s,i,j):
        if not hasattr(self,'C_m'):
            self.covariance_matrix_m()
        if not hasattr(self,'s_history'):
            self.s_history=[]
        if not hasattr(self,'i_history'):
            self.i_history=[]
        
        # m=np.arange(64).reshape((8,8))
        
        m=self.C_m_history[-1].copy()
        # i<-> -2
        m[[i,-2]]=m[[-2,i]]
        m[:,[i,-2]]=m[:,[-2,i]]
        # j<->-1
        m[[j,-1]]=m[[-1,j]]
        m[:,[j,-1]]=m[:,[-1,j]]

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
            # print(i)
            # print(np.abs(np.round(Gamma_LR,2)).max())
            # print(np.round(mat3,3))
            Psi=mat1+mat2@(la.solve(mat3,mat2.T))
            # Psi=mat1+mat2@(la.lstsq(mat3,mat2.T)[0])

            assert np.abs(np.trace(Psi))<1e-5, "Not trace zero {:e}".format(np.trace(Psi))
        else:
            Psi=mat1
        
        Psi[[j,-1]]=Psi[[-1,j]]
        Psi[:,[j,-1]]=Psi[:,[-1,j]]

        Psi[[i,-2]]=Psi[[-2,i]]
        Psi[:,[i,-2]]=Psi[:,[-2,i]]
        
        Psi=(Psi-Psi.T)/2   # Anti-symmetrize
        self.C_m_history.append(Psi)
        self.s_history.append(s)
        self.i_history.append(i)


    # def c_subregion_f_obs(self,subregion):
    #     if not hasattr(self,'C'):
    #         self.covariance_matrix_f()
    #     try:
    #         subregion=np.array(subregion)
    #     except:
    #         raise ValueError("The subregion is ill-defined"+subregion)
    #     XX,YY=np.meshgrid(np.arange(2*self.L),np.arange(2*self.L))
    #     mask_hh=np.isin(XX,subregion)*np.isin(YY,subregion)
    #     mask_hp=np.isin(XX,subregion)*np.isin(YY,subregion+self.L)
    #     mask_ph=np.isin(XX,subregion+self.L)*np.isin(YY,subregion)
    #     mask_pp=np.isin(XX,subregion+self.L)*np.isin(YY,subregion+self.L)
    #     mask=mask_hh+mask_hp+mask_ph+mask_pp
    #     return self.C_f[mask].reshape((2*subregion.shape[0],2*subregion.shape[0]))

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

    # def c_subregion_m_obs(self,subregion,Gamma=None):
    #     if not hasattr(self,'C_m'):
    #         self.covariance_matrix_m()

    #     if Gamma is None:
    #         Gamma=self.C_m_history[-1]
    #     try:
    #         subregion=np.array(subregion)
    #     except:
    #         raise ValueError("The subregion is ill-defined"+subregion)
    #     XX,YY=np.meshgrid(np.arange(2*self.L),np.arange(2*self.L))
    #     mask=np.isin(XX,subregion)*np.isin(YY,subregion)  
    #     return Gamma[mask].reshape((subregion.shape[0],subregion.shape[0]))

    def c_subregion_m(self,subregion,Gamma=None):
        if not hasattr(self,'C_m'):
            self.covariance_matrix_m()
        if Gamma is None:
            Gamma=self.C_m_history[-1]
        try:
            subregion=np.array(subregion)
        except:
            raise ValueError("The subregion is ill-defined"+subregion)
        return Gamma[np.ix_(subregion,subregion)]
        


    def von_Neumann_entropy_m(self,subregion):
        c_A=self.c_subregion_m(subregion)
        # c_A=self.c_subregion_m_obs(subregion)
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
        # self.i_history=[]
        # self.s_history=[]
        for _ in range(batchsize):
            i=np.random.randint(*proj_range)
            s=np.random.randint(0,2)
            # self.i_history.append(i)
            # self.s_history.append(s)
            self.measure(s,i,i+1)

    def measure_all(self,s_prob,proj_range=None):
        '''
        The probability of s=0 (unoccupied)

        '''
        # self.i_history=[]
        # self.s_history=[]
        if proj_range is None:
            proj_range=np.arange(int(self.L/2),self.L,2)
        for i in proj_range:
            # self.i_history.append(i)
            if s_prob==0:
                s=1
            elif s_prob==1:
                s=0
            else:           
                s=s_prob<np.random.rand()
            # self.s_history.append(s)
            self.measure(s,i,i+1)

    def measure_all_position(self,s_prob):
        '''
        The random position, the prob of s=0 (unoccupied)
        '''
        # self.i_history=[]
        # self.s_history=[]
        proj_range=np.arange(int(self.L/2),self.L,2)
        if random:
            s_choice=np.random.choice(range(len(proj_range)),int(s_prob*len(proj_range)),replace=False)
            s_list=np.ones(len(proj_range),dtype=int)
            s_list[s_choice]=0
            for i,s in zip(proj_range,s_list):            
                # self.i_history.append(i)
                # self.s_history.append(s)
                self.measure(s,i,i+1)  

    def generate_position_list(self,proj_range,s_prob):
        '''
        proj_range: the list of first index of the specific projection operator 
        return: a iterator for s=0
        Generate position list, then feed into measure_list()
        '''        
        r=int(len(proj_range)*(s_prob))
        index_all=range(len(proj_range))
        index_s_0=itertools.combinations(index_all,r)
        s_list_list=[]
        for s_0 in index_s_0:
            s_list=np.ones(len(proj_range),dtype=int)
            s_list[list(s_0)]=0
            s_list_list.append(s_list)
        return s_list_list
        

    def measure_list(self,proj_range,s_list):
        '''
        proj_range: the list of first index of the specific projection operator
        s_list: 0: emtpy; 1: filled; other: no measurement
        '''
        assert len(proj_range) == len(s_list), 'Length of proj_range ({}) is not equal to the length of s_list ({})'.format(len(proj_range),len(s_list))
        for position,s in zip(proj_range,s_list):
            if s == 0 or s ==1:
                self.measure(s,position,position+1)

    def measure_all_born(self,proj_range=None,order=None):
        if proj_range is None:
            proj_range=np.arange(int(self.L/2),self.L,2)
        if order=='e2':
            proj_range=np.concatenate((proj_range[::2],proj_range[1::2]))
        if order=='e3':
            proj_range=np.concatenate((proj_range[::3],proj_range[1::3],proj_range[2::3]))
        if order=='e4':
            proj_range=np.concatenate((proj_range[::4],proj_range[1::4],proj_range[2::4]+proj_range[3::4]))

        self.covariance_matrix_m()
        for i in proj_range:
            P_0=(self.C_m_history[-1][i,i+1]+1)/2
            # print(P_0)
            if np.random.rand() < P_0:                
                self.measure(0,i,i+1)
            else:
                self.measure(1,i,i+1)

    def measure_all_random(self,batchsize,proj_range):
        # self.i_history=[]
        # self.s_history=[]        
        # if batchsize>proj_range.shape[0]:
        #     raise ValueError("The batchsize {} cannot be larger than the proj_range {}".format(batchsize,proj_range.shape[0]))
        choice=np.random.choice(range(*proj_range),batchsize,replace=False)
        for i in choice:
            s=np.random.randint(0,2)
            # self.i_history.append(i)  
            # self.s_history.append(s)
            self.measure(s,i,i+1)      


    def measure_all_random_even(self,batchsize,proj_range):
        '''
        proj_range: (start,end) tuple
        '''
        # self.i_history=[]
        # self.s_history=[]        
        proj_range_even=[i//2 for i in proj_range]
        choice=np.random.choice(range(*proj_range_even),batchsize,replace=False)
        for i in choice:
            s=np.random.randint(0,2)
            # self.i_history.append(2*i)  #only even is accepted 
            # self.s_history.append(s)
            self.measure(s,2*i,2*i+1)  

    def log_neg(self,La=None,Gamma=None):
        '''
        La: number of Majorana site in A, the corresponding Majorana site in B is 2*L-La
        '''
        
        if not hasattr(self,'C_m'):
            self.covariance_matrix_m()
        if La is None:
            La=self.L
        if Gamma is None:
            Gamma=self.C_m_history[-1]

        Gm_1= np.block([
            [-Gamma[:La,:La], -1j*Gamma[:La,La:]],
            [-1j*Gamma[La:,:La], Gamma[La:,La:]]
        ])

        Gm_2= np.block([
        [-Gamma[:La,:La], 1j*Gamma[:La,La:]],
        [1j*Gamma[La:,:La], Gamma[La:,La:]]
        ])

        Gx=np.eye(2*self.L)-np.dot(np.eye(2*self.L)+1j*Gm_2,np.dot(np.linalg.inv(np.eye(2*self.L)-np.dot(Gm_1,Gm_2)),np.eye(2*self.L)+1j*Gm_1))
        Gx=(Gx+Gx.conj().T)/2
        nu=np.linalg.eigvalsh(Gx)
        eA=np.sum(np.log(((1+nu+1j*0)/2)**0.5+((1-nu+1j*0)/2)**0.5))/2
        chi =np.linalg.eigvalsh(1j*Gamma)
        sA=np.sum(np.log(((1+chi)/2)**2+((1-chi)/2)**2))/4
        return np.real(eA+sA) 

    def CFT_correlator(self,x):
        xx=lambda i,j: (np.sin(np.pi/(2*self.L)*np.abs(x[i]-x[j])))
        eta=(xx(0,1)*xx(2,3))/(xx(0,2)*xx(1,3))
        subregionA=np.arange(x[0],x[1])
        subregionB=np.arange(x[2],x[3])
        MI=self.mutual_information_m(subregionA,subregionB)
        return eta, MI       
        
