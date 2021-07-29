import numpy as np
import scipy.linalg as la
import numpy.linalg as nla
import numpy.matlib
import itertools

class Params:
    '''
    example: params=Params(delta=2)
    '''
    def __init__(self,
    delta=0,    
    L=100,
    T=0,
    bc=-1,    # 0: open boundary condition; +1: PBC; -1: APBC
    basis='m',    # 'generate Hamiltonian of fermionic ('f') and Majorana basis ('m') or both ('mf')
    dE=None,
    E0=None,
    kappa=0.5,
    history=True
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
        self.dE=dE
        self.E0=E0
        self.kappa=kappa
        self.history=history
        if self.T==np.inf:
            self.once=0
        if 'f' in basis:
            band1sm=np.diag(np.ones(L-1),1)
            band1sm[-1,0]=bc
            id_mat=np.eye(L)
            # Hamiltonian in the ferimion basis
            # self.Hamiltonian_f=np.block([[-self.mu*id_mat-self.t*(band1sm+band1sm.T),-self.Delta*(band1sm-band1sm.T)],
            # [self.Delta*(band1sm-band1sm.T),self.mu*id_mat+self.t*(band1sm+band1sm.T)]])
            # BdG Hamiltonian back to original        
            # self.Hamiltonian_f/=2
            A=-(1-delta)*id_mat+(1+delta)/2*(band1sm+band1sm.T)
            B=-(1+delta)/2*(band1sm.T-band1sm)
            self.Hamiltonian_f=np.block([[A,B],[-B,-A]])


        if 'm' in basis:    
            # Hamiltonian in the Majorana basis
            band=np.vstack([np.ones(L)*(1-delta)*1j,np.ones(L)*(1+delta)*1j]).flatten('F')
            Ham=np.diag(band[:-1],-1)
            Ham[0,-1]=(1+delta)*1j*bc
            Ham=Ham+Ham.conj().T
            self.Hamiltonian_m=Ham

    def bandstructure(self,basis='mf'):
        if 'f' in basis:    
            val,vec=nla.eigh(self.Hamiltonian_f)
            sortindex=np.argsort(val)
            self.val_f=val[sortindex]
            self.vec_f=vec[:,sortindex]
        if 'm' in basis:
            val,vec=nla.eigh(self.Hamiltonian_m) 
            sortindex=np.argsort(val)
            self.val_m=val[sortindex]
            self.vec_m=vec[:,sortindex]       

    def fermi_dist(self,energy,E_F):      
        if self.T==0:
            return np.heaviside(E_F-energy,0)
        elif self.T<np.inf:
            return 1/(1+np.exp((energy-E_F)/self.T))
        else:
            # occ=np.array([1]*64+[0]*64)
            # occ[63],occ[64]=occ[64],occ[63]
            # return occ

            assert self.dE is not None, 'dE is unspecified when T is inf'
            k=int(len(energy)*self.kappa)
            if self.L%2==0:
                index0=np.arange(0,self.L*2,2)
            else:
                index0=np.arange(1,self.L*2,2)
            index=np.random.choice(index0,k//2,replace=False)
            index=np.hstack((index,(index+1)%(2*self.L)))
            E_mean=np.sum(energy[index])/self.L
            while np.abs(E_mean-self.E0)>self.dE:
                if self.L%2==0:
                    index0=np.arange(0,self.L*2,2)
                else:
                    index0=np.arange(1,self.L*2,2)
                index=np.random.choice(index0,k//2,replace=False)
                index=np.hstack((index,(index+1)%(2*self.L)))
                E_mean=np.sum(energy[index])/self.L
                # print(Esum)
            filt=np.zeros(len(energy),dtype=int)
            filt[index]=1
            self.filt=filt
            self.index=index
            self.E_mean=E_mean
            return filt


    def correlation_matrix(self,E_F=0):
        '''
        ??? may be wrong by a transpose
        G_{ij}=[[<f_i f_j^\dagger>,<f_i f_j>],
                [<f_i^\dagger f_j^\dagger>,<f_i^\dagger f_j>]]
        '''
        if not (hasattr(self,'val_f') and hasattr(self,'vec_f')):
            self.bandstructure('f')
        occupancy=self.fermi_dist(self.val_f,E_F)
        occupancy_mat=np.matlib.repmat(occupancy,self.vec_f.shape[0],1)
        # print('Max of imag {:.2f}'.format(np.abs((occupancy_mat*self.vec_f)@self.vec_f.T.conj()).max()))
        self.C_f=((occupancy_mat*self.vec_f)@self.vec_f.T.conj())

    def covariance_matrix_f(self,E_F=0):
        '''
        Gamma from fermionic basis
        Gamma_ij=i<gamma_i gamma_j>/2
        '''
        if not (hasattr(self,'C_f')):
            self.correlation_matrix(E_F)
        # G=self.C_f[self.L:,self.L:]
        # F=self.C_f[self.L:,:self.L]
        G=self.C_f[:self.L,:self.L]
        F=self.C_f[:self.L,self.L:]

        self.G=G
        self.F=F
        A11=1j*(F.T.conj()+F+G-G.T)
        A22=-1j*(F.T.conj()+F-G+G.T)
        A21=-(np.eye(F.shape[0])+F.T.conj()-F-G-G.T)
        A12=-A21.T
        A=np.zeros((2*self.L,2*self.L),dtype=complex)
        even=np.arange(2*self.L)[::2]
        odd=np.arange(2*self.L)[1::2]
        A[np.ix_(even,even)]=A11
        A[np.ix_(even,odd)]=A12
        A[np.ix_(odd,even)]=A21
        A[np.ix_(odd,odd)]=A22
        assert np.abs(np.imag(A)).max()<1e-10, "Covariance matrix not real"
        self.C_m=A
        self.C_m=np.real(A-A.T.conj())/2   
        self.C_m_history=[self.C_m]    

    def covariance_matrix_m(self,E_F=0):
        '''
        Gamma from Majorana basis
        '''
        if not (hasattr(self,'val_m') and hasattr(self,'vec_m')):
            self.bandstructure('m')
        occupancy=self.fermi_dist(self.val_m,E_F)
        occupancy_mat=np.matlib.repmat(occupancy,self.vec_m.shape[0],1)
        self.C_m=(1j*2*(occupancy_mat*self.vec_m)@self.vec_m.T.conj())-1j*np.eye(self.L*2)
        assert np.abs(np.imag(self.C_m)).max()<1e-10, "Covariance matrix not real"        
        self.C_m=np.real(self.C_m-self.C_m.T.conj())/2
        self.C_m_history=[self.C_m]
    
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


    def measure(self,s,ix):
        if not hasattr(self,'C_m'):
            self.covariance_matrix_m()
        if not hasattr(self,'s_history'):
            self.s_history=[]
        if not hasattr(self,'i_history'):
            self.i_history=[]
                
        m=self.C_m_history[-1].copy()

        for i_ind,i in enumerate(ix):
            m[[i,-(len(ix)-i_ind)]]=m[[-(len(ix)-i_ind),i]]
            m[:,[i,-(len(ix)-i_ind)]]=m[:,[-(len(ix)-i_ind),i]]

        self.m=m

        Gamma_LL=m[:-len(ix),:-len(ix)]
        Gamma_LR=m[:-len(ix),-len(ix):]
        Gamma_RR=m[-len(ix):,-len(ix):]       

        proj=self.projection(s)
        Upsilon_LL=proj[:-len(ix),:-len(ix)]
        Upsilon_RR=proj[-len(ix):,-len(ix):]
        Upsilon_RL=proj[-len(ix):,:-len(ix)]
        zero=np.zeros((m.shape[0]-len(ix),len(ix)))
        zero0=np.zeros((len(ix),len(ix)))
        mat1=np.block([[Gamma_LL,zero],[zero.T,Upsilon_RR]])
        mat2=np.block([[Gamma_LR,zero],[zero0,Upsilon_RL]])
        mat3=np.block([[Gamma_RR,np.eye(len(ix))],[-np.eye(len(ix)),Upsilon_LL]])
        self.mat2=mat2
        if np.count_nonzero(mat2):
            Psi=mat1+mat2@(la.solve(mat3,mat2.T))
            # Psi=mat1+mat2@(la.lstsq(mat3,mat2.T)[0])
            assert np.abs(np.trace(Psi))<1e-5, "Not trace zero {:e}".format(np.trace(Psi))
        else:
            Psi=mat1
        
        for i_ind,i in enumerate(ix):
            Psi[[i,-(len(ix)-i_ind)]]=Psi[[-(len(ix)-i_ind),i]]
            Psi[:,[i,-(len(ix)-i_ind)]]=Psi[:,[-(len(ix)-i_ind),i]]
        
        Psi=(Psi-Psi.T)/2   # Anti-symmetrize
        if self.history:
            self.C_m_history.append(Psi)
            self.s_history.append(s)
            self.i_history.append(i)
        else:
            self.C_m_history=[Psi]
            self.s_history=[s]
            self.i_history=[i]

    def c_subregion_f(self,subregion):
        if not hasattr(self,'C'):
            self.correlation_matrix()
        try:
            subregion=np.array(subregion)
        except:
            raise ValueError("The subregion is ill-defined"+subregion)
        subregion_ph=np.concatenate([subregion,subregion+self.L])
        return self.C_f[np.ix_(subregion_ph,subregion_ph)]

    def von_Neumann_entropy_f(self,subregion):
        c_A=self.c_subregion_f(subregion)
        val=nla.eigvalsh(c_A)
        self.val_sh=val
        val=np.sort(val)[:subregion.shape[0]]
        return np.real(-np.sum(val*np.log(val+1e-18j))-np.sum((1-val)*np.log(1-val+1e-18j)))


    def c_subregion_m(self,subregion,Gamma=None):
        if not hasattr(self,'C_m'):
            self.covariance_matrix_f()
        if Gamma is None:
            Gamma=self.C_m_history[-1]
        try:
            subregion=np.array(subregion)
        except:
            raise ValueError("The subregion is ill-defined"+subregion)
        return Gamma[np.ix_(subregion,subregion)]
        
    def von_Neumann_entropy_m(self,subregion):
        c_A=self.c_subregion_m(subregion)
        val=nla.eigvalsh(1j*c_A)
        self.val_sh=val
        val=np.sort(val)
        val=(1-val)/2+1e-18j   #\lambda=(1-\xi)/2
        
        return np.real(-np.sum(val*np.log(val))-np.sum((1-val)*np.log(1-val)))/2

    def mutual_information_f(self,subregion_A,subregion_B):
        s_A=self.von_Neumann_entropy_f(subregion_A)
        s_B=self.von_Neumann_entropy_f(subregion_B)
        assert np.intersect1d(subregion_A,subregion_B).size==0 , "Subregion A and B overlap"
        subregion_AB=np.concatenate([subregion_A,subregion_B])
        s_AB=self.von_Neumann_entropy_f(subregion_AB)
        return s_A+s_B-s_AB

    def mutual_information_m(self,subregion_A,subregion_B):
        assert np.intersect1d(subregion_A,subregion_B).size==0 , "Subregion A and B overlap"
        s_A=self.von_Neumann_entropy_m(subregion_A)
        s_B=self.von_Neumann_entropy_m(subregion_B)
        subregion_AB=np.concatenate([subregion_A,subregion_B])
        s_AB=self.von_Neumann_entropy_m(subregion_AB)
        return s_A+s_B-s_AB

    def measure_batch(self,batchsize,proj_range):
        for _ in range(batchsize):
            i=np.random.randint(*proj_range)
            s=np.random.randint(0,2)
            self.measure(s,[i,i+1])
        return self

    def measure_all(self,s_prob,proj_range=None):
        '''
        The probability of s=0 (unoccupied)
        '''
        if proj_range is None:
            proj_range=np.arange(int(self.L/2),self.L,2)
        for i in proj_range:
            if s_prob==0:
                s=1
            elif s_prob==1:
                s=0
            else:           
                s=s_prob<np.random.rand()
            self.measure(s,[i,i+1])
        return self
    
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
                self.measure(s,[position,position+1])
        return self

    def measure_all_Born(self,proj_range=None,order=None):
        if proj_range is None:
            proj_range=np.arange(int(self.L/2),self.L,2)
        if order=='e2':
            proj_range=np.concatenate((proj_range[::2],proj_range[1::2]))
        if order=='e3':
            proj_range=np.concatenate((proj_range[::3],proj_range[1::3],proj_range[2::3]))
        if order=='e4':
            proj_range=np.concatenate((proj_range[::4],proj_range[1::4],proj_range[2::4]+proj_range[3::4]))
        self.P_0_list=[]
        self.covariance_matrix_f()
        for i in proj_range:
            P_0=(self.C_m_history[-1][i,i+1]+1)/2
            self.P_0_list.append(P_0)
            if np.random.rand() < P_0:                
                self.measure(0,[i,i+1])
            else:
                self.measure(1,[i,i+1])
        return self

    def measure_all_random(self,batchsize,proj_range):
        choice=np.random.choice(range(*proj_range),batchsize,replace=False)
        for i in choice:
            s=np.random.randint(0,2)
            self.measure(s,[i,i+1])  
        return self


    def measure_all_random_even(self,batchsize,proj_range):
        '''
        proj_range: (start,end) tuple
        '''       
        proj_range_even=[i//2 for i in proj_range]
        choice=np.random.choice(range(*proj_range_even),batchsize,replace=False)
        for i in choice:
            s=np.random.randint(0,2)
            self.measure(s,[2*i,2*i+1])  
        return self

    def log_neg(self,subregionA,subregionB,Gamma=None):
        assert np.intersect1d(subregionA,subregionB).size==0 , "Subregion A and B overlap"
        if not hasattr(self,'C_m'):
            self.covariance_matrix_f()
        
        if Gamma is None:
            Gamma=self.C_m_history[-1]
        subregionA=np.array(subregionA)
        subregionB=np.array(subregionB)
        Gm_p= np.block([
            [-Gamma[np.ix_(subregionA,subregionA)],1j*Gamma[np.ix_(subregionA,subregionB)]],
            [1j*Gamma[np.ix_(subregionB,subregionA)],Gamma[np.ix_(subregionB,subregionB)]]
        ])
        Gm_n= np.block([
            [-Gamma[np.ix_(subregionA,subregionA)],-1j*Gamma[np.ix_(subregionA,subregionB)]],
            [-1j*Gamma[np.ix_(subregionB,subregionA)],Gamma[np.ix_(subregionB,subregionB)]]
        ])
        idm=np.eye(Gm_p.shape[0])
        # Gm_x=idm-(idm+1j*Gm_p)@nla.inv(idm-Gm_n@Gm_p)@(idm+1j*Gm_n)
        Gm_x=idm-(idm+1j*Gm_p)@(la.solve((idm-Gm_n@Gm_p),(idm+1j*Gm_n)))
        Gm_x=(Gm_x+Gm_x.T.conj())/2
        xi=nla.eigvalsh(Gm_x)
        subregionAB=np.concatenate([subregionA,subregionB])
        eA=np.sum(np.log(((1+xi+0j)/2)**0.5+((1-xi+0j)/2)**0.5))/2
        chi=nla.eigvalsh(1j*Gamma[np.ix_(subregionAB,subregionAB)])
        sA=np.sum(np.log(((1+chi)/2)**2+((1-chi)/2)**2))/4
        return np.real(eA+sA)

     
    
def cross_ratio(x,L):
        xx=lambda i,j: (np.sin(np.pi/(L)*np.abs(x[i]-x[j])))
        eta=(xx(0,1)*xx(2,3))/(xx(0,2)*xx(1,3))
        return eta
