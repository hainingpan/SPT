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
    dmax=100,
    bc=-1,    # 0: open boundary condition; +1: PBC; -1: APBC
    basis='f',    # 'generate Hamiltonian of fermionic ('f') and Majorana basis ('m') or both ('mf')
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

        if L<np.inf:
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
        else:
            self.dmax=dmax

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
    
    def E_k(self,k,branch):
        return branch*np.sqrt(2+2*self.delta**2-2*(1-self.delta**2)*np.cos(k))

    def fermi_dist_k(self,k,branch,E_F=0):
        if self.T==0:
            # return np.heaviside(E_F-self.E_k(k,branch),0)
            if branch==1:
                return 0*k
            if branch==-1:
                return 1-0*k
            ValueError('branch (%d) not defined'.format(branch))
        else:
            return 1/(1+np.exp((self.E_k(k,branch)-E_F)/self.T))

    def fermi_dist(self,energy,E_F):      
        if self.T==0:
            return np.heaviside(E_F-energy,0)
        elif self.T<np.inf:
            return 1/(1+np.exp((energy-E_F)/self.T))
        else:

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

    def correlation_matrix_inf_fft(self,threshold=1024):
        '''
        self.dmax: the maximal distance (in terms of unit cell) 
        Directly call fft to evaluate the integral
        '''
        assert self.L==np.inf, "Wire length should be inf"
        d=max(2*self.dmax,threshold)
        k_list=np.arange(0,2*np.pi,2*np.pi/d)
        fermi_dist_k_p=self.fermi_dist_k(np.arange(0,2*np.pi,2*np.pi/d),1)
        fermi_dist_k_m=self.fermi_dist_k(np.arange(0,2*np.pi,2*np.pi/d),-1)

        costheta=1/2*(-self.mu-2*np.cos(k_list)*self.t)/(self.E_k(k_list,1)+1e-18)
        sintheta=-1/2*(2*self.Delta*np.sin(k_list))/(self.E_k(k_list,1)+1e-18)
        integrand_11=fermi_dist_k_m*(1-costheta)/2+fermi_dist_k_p*(1+costheta)/2
        integrand_22=fermi_dist_k_m*(1+costheta)/2+fermi_dist_k_p*(1-costheta)/2
        integrand_12=fermi_dist_k_m*(1j/2)*sintheta+fermi_dist_k_p*(-1j/2)*sintheta
        integrand_21=fermi_dist_k_m*(-1j/2)*sintheta+fermi_dist_k_p*(1j/2)*sintheta

        A_11=np.fft.ifftshift(np.fft.ifft(integrand_11))
        A_22=np.fft.ifftshift(np.fft.ifft(integrand_22))
        A_12=np.fft.ifftshift(np.fft.ifft(integrand_12))
        A_21=np.fft.ifftshift(np.fft.ifft(integrand_21))

        C_f_11=np.zeros((self.dmax,self.dmax),dtype=complex)
        C_f_22=np.zeros((self.dmax,self.dmax),dtype=complex)
        C_f_12=np.zeros((self.dmax,self.dmax),dtype=complex)
        C_f_21=np.zeros((self.dmax,self.dmax),dtype=complex)
        for i in range(self.dmax):
            C_f_11[i]=A_11[d//2-i:d//2+self.dmax-i]
            C_f_22[i]=A_22[d//2-i:d//2+self.dmax-i]
            C_f_12[i]=A_12[d//2-i:d//2+self.dmax-i]
            C_f_21[i]=A_21[d//2-i:d//2+self.dmax-i]

        C_f=np.block([[C_f_11,C_f_12],[C_f_21,C_f_22]])
        C_f_err=np.imag(C_f).__abs__().max()
        assert C_f_err<1e-12, 'C_f not real; the max imag is {:e}'.format(C_f_err)
        self.C_f=np.real(C_f)


    def covariance_matrix_f(self,E_F=0):
        '''
        Gamma from fermionic basis
        Gamma_ij=i<gamma_i gamma_j>/2
        '''
        if not (hasattr(self,'C_f')):
            if self.L<np.inf:
                self.correlation_matrix(E_F)
            else:
                self.correlation_matrix_inf_fft()

        # G=self.C_f[self.L:,self.L:]
        # F=self.C_f[self.L:,:self.L]
        if self.L<np.inf:
            G=self.C_f[:self.L,:self.L]
            # F=self.C_f[:self.L,self.L:]
            F=self.C_f[self.L:,:self.L]
        else:
            G=self.C_f[:self.dmax,:self.dmax]
            F=self.C_f[self.dmax:,:self.dmax]

        self.G=G
        self.F=F
        A11=1j*(F.T.conj()+F+G-G.T)
        A22=-1j*(F.T.conj()+F-G+G.T)
        A21=-(np.eye(F.shape[0])+F.T.conj()-F-G-G.T)
        A12=-A21.T
        if self.L<np.inf:
            A=np.zeros((2*self.L,2*self.L),dtype=complex)
        else:
            A=np.zeros((2*self.dmax,2*self.dmax),dtype=complex)
        # even=np.arange(2*self.L)[::2]
        # odd=np.arange(2*self.L)[1::2]
        # A[np.ix_(even,even)]=A11
        # A[np.ix_(even,odd)]=A12
        # A[np.ix_(odd,even)]=A21
        # A[np.ix_(odd,odd)]=A22
        A[::2,::2]=A11
        A[::2,1::2]=A12
        A[1::2,::2]=A21
        A[1::2,1::2]=A22

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
            self.covariance_matrix_f()
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

    def linearize_index(self,subregion,n,k=2,proj=False):
        try:
            subregion=np.array(subregion)
        except:
            raise ValueError("The subregion is ill-defined"+subregion)
        if proj:
            return sorted(np.concatenate([n*subregion+i for i in range(0,n,k)]))
        else:
            return sorted(np.concatenate([n*subregion+i for i in range(n)]))

    def c_subregion_m(self,subregion,Gamma=None):
        if not hasattr(self,'C_m'):
            self.covariance_matrix_f()
        if Gamma is None:
            Gamma=self.C_m_history[-1]
        subregion=self.linearize_index(subregion,2)
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
            proj_range=np.arange(self.L//4,self.L//2)
        proj_range=self.linearize_index(proj_range,2,proj=True,k=2)
        for i in proj_range:
            if s_prob==0:
                s=1
            elif s_prob==1:
                s=0
            else:           
                s=s_prob<np.random.rand()
            self.measure(s,[i,i+1])
        return self
    
   

    def measure_all_Born(self,proj_range=None,order=None,prob=None):
        # proj_range should be in the format of fermionic sites
        if proj_range is None:
            proj_range=np.arange(self.L//4,self.L//2)
        if order=='e2':
            proj_range=np.concatenate((proj_range[::2],proj_range[1::2]))
        if order=='e3':
            proj_range=np.concatenate((proj_range[::3],proj_range[1::3],proj_range[2::3]))
        if order=='e4':
            proj_range=np.concatenate((proj_range[::4],proj_range[1::4],proj_range[2::4]+proj_range[3::4]))
        proj_range=self.linearize_index(proj_range,2,proj=True,k=2)
        # self.P_0_list=[]
        if not hasattr(self, 'C_m'):
            self.covariance_matrix_f()
        for index,i in enumerate(proj_range):
            if prob is None:
                P_0=(self.C_m_history[-1][i,i+1]+1)/2
                # self.P_0_list.append(P_0)
            else:
                if isinstance(prob,list):
                    assert len(prob)==len(proj_range), "len of prob {:d} not equal to len of proj_range {:d}".format(len(prob),len(proj_range))
                    P_0=prob[index]
                else:
                    P_0=prob
            if np.random.rand() < P_0:                
                self.measure(0,[i,i+1])
            else:
                self.measure(1,[i,i+1])
        return self


    def log_neg(self,subregionA,subregionB,Gamma=None):
        subregionA=self.linearize_index(subregionA,2)
        subregionB=self.linearize_index(subregionB,2)
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
    if L<np.inf:
        xx=lambda i,j: (np.sin(np.pi/(L)*np.abs(x[i]-x[j])))
    else:
        xx=lambda i,j: np.abs(x[i]-x[j])
    eta=(xx(0,1)*xx(2,3))/(xx(0,2)*xx(1,3))
    return eta

from scipy.interpolate import UnivariateSpline
def find_inflection(x,y):
    spl=UnivariateSpline(x,y,s=0)
    spld=spl.derivative()
    x_fit=np.linspace(x.min(),x.max(),500)
    y_fit=spl(x_fit)
    yd_fit=spld(x_fit)    
    x_max_index=np.argmax(np.abs(yd_fit))
    return x_fit[x_max_index],y_fit[x_max_index]
