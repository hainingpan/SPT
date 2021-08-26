import numpy as np
import scipy.linalg as la
import numpy.linalg as nla
import numpy.matlib

class Params:
    def __init__(self,
    L=100,
    J=1,
    mu=0,
    kappa=0.5,
    dE=2,
    E0=0,
    history=False):
        self.L=L
        self.J=J
        self.mu=mu
        self.kappa=kappa
        self.variance=J**2/L
        self.dE=dE
        self.E0=E0
        self.history=history

        H=np.random.normal(0,J/np.sqrt(L),size=(L,L))
        H=(H+H.conj().T)/np.sqrt(2)+mu*np.eye(L)
        self.Hamiltonian=H
        
    def bandstructure(self):
        val,vec=nla.eigh(self.Hamiltonian)
        sortindex=np.argsort(val)
        self.val=val[sortindex]
        self.vec=vec[:,sortindex]

    def fermi_dist(self,energy,E_F=0):      
        assert self.dE is not None, 'dE is unspecified when T is inf'
        k=int(len(energy)*self.kappa)
        index=np.random.choice(np.arange(len(energy)),k,replace=False)
        E_mean=np.sum(energy[index])/self.L
        
        while np.abs(E_mean-self.E0)>self.dE:
            index=np.random.choice(np.arange(len(energy)),k,replace=False)
            E_mean=np.sum(energy[index])/self.L
        filt=np.zeros(len(energy),dtype=int)
        filt[index]=1
        self.index=index
        self.E_mean=E_mean
        return filt

    def correlation_matrix(self,E_F=0):
        '''
        G_{ij}=<f_i^\dagger f_j>
        '''
        if not (hasattr(self,'val') and hasattr(self,'vec')):
            self.bandstructure()
        occupancy_mat=np.matlib.repmat(self.fermi_dist(self.val,E_F),self.vec.shape[0],1)
        self.C_f=((occupancy_mat*self.vec)@self.vec.T.conj())
    def covariance_matrix(self,E_F=0):
        '''
        c.f. notes
        Maybe differs by a minus sign
        '''
        if not hasattr(self,'C_f'):
            if self.L<np.inf:
                self.correlation_matrix()
            else:
                self.correlation_matrix_inf_fft()
        G=self.C_f
        Gamma_11=1j*(G-G.T)
        Gamma_21=-(np.eye(G.shape[0])-G-G.T)
        Gamma_12=-Gamma_21.T
        Gamma_22=-1j*(G.T-G)
        Gamma=np.zeros((2*G.shape[0],2*G.shape[0]),dtype=complex)
        even=np.arange(2*G.shape[0])[::2]
        odd=np.arange(2*G.shape[0])[1::2]
        Gamma[np.ix_(even,even)]=Gamma_11
        Gamma[np.ix_(even,odd)]=Gamma_12
        Gamma[np.ix_(odd,even)]=Gamma_21
        Gamma[np.ix_(odd,odd)]=Gamma_22
        assert np.abs(np.imag(Gamma)).max()<1e-10, "Covariance matrix not real"        
        self.C_m=np.real(Gamma-Gamma.T.conj())/2
        self.C_m_history=[self.C_m]

    def linearize_index(self,subregion,n,proj=False,k=2):
        try:
            subregion=np.array(subregion)
        except:
            raise ValueError("The subregion is ill-defined"+subregion)
        if proj:
            return sorted(np.concatenate([n*subregion+i for i in range(0,n,k)]))
        else:
            return sorted(np.concatenate([n*subregion+i for i in range(n)]))

    def c_subregion_m(self,subregion,Gamma=None):
        # subregion should be in the format of fermionic sites
        if not hasattr(self,'C_m'):
            self.covariance_matrix()
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

    def mutual_information_m(self,subregion_A,subregion_B):
        assert np.intersect1d(subregion_A,subregion_B).size==0 , "Subregion A and B overlap"
        s_A=self.von_Neumann_entropy_m(subregion_A)
        s_B=self.von_Neumann_entropy_m(subregion_B)
        subregion_AB=np.concatenate([subregion_A,subregion_B])
        s_AB=self.von_Neumann_entropy_m(subregion_AB)
        return s_A+s_B-s_AB,(s_A,s_B,s_AB)
        
    def log_neg(self,subregionA,subregionB,Gamma=None):
        subregionA=self.linearize_index(subregionA,2)
        subregionB=self.linearize_index(subregionB,2)
        assert np.intersect1d(subregionA,subregionB).size==0 , "Subregion A and B overlap"
        if not hasattr(self,'C_m'):
            self.covariance_matrix()
        
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

    def projection(self,s,type='onsite',ignore=False):
        '''
        For type:'onsite'
            occupancy number: s= 0,1 
            (-1)^0 even parity, (-1)^1 odd parity
        For type:'link'
            (o,+)|(o,-)|(e,+)|(e,-)
        '''
        if type=='onsite':
            assert (s==0 or s==1),"s={} is either 0 or 1".format(s)
            blkmat=np.array([[0,-(-1)**s,0,0],
                            [(-1)**s,0,0,0],
                            [0,0,0,(-1)**s],
                            [0,0,-(-1)**s,0]])
            return blkmat

        if type=='link':
            assert (s in ['o+','o-','e+','e-']), "s={} for {} is not defined".format(s,type)
            if not ignore:
                if s=='o+':
                    antidiag=[1,-1,1,-1]
                if s=='o-':
                    antidiag=[-1,1,-1,1]
                if s=='e+':
                    antidiag=[-1,-1,1,1]
                if s=='e-':
                    antidiag=[1,1,-1,-1]
                blkmat=np.diag(antidiag)
                blkmat=np.fliplr(blkmat)
            else:
                if s=='e+':
                    blkmat=np.array([[0,-1,0,0],
                                    [1,0,0,0],
                                    [0,0,0,-1],
                                    [0,0,1,0]])
                if s=='e-':
                    blkmat=-np.array([[0,-1,0,0],
                                    [1,0,0,0],
                                    [0,0,0,-1],
                                    [0,0,1,0]])
                if s=='o+':
                    antidiag=[1,-1,1,-1]
                    blkmat=np.diag(antidiag)
                    blkmat=np.fliplr(blkmat)
                if s=='o-':
                    antidiag=[-1,1,-1,1]
                    blkmat=np.diag(antidiag)
                    blkmat=np.fliplr(blkmat)

            proj=np.zeros((8,8))
            proj[:4,:4]=blkmat
            proj[4:,4:]=blkmat.T
            return proj            

        raise ValueError("type '{}' is not defined".format(type))

    def measure(self,s,ix,type='onsite',ignore=False):
        if not hasattr(self,'C_m'):
            self.covariance_matrix()
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

        proj=self.projection(s,type=type,ignore=ignore)
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


    def measure_all_Born(self,proj_range=None,type='onsite'):
        # proj_range should be in the format of fermionic sites
        if type=='onsite':
            if proj_range is None:
                proj_range=np.arange(self.L//4,self.L//2)

        if type=='link':
            if proj_range is None:
                proj_range=np.arange(self.L//4,self.L//2,2)
        proj_range=self.linearize_index(proj_range,2,proj=True,k=2)

        if not hasattr(self, 'C_m'):
            self.covariance_matrix()
        
        if type=='onsite':
            for i in proj_range:
                P_0=(self.C_m_history[-1][i,i+1]+1)/2
                if np.random.rand() < P_0:                
                    self.measure(0,[i,i+1])
                else:
                    self.measure(1,[i,i+1])
            return self
        if type=='link':
            for i in proj_range:
                Gamma=self.C_m_history[-1][i:i+4,i:i+4]
                gamma1234=-Gamma[0,1]*Gamma[2,3]+Gamma[0,2]*Gamma[1,3]-Gamma[0,3]*Gamma[1,2]
                P={}
                P['o+']=(1+Gamma[1,2]-Gamma[0,3]+gamma1234)/4
                P['o-']=(1-Gamma[1,2]+Gamma[0,3]+gamma1234)/4
                P['e+']=(1+Gamma[1,2]+Gamma[0,3]-gamma1234)/4
                P['e-']=(1-Gamma[1,2]-Gamma[0,3]-gamma1234)/4   
                assert P['o+']>-1e-12,'P[o+]={:e}'.format(P['o+'])     
                assert P['o-']>-1e-12,'P[o-]={:e}'.format(P['o-'])     
                assert P['e+']>-1e-12,'P[e+]={:e}'.format(P['e+'])     
                assert P['e-']>-1e-12,'P[e-]={:e}'.format(P['e-'])  
                P['o+']=max([0,P['o+']])
                P['o-']=max([0,P['o-']])
                P['e+']=max([0,P['e+']])
                P['e-']=max([0,P['e-']])
                s=np.random.choice(['o+','o-','e+','e-'],p=[P['o+'],P['o-'],P['e+'],P['e-']])
                self.measure(s,[i,i+1,i+2,i+3],type='link')
            return self
        raise ValueError("type '{}' is not defined".format(type))
