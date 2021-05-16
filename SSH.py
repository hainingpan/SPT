import numpy as np
import scipy.linalg as la
import numpy.linalg as nla
import numpy.matlib
import itertools

class Params:
    def __init__(self,
    L=100,
    delta=0,
    T=0,
    bc=-1):
        self.L=L
        self.delta=delta
        self.v=1-delta
        self.w=1+delta
        self.T=T
        self.bc=bc
        band=np.vstack([np.ones(L)*self.v,np.ones(L)*self.w]).flatten('F')
        Ham=np.diag(band[:-1],1)
        Ham[0,2*L-1]=self.w*bc
        self.Hamiltonian=Ham+Ham.T
    
    def bandstructure(self):
        val,vec=nla.eigh(self.Hamiltonian)
        sortindex=np.argsort(val)
        self.val=val[sortindex]
        self.vec=vec[:,sortindex]    


    def fermi_dist(self,energy,E_F):      
        if self.T==0:
            return np.heaviside(E_F-energy,0)
        else:
            return 1/(1+np.exp((energy-E_F)/self.T)) 

    def correlation_matrix(self,E_F=0):
        '''
        G_{ij}=<f_i f_j^\dagger>
        '''
        if not (hasattr(self,'val') and hasattr(self,'vec')):
            self.bandstructure()
        occupancy_mat=np.matlib.repmat(self.fermi_dist(self.val,E_F),self.vec.shape[0],1)
        self.C_f=(occupancy_mat*self.vec)@self.vec.T.conj()
    
    def covariance_matrix(self,E_F=0):
        '''
        c.f. notes
        Maybe differs by a minus sign
        '''
        if not hasattr(self,'C_f'):
            self.correlation_matrix()
        G=self.C_f
        Gamma_11=1j*(G-G.T)
        Gamma_21=-(np.eye(2*self.L)-G-G.T)
        Gamma_12=-Gamma_21.T
        Gamma_22=-1j*(G.T-G)
        Gamma=np.zeros((4*self.L,4*self.L),dtype=complex)
        even=np.arange(4*self.L)[::2]
        odd=np.arange(4*self.L)[1::2]
        Gamma[np.ix_(even,even)]=Gamma_11
        Gamma[np.ix_(even,odd)]=Gamma_12
        Gamma[np.ix_(odd,even)]=Gamma_21
        Gamma[np.ix_(odd,odd)]=Gamma_22
        assert np.abs(np.imag(Gamma)).max()<1e-10, "Covariance matrix not real"        
        self.C_m=np.real(Gamma-Gamma.T.conj())/2
        self.C_m_history=[self.C_m]

    def c_subregion_f(self,subregion):
        if not hasattr(self,'C'):
            self.correlation_matrix()
        try:
            subregion=np.array(subregion)
        except:
            raise ValueError("The subregion is ill-defined"+subregion)
        return self.C_f[np.ix_(subregion,subregion)]

    def von_Neumann_entropy_f(self,subregion):
        c_A=self.c_subregion_f(subregion)
        val=nla.eigvalsh(c_A)
        self.val_sh=val
        val=np.sort(val)
        return np.real(-np.sum(val*np.log(val+1e-18j))-np.sum((1-val)*np.log(1-val+1e-18j)))

    def c_subregion_m(self,subregion,Gamma=None):
        if not hasattr(self,'C_m'):
            self.covariance_matrix()
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

    def projection(self,s,type='onsite'):
        '''
        For type:'onsite'
            occupancy number: s= 0,1 
            (-1)^0 even parity, (-1)^1 odd parity
        For type:'link'
            (o,+):
            (o,-):
            (e,+):
            (e,-):
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
            proj=np.zeros((8,8))
            proj[:4,:4]=blkmat
            proj[4:,4:]=blkmat.T
            return proj            

        raise ValueError("type '{}' is not defined".format(type))
        

    def measure(self,s,ix,type='onsite'):
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
        # # i<-> -2
        # m[[i,-2]]=m[[-2,i]]
        # m[:,[i,-2]]=m[:,[-2,i]]
        # # j<->-1
        # m[[j,-1]]=m[[-1,j]]
        # m[:,[j,-1]]=m[:,[-1,j]]

        self.m=m

        Gamma_LL=m[:-len(ix),:-len(ix)]
        Gamma_LR=m[:-len(ix),-len(ix):]
        Gamma_RR=m[-len(ix):,-len(ix):]       

        proj=self.projection(s,type=type)
        Upsilon_LL=proj[:-len(ix),:-len(ix)]
        # Upsilon_LR=proj[:-len(ix),-len(ix):]
        Upsilon_RR=proj[-len(ix):,-len(ix):]
        Upsilon_RL=proj[-len(ix):,:-len(ix)]
        zero=np.zeros((self.L*4-len(ix),len(ix)))
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
        # Psi[[j,-1]]=Psi[[-1,j]]
        # Psi[:,[j,-1]]=Psi[:,[-1,j]]

        # Psi[[i,-2]]=Psi[[-2,i]]
        # Psi[:,[i,-2]]=Psi[:,[-2,i]]
        
        Psi=(Psi-Psi.T)/2   # Anti-symmetrize
        self.C_m_history.append(Psi)
        self.s_history.append(s)
        self.i_history.append(i)

    # def measure(self,s,i,j):
    #     if not hasattr(self,'C_m'):
    #         self.covariance_matrix()
    #     if not hasattr(self,'s_history'):
    #         self.s_history=[]
    #     if not hasattr(self,'i_history'):
    #         self.i_history=[]
                
    #     m=self.C_m_history[-1].copy()
    #     # i<-> -2
    #     m[[i,-2]]=m[[-2,i]]
    #     m[:,[i,-2]]=m[:,[-2,i]]
    #     # j<->-1
    #     m[[j,-1]]=m[[-1,j]]
    #     m[:,[j,-1]]=m[:,[-1,j]]

    #     self.m=m

    #     Gamma_LL=m[:-2,:-2]
    #     Gamma_LR=m[:-2,-2:]
    #     Gamma_RR=m[-2:,-2:]       

    #     proj=self.projection(s)
    #     Upsilon_LL=proj[:-2,:-2]
    #     # Upsilon_LR=proj[:-2,-2:]
    #     Upsilon_RR=proj[-2:,-2:]
    #     Upsilon_RL=proj[-2:,:-2]
    #     zero=np.zeros((self.L*4-2,2))
    #     zero0=np.zeros((2,2))
    #     mat1=np.block([[Gamma_LL,zero],[zero.T,Upsilon_RR]])
    #     mat2=np.block([[Gamma_LR,zero],[zero0,Upsilon_RL]])
    #     mat3=np.block([[Gamma_RR,np.eye(2)],[-np.eye(2),Upsilon_LL]])
    #     self.mat2=mat2
    #     if np.count_nonzero(mat2):
    #         Psi=mat1+mat2@(la.solve(mat3,mat2.T))
    #         # Psi=mat1+mat2@(la.lstsq(mat3,mat2.T)[0])
    #         assert np.abs(np.trace(Psi))<1e-5, "Not trace zero {:e}".format(np.trace(Psi))
    #     else:
    #         Psi=mat1
        
    #     Psi[[j,-1]]=Psi[[-1,j]]
    #     Psi[:,[j,-1]]=Psi[:,[-1,j]]

    #     Psi[[i,-2]]=Psi[[-2,i]]
    #     Psi[:,[i,-2]]=Psi[:,[-2,i]]
        
    #     Psi=(Psi-Psi.T)/2   # Anti-symmetrize
    #     self.C_m_history.append(Psi)
    #     self.s_history.append(s)
    #     self.i_history.append(i)

    def measure_all(self,s_prob,proj_range=None):
        '''
        The probability of s=0 (unoccupied)
        '''
        if proj_range is None:
            proj_range=np.arange(self.L,self.L*2,2)
        for i in proj_range:
            if s_prob==0:
                s=1
            elif s_prob==1:
                s=0
            else:           
                s=s_prob<np.random.rand()
            self.measure(s,i,i+1)
        return self

    def measure_all_Born(self,proj_range=None,order=None,type='onsite'):
        if proj_range is None:
            if type=='onsite':
                proj_range=np.arange(self.L,self.L*2,2)
            if type=='link':
                proj_range=np.arange(self.L,self.L*2,4)

        if order=='e2':
            proj_range=np.concatenate((proj_range[::2],proj_range[1::2]))
        if order=='e3':
            proj_range=np.concatenate((proj_range[::3],proj_range[1::3],proj_range[2::3]))
        if order=='e4':
            proj_range=np.concatenate((proj_range[::4],proj_range[1::4],proj_range[2::4]+proj_range[3::4]))
        # self.P_0_list=[]
        self.covariance_matrix()
        if type=='onsite':
            for i in proj_range:
                P_0=(self.C_m_history[-1][i,i+1]+1)/2
                # self.P_0_list.append(P_0)
                if np.random.rand() < P_0:                
                    self.measure(0,[i,i+1])
                else:
                    self.measure(1,[i,i+1])
            return self
        if type=='link':
            for i in proj_range:
                Gamma=self.C_m_history[-1][i:i+4,i:i+4]
                P={}
                P['o+']=(1+Gamma[1,2])/2*(1-Gamma[0,3])/2
                P['o-']=(1-Gamma[1,2])/2*(1+Gamma[0,3])/2
                P['e+']=(1+Gamma[1,2])/2*(1+Gamma[0,3])/2
                P['e-']=(1-Gamma[1,2])/2*(1-Gamma[0,3])/2
                # print((P.values()))
                s=np.random.choice(['o+','o-','e+','e-'],p=[P['o+'],P['o-'],P['e+'],P['e-']])
                self.measure(s,[i,i+1,i+2,i+3],type='link')
            return self

        raise ValueError("type '{}' is not defined".format(type))

    def log_neg(self,subregionA,subregionB,Gamma=None):
        assert np.intersect1d(subregionA,subregionB).size==0 , "Subregion A and B overlap"
        if not hasattr(self,'C_m'):
            self.covariance_matrix_m()
        
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
