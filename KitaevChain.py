import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la

class Params:
    '''
    example: params=Params(mu=2)
    '''
    def __init__(self,
    mu=1,
    t=1,
    Delta=1,
    L=100,
    T=0,
    bc=1    # 0: open boundary condition; >0: PBC; <0: APBC
    ):
        self.mu=mu
        self.t=t
        self.Delta=Delta
        self.L=L
        self.tau_z=sp.dia_matrix(np.diag([1,-1]))
        self.tau_y=sp.dia_matrix(np.array([[0,-1j],[1j,0]]))
        self.bc=bc
        self.T=T
        self.band1sm=sp.diags([1],[1],(L,L)).tocsr()
        self.bandm1sm=sp.diags([1],[-1],(L,L)).tocsr()
        self.band1sm[-1,0]=1*(2*np.heaviside(bc,1/2)-1)
        self.bandm1sm[0,-1]=1*(2*np.heaviside(bc,1/2)-1)
        self.Hamiltonian=-self.mu*sp.kron(self.tau_z,sp.identity(self.L))-sp.kron(self.t*self.tau_z+1j*self.Delta*self.tau_y,self.band1sm)-sp.kron(self.t*self.tau_z-1j*self.Delta*self.tau_y,self.bandm1sm)
    
    def bandstructure(self):        
        val,vec=la.eigh(self.Hamiltonian.toarray())
        sortindex=np.argsort(val)
        self.val=val[sortindex]
        self.vec=vec[:,sortindex]

    def fermi_dist(self,energy,E_F):      
        if self.T==0:
            return np.heaviside(E_F-energy,0)
        else:
            return 1/(1+np.exp((energy-E_F)/self.T)) 


    def covariance_matrix(self,E_F=0):
        if not (hasattr(self,'val') and hasattr(self,'vec')):
            self.bandstructure()
        occupancy=self.fermi_dist(self.val,E_F)
        occupancy_mat=np.matlib.repmat(self.fermi_dist(self.val,E_F),self.vec.shape[0],1)
        self.C=np.real((occupancy_mat*self.vec)@self.vec.T.conj())

    def c_subregion(self,subregion):
        if not hasattr(self,'C'):
            self.covariance_matrix()
        XX,YY=np.meshgrid(np.arange(2*self.L),np.arange(2*self.L))
        mask_hh=np.isin(XX,subregion)*np.isin(YY,subregion)
        mask_hp=np.isin(XX,subregion)*np.isin(YY,subregion+self.L)
        mask_ph=np.isin(XX,subregion+self.L)*np.isin(YY,subregion)
        mask_pp=np.isin(XX,subregion+self.L)*np.isin(YY,subregion+self.L)
        mask=mask_hh+mask_hp+mask_ph+mask_pp
        return self.C[mask].reshape((2*subregion.shape[0],2*subregion.shape[0]))

    def von_Neumann_entropy(self,subregion):
        c_A=self.c_subregion(subregion)
        val,vec=la.eigh(c_A)
        self.val_sh=val
        val=np.sort(val)[:subregion.shape[0]]
        return np.real(-np.sum(val*np.log(val+1e-18j))-np.sum((1-val)*np.log(1-val+1e-18j)))

    def mutual_information(self,subregion_A,subregion_B):
        s_A=self.von_Neumann_entropy(subregion_A)
        s_B=self.von_Neumann_entropy(subregion_B)
        subregion_AB=np.hstack([subregion_A,subregion_B])
        s_AB=self.von_Neumann_entropy(subregion_AB)
        return s_A+s_B-s_AB
    