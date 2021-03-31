import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la

class Params:
    def __init__(self,
    L=100,
    v=1,
    w=1,
    T=0,
    pbc=True):
        self.L=L
        self.v=v
        self.w=w
        self.T=T
        self.pbc=pbc
        band=np.vstack([np.ones(L)*v,np.ones(L)*w]).flatten('F')        
        Ham=sp.diags(np.array([band[:-1],band[:-1]]),np.array([-1,1]),shape=(2*L,2*L)).tocsr()
        Ham[0,2*L-1]=w*pbc
        Ham[2*L-1,0]=w*pbc
        self.Hamiltonian=Ham

    def fermi_dist(self,energy,E_F):      
        if self.T==0:
            return np.heaviside(E_F-energy,0)
        else:
            return 1/(1+np.exp((energy-E_F)/self.T))

    def bandstructure(self):
        val,vec=la.eigh(self.Hamiltonian.toarray())
        sortindex=np.argsort(val)
        self.val=val[sortindex]
        self.vec=vec[:,sortindex]

    def covariance_matrix_loop(self):
        '''
        Use loop to calculate covariance matrix (deprecated)
        '''
        self.C_loop=np.zeros((2*self.L,2*self.L))
        for i in range(2*self.L):
            for j in range(2*self.L):
                self.C_loop[i,j]=self.c_ij(i,j)

    def covariance_matrix(self,E_F=0):
        if not (hasattr(self,'val') and hasattr(self,'vec')):
            self.bandstructure()
        occupancy=self.fermi_dist(self.val,E_F)
        occupancy_mat=np.matlib.repmat(self.fermi_dist(self.val,E_F),self.vec.shape[0],1)
        self.C=(occupancy_mat*self.vec)@self.vec.T.conj()
        


    def c_ij(self,i,j,E_F=0):
        if not (hasattr(self,'val') and hasattr(self,'vec')):
            self.bandstructure()

        occupancy=self.fermi_dist(self.val,E_F)
        bra_i=self.vec[i,:] # c_i
        ket_j=self.vec[j,:] # c_j
        return np.sum(bra_i.conj()*ket_j*occupancy)

    def c_subregion(self,subregion):
        if not hasattr(self,'C'):
            self.covariance_matrix()
        XX,YY=np.meshgrid(np.arange(2*self.L),np.arange(2*self.L))
        mask=np.isin(XX,subregion)*np.isin(YY,subregion)        
        return self.C[mask].reshape((subregion.shape[0],subregion.shape[0]))

    def von_Neumann_entropy(self,subregion):
        c_A=self.c_subregion(subregion)
        val,vec=la.eigh(c_A)
        self.val_sh=val
        # return np.real(-np.sum(val*np.log(val+1e-18j)))
        return np.real(-np.sum(val*np.log(val+1e-18j))-np.sum((1-val)*np.log(1-val+1e-18j)))

    def mutual_information(self,subregion_A,subregion_B):
        s_A=self.von_Neumann_entropy(subregion_A)
        s_B=self.von_Neumann_entropy(subregion_B)
        subregion_AB=np.hstack([subregion_A,subregion_B])
        s_AB=self.von_Neumann_entropy(subregion_AB)
        return s_A+s_B-s_AB