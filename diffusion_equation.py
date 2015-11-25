import numpy as np
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep

class DiffusionEquation(Equation):
    ''' Diffusion Equation formulation by CITE HERE'''

    def __init__(self, dest, sources):
        super(DiffusionEquation, self).__init__(dest, sources)


    def initialize(self, d_idx, d_ac):
        d_ac[d_idx] = 0.0
        
        
    def loop(self, d_idx, s_idx, d_c, d_ac, s_c, s_m, d_rho, s_rho, d_D, s_D, XIJ, R2IJ, DWIJ, EPS):
        # compute average diffusion coefficient
        Di = d_D[d_idx]
        Dj = s_D[s_idx]
        Dij = 4.0*Di*Dj/(Di+Dj)

        EPS = 1e-6
        
        # compute viscous term
        rijdotdwij = XIJ[0]*DWIJ[0]
        visc = rijdotdwij/(R2IJ + EPS)

        # compute density sum
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        
        mj = s_m[s_idx]

        cij = -(s_c[s_idx] - d_c[d_idx])

        # compute acceleration term
        d_ac[d_idx] += mj*Dij*(rhoi+rhoj)*visc*cij/(rhoi*rhoj)

        


class DKDSPHStep(IntegratorStep):
    def initialise(self, d_idx, d_c, d_co):
        d_co[d_idx] = d_c[d_idx]
    
    def stage1(self, d_idx, d_c, d_co, d_ac, dt):
        dtb2 = 0.5*dt
        d_c[d_idx] = d_co[d_idx] + dtb2*d_ac[d_idx]
    
    def stage2(self, d_idx, d_co, d_c, d_ac, dt):
        d_c[d_idx] = d_co[d_idx] + dt*d_ac[d_idx]

class EulerDiffusionStep(IntegratorStep):
    def stage1(self, d_idx, d_c, d_ac, dt):
        d_c[d_idx] += dt*d_ac[d_idx]

