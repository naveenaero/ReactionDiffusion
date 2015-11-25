"""Reaction Diffusion Equation(5 seconds)

"""

# NumPy and standard library imports
import numpy

# PySPH base and carray/ imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array as gpa
from pysph.base.kernels import CubicSpline

# PySPH solver and integrator
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EulerIntegrator, PECIntegrator
from pysph.sph.integrator_step import EulerStep

# PySPH sph imports
from pysph.sph.equation import Group
from pysph.sph.basic_equations import SummationDensity
# import Diffusion equation
from diffusion_equation import DiffusionEquation, EulerDiffusionStep, DKDSPHStep


# Numerical constants
dim = 1


# domain size and discretization parameters
xmin = -0.5; xmax = 0.5
n = 401
dx = (xmax-xmin)/(n-1)

# CFL value should be less then 0.5 for diffusion equation
cfl = 0.1
# Diffuion coefficient
Do = 0.5
# time step according to cfl condition
dt = cfl*dx**2/Do

# simulation time
tf = 0.01


print 'Step size:', dx
print 'Time step:', dt

# scheme constants
kernel_factor = 1.2
h0 = kernel_factor*dx


class Diffusion1D(Application):
    def create_domain(self):
        return DomainManager(xmin=xmin, xmax=xmax)

    def create_particles(self):
        # particle positions
        x = numpy.arange(xmin, xmax, dx)

        # diffusion coefficients
        D = numpy.ones_like(x)*Do

        # set density
        rho = numpy.ones_like(x)

        # set initial concentration
        cmax = 1
        b = 0
        sigma = 0.1
        c = cmax*numpy.exp(-(x-b)**2/(2.*sigma**2))

        # def initial accelerations
        ac = numpy.zeros_like(x)
        # copy initial condition
        co = numpy.copy(c)
        
        # const h and mass
        h = numpy.ones_like(x) * h0
        m = numpy.ones_like(x) * dx

        # generate fluid particle array
        fluid = gpa(name='fluid', x=x, rho=rho, c=c, h=h, D=D, m=m, h0=h.copy(),
                     ac=ac, co=c)

        # set output arrays
        fluid.add_output_arrays(['c'])
        
        print("1D Diffusion with %d particles"%(fluid.get_number_of_particles()))

        return [fluid,]

    def create_solver(self):
        kernel = CubicSpline(dim=dim)
        integrator = EulerIntegrator(fluid=EulerDiffusionStep())

        solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                        dt=dt, tf=tf, pfreq=5)
        return solver

    def create_equations(self):
        
        equations_diffusion = [

            # do the main acceleration block.
            Group(
                equations=[
                    DiffusionEquation(
                        dest='fluid', sources=['fluid']),
                    ]
                ),
            ]

        
        return equations_diffusion
        

if __name__ == '__main__':
    app = Diffusion1D()
    if app.solver is None:
        app.run()
