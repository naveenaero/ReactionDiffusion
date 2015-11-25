import numpy as np
import matplotlib.pyplot as plt
import interp_sph as interp


# geometric factor
xmin = -0.5
xmax = 0.5
xming = -0.50
xmaxg = 0.50
dx = 0.0025
n_points = (xmax-xmin)/dx + 1
print n_points
n_pointsg = (xmaxg-xming)/dx + 1


# cfl and time step
cfl = 0.1
D = 0.5
dt = cfl*dx**2/D

# time variables
tf = 0.002
nt = int(tf/dt)
print 'number of iterations:', nt
print 'dx:', dx
print 'dt:', dt

# kernel factor
kernel_factor = 1.2
h0 = kernel_factor*dx

class Particles:
	def __init__(self, D, cmax, b, sigma):
		self.xg = np.linspace(xming, xmaxg, n_pointsg)
		self.x = np.linspace(xmin, xmax, n_points)
		self.ngl = (len(self.xg)-len(self.x))/2.
		self.ngr = len(self.x) + self.ngl
		print self.ngl, self.ngr

		self.D = np.ones_like(self.xg)*D

		self.c = cmax*np.exp(-(self.xg-b)**2/(2*sigma**2))

		self.rho = np.ones_like(self.xg)

		self.m = np.ones_like(self.xg)*dx

		self.acc = np.zeros_like(self.x)

		self.h = h0

		self.EPS = self.h**2

	def compute_acc(self):
		mj = np.copy(self.m)
		rhoj = np.copy(self.rho)
		Dj = np.copy(self.D)
		cj = np.copy(self.c)
		DWij = np.zeros_like(self.xg)
		rj = np.copy(self.xg)
		hij = self.h
		self.acc = np.zeros_like(self.x)

		for i in range(int(self.ngl), int(self.ngl)+len(self.x)):
			Dij = 4*self.D[i]*Dj/(self.D[i]+Dj)
			
			rij = rj - self.xg[i]

			rhoipj = self.rho[i] + rhoj

			DWij = interp.cubic_spline_kernel_derivative(rij, hij)

			rijdotDWij = rij*DWij
			visc = rijdotDWij/(rij**2 + 1e-6)

			cij = self.c[i] - cj
			
			self.acc[i-int(self.ngl)] = np.sum(mj*Dij*rhoipj*visc*cij/(rhoj*self.rho[i]))


class Integrate:
	def __init__(self, dt):
		self.dt = dt
		self.dtb2 = dt*0.5

	def stage1(self, fluid):
		ngl = fluid.ngl
		ngr = fluid.ngr
		fluid.c[ngl:ngr] += self.dtb2*fluid.acc[ngl:ngr]

	def stage2(self, fluid):
		ngl = fluid.ngl
		ngr = fluid.ngr
		fluid.c[ngl:ngr] += self.dt*fluid.acc

def run(dt, nt, D):
	fluid = Particles(D, 1, 0, 0.1)
	integrator = Integrate(dt)
	for t in range(nt):
		# print t
		fluid.compute_acc()
		integrator.stage2(fluid)

	plt.plot(fluid.x, fluid.c[fluid.ngl:fluid.ngr])
	plt.show()

run(dt, nt, D)

