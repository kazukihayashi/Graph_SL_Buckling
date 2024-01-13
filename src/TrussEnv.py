import array
import numpy as np
np.random.seed(0)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import Plotter
from numba import f8, i4, b1
import numba as nb
import Bezier
import Bif

@nb.jit(nb.types.Tuple((i4,i4,f8[:,:],i4[:,:]))(i4,i4,f8))
def InitializeGeometry(nx,ny,span):

	## node
	nk = (nx+1)*(ny+1)
	node = np.zeros((nk,3),dtype=np.float64)
	for i in range(nk):
		iy, ix = np.divmod(i,nx+1)
		node[i,1] = iy*span/ny
		node[i,0] = ix*span/nx

	## member
	nm = (1+4*ny)*nx+ny # CAUTION: If wrong value is computed, 
	connectivity = np.zeros((nm,2),dtype=np.int32)

	count = 0
	## horizontal member
	for i in range(ny+1):
		for j in range(nx):
			connectivity[count,0] = i*(nx+1)+j
			connectivity[count,1] = i*(nx+1)+j+1
			count += 1
	## vertical member
	for i in range(ny):
		for j in range(nx+1):
			connectivity[count,0] = i*(nx+1)+j
			connectivity[count,1] = (i+1)*(nx+1)+j
			count += 1
	## bracing member
	for i in range(ny):
		for j in range(nx):
			connectivity[count,0] = i*(nx+1)+j
			connectivity[count,1] = (i+1)*(nx+1)+j+1
			count += 1
			connectivity[count,0] = i*(nx+1)+j+1
			connectivity[count,1] = (i+1)*(nx+1)+j
			count += 1

	assert nm == count, ('Number of members (nm) is incorrect.')

	return nk,nm,node,connectivity

class Truss():

	def __init__(self):
		self.E = 2.05e5 # Young's modulus [N/mm^2]
		# self.fy = 235 # [N/mm^2]
		self.nfv = 5
		self.nfw = 1

	def reset(self,nx=None,ny=None,span=10000,test=False):
		'''
		nx(int)
		ny(int)
		span(float)
		'''

		if test:
			self.NX = 6 # ex.1) 5 ex.2) 6
			self.NY = 6 # ex.1) 5 ex.2) 4
		else:
			self.NX = nx
			self.NY = ny

		self.nk,self.nm,self.init_node,self.connectivity = InitializeGeometry(self.NX,self.NY,span)

		# determine heights
		self.node = self.randomize_shape(test=test,span=span)

		# section
		self.section = np.ones(self.nm,dtype=np.float64)*1000 #[mm^2]
		
		# member lengths
		self.lengths = np.linalg.norm(self.node[self.connectivity[:,1]]-self.node[self.connectivity[:,0]],axis=1)

		# randomize boundary conditions
		self.set_support()
		self.set_load(self.connectivity,self.lengths,self.section)

		# stress
		self.stress = np.zeros(self.nm,dtype=np.float64)

		self.target = None

		return

	def set_support(self):

		# support condition
		self.support = np.zeros((self.nk,3),dtype=bool)
		# self.pin_nodes = np.array([0,self.NX,(self.NX+1)*self.NY,self.nk-1],dtype=np.int32) # 4点支持
		self.pin_nodes = np.sort(np.concatenate([np.arange(self.NX+1),self.nk-1-np.arange(self.NX+1),np.arange(1,self.NY)*(self.NX+1),np.arange(2,self.NY+1)*(self.NX+1)-1]))
		self.support[self.pin_nodes] = True

		return

	def set_load(self,connectivity,length,area):

		density = 78.5e-6 #[N/mm^3]
		weight = length*area*density #[N] for each member

		# loading condition
		self.load = np.zeros((self.nk,3),dtype=np.float64)
		for i in range(connectivity.shape[0]):
			self.load[connectivity[i,:],2] += -weight[i]/2

		return

	def randomize_shape(self,test,span):
		nu = 4 # Bezier curve with a degree of (nu-1)
		nv = 4 # Bezier curve with a degree of (nv-1)
		self.cp = np.zeros([nu,nv,3])
		for i in range(nu):
			self.cp[i,:,0] = i*span/(nu-1)
		for i in range(nv):
			self.cp[:,i,1] = i*span/(nv-1)
		self.cp[1,1,0:2] += 0.1*span/(nu-1)
		if test:
			node = np.copy(self.init_node)
			rise_to_span = 0.5 # caution: rise of Bezier control points, not structural shape
			self.cp[1:nu-1,1:nv-1,2] = np.ones([nu-2,nv-2])*span*rise_to_span
		else:
			node = np.copy(self.init_node)
			rise_to_span = 0.5 # caution: rise of Bezier control points, not structural shape
			rand = 0.5
			self.cp[1:nu-1,1:nv-1,2] = (np.ones([nu-2,nv-2])-rand/2+np.random.rand(nu-2,nv-2)*rand)*span*rise_to_span
		uv = node[:,0:2]/span
		xyz = Bezier.bezierplot(uv,self.cp)
		node[:,2] = xyz[:,2]
		return node

	def update_node_v(self,primary_mode):
		'''
		0 # support
		1 # load
		2-4 # 1次固有モード
		'''
		v = np.ones((self.nk,self.nfv),dtype=np.float32)
		v[self.pin_nodes,0] = 1
		v[:,1] = np.abs(self.load[:,2])/800
		v[:,2:5] = np.copy(primary_mode)/primary_mode.max()

		return v

	def update_edge_w(self):
		'''
		0 # stress
		'''
		w=np.zeros((self.nm,self.nfw),dtype=np.float32)
		# ## 0: length
		# w[:,0] = np.linalg.norm(self.node[self.connectivity[:,1]]-self.node[self.connectivity[:,0]],axis=1)/1500
		## 0: stress
		d, s, _ = Bif.LinearStructuralAnalysis(self.node*1e-3,self.connectivity,self.support,self.load,self.section*1e-6,np.ones(self.nm,dtype=np.float64)*self.E*1e6)
		self.disp = d*1e3
		self.stress = s*1e-6
		w[:,0] = self.stress/np.abs(self.stress).max()
			
		return w

	def run(self,prt=False,illustrate=False):

		# self.render(name=0)

		eig_vals, eig_vecs = Bif.LinearBucklingAnalysis(np.array(self.node*1e-3,dtype=np.float64),self.connectivity,self.support,np.array(self.load,dtype=np.float64),np.array(self.section*1e-6,dtype=np.float64),np.ones(self.nm,dtype=np.float64)*self.E*1e6)
		positive_eig_vals = eig_vals[eig_vals>1.0e-3]
		linear_buckling_load_factor = np.min(positive_eig_vals) # linear bucling load factor (not considering geometric nonlinearity)
		primary_mode = eig_vecs[np.where(eig_vals==linear_buckling_load_factor)[0][0]]

		lambda_history, disp = Bif.ElasticBucklingAnalysis(np.array(self.node*1e-3,dtype=np.float64),self.connectivity,self.support,np.array(self.load,dtype=np.float64),np.array(self.section*1e-6,dtype=np.float64),np.ones(self.nm,dtype=np.float64)*self.E*1e6,linear_buckling_load_factor)
		elastic_buckling_load_factor = np.max(lambda_history) # elastic buckling load factor (considering geometric nonlinearity)
		self.target = elastic_buckling_load_factor/linear_buckling_load_factor
		if prt:
			print("lin: {0:7.2f}, ela: {1:7.2f}, alpha: {2:1.4f}".format(linear_buckling_load_factor,elastic_buckling_load_factor,self.target))

		self.v = self.update_node_v(primary_mode)
		self.w = self.update_edge_w()
		
		# self.render(show=True)
		scale = 1000.0
		if illustrate:
			self.render_buckling_shape(self.node,self.node+primary_mode*linear_buckling_load_factor*scale,color=(0.5,0.0,0.0,1.0),name="init",show=True)
			self.render_buckling_shape(self.node,[self.node+primary_mode/np.max(primary_mode)*scale,self.node+disp/np.max(disp)*scale],color=[(0.8,0.0,0.0,1.0),(0.0,0.0,0.8,1.0)],name="two",show=True)

		return np.copy(self.v), np.copy(self.w), self.target # Use np.copy if self.target is an numpy array

	def render_buckling_shape(self,node_init,node_deformed,color,name=None,show=False):

		if not isinstance(color,(list,array)):
			color = [color]
			node_deformed = [node_deformed]
		rs = 4
		nsize = np.tile([rs for i in range(self.nk)],1+len(color))
		nshape = ['o' for i in range(self.nk)]
		for sn in self.pin_nodes:
			nshape[sn] = '^'
		nshape = np.tile(nshape,1+len(color))
		ncolor = np.concatenate(([(0.5,0.5,0.5,0.6) for i in range(self.nk)],*[[color[j] for i in range(self.nk)] for j in range(len(color))]))
		nzorder = np.concatenate(([1000 for i in range(self.nk)],*[[2000 for i in range(self.nk)] for j in range(len(color))]))
		nd = np.concatenate((node_init,*node_deformed))
		ct = np.concatenate((self.connectivity,*[self.connectivity+self.nk*(i+1) for i in range(len(color))]))
		ecolor = np.concatenate(([(0.5,0.5,0.5,0.6) for i in range(self.nm)],*[[color[j] for i in range(self.nm)] for j in range(len(color))]))
		esize = np.concatenate(([0.5 for i in range(self.nm)],*[[1 for i in range(self.nm)] for j in range(len(color))]))
		ezorder = np.concatenate(([1001 for i in range(self.nk)],*[[2001 for i in range(self.nk)] for j in range(len(color))]))

		outfile = Plotter.Draw(nd,ct,node_color=ncolor,node_size=nsize,node_shape=nshape,node_zorder=nzorder,edge_color=ecolor,edge_size=esize,edge_zorder=ezorder,name=name,show=show)
	
	def render(self, name=None, show=False):
		nsize = [10 for i in range(self.nk)]
		nshape = ['o' for i in range(self.nk)]
		ncolor = [(1.0,1.0,1.0) for i in range(self.nk)]

		if name == None:
			name = 0

		for sn in self.pin_nodes:
			nshape[sn] = '^'
			nsize[sn] = 15

		outfile = Plotter.Draw(self.node,self.connectivity,node_color=ncolor,node_size=nsize,node_shape=nshape,name=name,show=show)

		return outfile
