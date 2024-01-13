import numpy as np
from numba import njit, f8, i4, b1
from numba.types import Tuple
# from scipy.linalg import eigh as sp_eigh
# from scipy.linalg import eig as sp_eig
# from scipy.linalg import solve as sp_solve
# import warnings

CACHE = True
PARALLEL = False

bb = np.array([-1., 0., 0., 1., 0., 0.],dtype=np.float64)

'''
for linear stiffness matrix
'''
bb1 = np.zeros((6,6),dtype=np.float64)
bb1[0,0] = bb1[3,3] = 1
bb1[0,3] = bb1[3,0] = -1

'''
for geometry stiffness matrix (Saka 1991 "Optimum design of geometrically nonlinear space trusses")
'''
bb2 = np.zeros((6,6),dtype=np.float64)
bb2[1,1] = bb2[2,2] = bb2[4,4] = bb2[5,5] = 1
bb2[1,4] = bb2[4,1] = bb2[2,5] = bb2[5,2] = -1

@njit(Tuple((f8[:,:,:],f8[:],f8[:,:]))(f8[:,:],i4[:,:]),cache=CACHE,parallel=PARALLEL)
def TransformationMatrices(node,member):
	'''
	(input)
	node[nn,3]<float>  : Nodal coordinates
	member[nm,3]<int>  : Member connectivity

	(output)
	tt[nm,6,6]<float>  : Transformation matrices (local to global)
	length[nm]<float>  : Member lengths
	tp[nm,nn*3]<float> : Transformation matrices (global to local)
	'''
	nn = np.shape(node)[0]
	nm = np.shape(member)[0]
	dxyz = np.zeros((nm,3),dtype=np.float64)
	length = np.zeros(nm,dtype=np.float64)
	for i in range(nm):
		dxyz[i] = node[member[i,1],:] - node[member[i,0],:]
		length[i] = np.linalg.norm(dxyz[i])
	tt = np.zeros((nm,6,6),dtype=np.float64)
	for i in range(nm):
		tt[i,0:3,0] = dxyz[i]/length[i]
	flag = np.abs(tt[:,0,0]) >= 0.9
	tt[flag,1,1] = 1.0
	tt[~flag,0,1] = 1.0
	for i in range(nm):
		for j in range(3):
			tt[i,j,2] = tt[i,(j+1)%3,0] * tt[i,(j+2)%3,1] - tt[i,(j+2)%3,0] * tt[i,(j+1)%3,1]
		tt[i,:,2] /= np.linalg.norm(tt[i,:,2])
		for j in range(3):
			tt[i,j,1] = tt[i,(j+1)%3,2] * tt[i,(j+2)%3,0] - tt[i,(j+2)%3,2] * tt[i,(j+1)%3,0]
	tt[:,3:,3:] = tt[:,:3,:3]

	tp = np.zeros((nm,nn*3),dtype=np.float64)
	for i in range(nm):
		indices = np.array((3*member[i,0],3*member[i,0]+1,3*member[i,0]+2,3*member[i,1],3*member[i,1]+1,3*member[i,1]+2),dtype=np.int32)
		for j in range(6):
			tp[i, indices[j]] -= tt[i, j, 0]
			tp[i, indices[j]] += tt[i, j, 3]
 
	return tt, length, tp

@njit(f8[:,:](f8[:],b1[:],i4),cache=CACHE,parallel=PARALLEL)
def Reshape_freevec_to_n3(vec,free,nn):
	'''
	(input)
	vec[np.sum(free)]<float> : Vector with respect to DOFs to be converted to a matrix
	free[nn*3]<bool>         : True if the coordinate is released, False if fixed
	nn<int>                  : Number of nodes
	
	(output)
	mat[nn,3]<float>         : Output matrix
	'''
	extended_vec = np.zeros(nn*3,dtype=np.float64)
	extended_vec[free] = vec
	mat = extended_vec.reshape((nn,np.int32(3)))
	return mat

@njit(Tuple((f8[:,:],f8[:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:],f8[:],f8[:],f8[:,:,:]),cache=CACHE,parallel=PARALLEL)
def LinearStiffnessMatrix(node,member,support,A,E,L,tt):
	'''
	(input)
	node[nn,3]<float>   : Nodal coordinates
	member[nm,2]<int>   : Member connectivity
	support[nn,3]<bool> : True if supported, else False
	A[nm]<float>        : Cross-sectional area.
	E[nm]<float>        : Young's modulus.
	L[nm]<float>        : Member lengths.
	tt[nm,6,6]<float>   : transformation matrices.

	(output)
	Kl_free[nn,nn]<float> : Global linear stiffness matrix with respect to DOFs only
	Kl[nn,nn]<float>      : Global linear stiffness matrix
	'''

	## Organize input model
	nn = node.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(support.flatten()) # DOFs are True, otherwise False
	tt = np.ascontiguousarray(tt)

	## Linear element stiffness matrices
	kl_el = np.zeros((nm,6,6),dtype=np.float64)
	for i in range(nm):
		kl_el[i] = np.dot(tt[i],E[i]*A[i]/L[i]*bb1)
		kl_el[i] = np.dot(kl_el[i],tt[i].transpose())

	## Assembling element matrices to the global matrix
	Kl = np.zeros((3*nn,3*nn),np.float64)
	for i in range(nm): # assemble element matrices into one matrix
		Kl[3*member[i,0]:3*member[i,0]+3,3*member[i,0]:3*member[i,0]+3] += kl_el[i,0:3,0:3]
		Kl[3*member[i,0]:3*member[i,0]+3,3*member[i,1]:3*member[i,1]+3] += kl_el[i,0:3,3:6]
		Kl[3*member[i,1]:3*member[i,1]+3,3*member[i,0]:3*member[i,0]+3] += kl_el[i,3:6,0:3]
		Kl[3*member[i,1]:3*member[i,1]+3,3*member[i,1]:3*member[i,1]+3] += kl_el[i,3:6,3:6]

	Kl_free = Kl[free][:,free] # Extract DOFs	

	return Kl_free, Kl

@njit(Tuple((f8[:,:],f8[:],f8[:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:,:],f8[:],f8[:]),cache=CACHE,parallel=PARALLEL)
def LinearStructuralAnalysis(node0,member,support,load,A,E):

	'''
	(input)
	node[nn,3]: Nodal coordinates
	member[nm,2]: Member connectivity
	support[nn,3]: True if supported, else False
	load[nn,3]: Load magnitude. 0 if no load is applied.
	A[nm]: Cross-sectional area.
	E[nm]: Young's modulus.

	(output)
	deformation[nn,3]<float> : Nodal deformations
	stress[nm]<float>        : Member stresses
	reaction[nn,3]<float>    : Reaction forces
	  (note): Only supported coordinates can take a non-zero value. The other coordinates (i.e., DOFs) takes 0.
	'''

	## Organize input model
	nn = node0.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(support.flatten()) # DOFs are True, otherwise False
	pp = load.flatten()[free].astype(np.float64)

	## Transformation matrices (tt) and initial lengths (ll0)
	tt,ll0,tp = TransformationMatrices(node0,member)
	tt = np.ascontiguousarray(tt)
	tp = np.ascontiguousarray(tp)

	## Linear stiffness matrix 
	Kl_free, Kl = LinearStiffnessMatrix(node0,member,support,A,E,ll0,tt)

	## Solve the stiffness equation (Kl_free)(Up) = (pp) to obtain the deformation
	Up = np.linalg.solve(Kl_free,pp) # Compute displacement Up (size:nDOF), Error may occur at this point when numba is not in use

	## Deformation and stresses
	deformation = Reshape_freevec_to_n3(Up,free,nn)
	U = np.zeros(nn*3,dtype=np.float64) # Displacement vector U (size:nn)
	U[free] = Up # Synchronize U to Up
	stress = np.dot(tp,U)*E/ll0 # axial forces
	
	## Reaction forces
	Rp = np.dot(Kl[~free][:,free],Up)
	R = np.zeros(nn*3,dtype=np.float64)
	R[~free] = Rp
	R[~free] -= load.flatten()[~free]
	reaction = R.reshape((nn,3))

	return deformation, stress, reaction

@njit(Tuple((f8[:,:],f8[:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:],f8[:],f8[:,:,:]),cache=CACHE,parallel=PARALLEL)
def GeometryStiffnessMatrix(node,member,support,N,L,tt):
	'''
	(input)
	node[nn,3]<float>   : Nodal coordinates
	member[nm,2]<int>   : Member connectivity
	support[nn,3]<bool> : True if supported, else False
	N[nm]<float>        : Axial forces (positive for tension, negative for compression).
	L[nm]<float>        : Member lengths.
	tt[nm,6,6]<float>   : Transformation matrices.

	(output)
	Kg_free[nn,nn]<float> : Global geometry stiffness matrix with respect to DOFs only
	Kg[nn,nn]<float>      : Global geometry stiffness matrix
	'''
	## Organize input model
	nn = node.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(support.flatten()) # DOFs are True, otherwise False
	tt = np.ascontiguousarray(tt)

	## Geometry element stiffness matrices
	kg_el = np.zeros((nm,6,6),dtype=np.float64)
	for i in range(nm):
		kg_el[i] = np.dot(tt[i],bb2*N[i]/L[i])
		kg_el[i] = np.dot(kg_el[i],tt[i].transpose())

	## Assembling element matrices to the global matrix
	Kg = np.zeros((3*nn,3*nn),np.float64) # geometry stiffness matrix
	for i in range(nm): # assemble element matrices into one matrix
		Kg[3*member[i,0]:3*member[i,0]+3,3*member[i,0]:3*member[i,0]+3] += kg_el[i,0:3,0:3]
		Kg[3*member[i,0]:3*member[i,0]+3,3*member[i,1]:3*member[i,1]+3] += kg_el[i,0:3,3:6]
		Kg[3*member[i,1]:3*member[i,1]+3,3*member[i,0]:3*member[i,0]+3] += kg_el[i,3:6,0:3]
		Kg[3*member[i,1]:3*member[i,1]+3,3*member[i,1]:3*member[i,1]+3] += kg_el[i,3:6,3:6]

	Kg_free = Kg[free][:,free]

	return Kg_free, Kg

@njit(Tuple((f8[:],f8[:,:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:,:],f8[:],f8[:]),cache=CACHE,parallel=PARALLEL)
def LinearBucklingAnalysis(node0,member,support,load,A,E):
	'''
	(input)
	node[nn,3]<float>   : Nodal coordinates
	member[nm,2]<int>   : Member connectivity
	support[nn,3]<bool> : True if supported, else False
	load[nn,3]<float>   : Load magnitude. 0 if no load is applied.
	A[nm]<float>        : Cross-sectional area.
	E[nm]<float>        : Young's modulus.

	(output)
	eig_vals[nDOF]<float>       : Eigen-values (load factors that cause the buckling)
	eig_modes[nDOF,nn,3]<float> : Eigen-modes (buckling modes)
	'''

	## Organize input model
	nn = node0.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(support.flatten()) # DOFs are True, otherwise False
	pp = load.flatten()[free].astype(np.float64)

	## Transformation matrices (tt) and initial lengths (ll0)
	tt,ll0,tp = TransformationMatrices(node0,member)
	tt = np.ascontiguousarray(tt)
	tp = np.ascontiguousarray(tp)

	## Linear stiffness matrix 
	Kl_free, _ = LinearStiffnessMatrix(node0,member,support,A,E,ll0,tt)

	## Solve the stiffness equation (Kl_free)(Up) = (pp) to obtain the deformation
	Up = np.linalg.solve(Kl_free,pp) # Compute displacement Up (size:nDOF), Error may occur at this point when numba is not in use

	## Deformed shape and forces
	U = np.zeros(nn*3,dtype=np.float64) # Displacement vector U (size:nn)
	U[free] = Up # Synchronize U to Up

	'''
	#<CAUTION># The relationship between external loads and internal forces is not linear if computing "force" using the following equations:
	# node = node0 + deformation # Deformed shape
	# _,ll = TransformationMatrices(node,member) # Recompute member lengths
	# N = A*E*(ll-ll0)/ll0 # axial forces (tensile forces are positive, compressive forces are negative)
	'''

	N = np.dot(tp,U)*E*A/ll0 # axial forces (tensile forces are positive, compressive forces are negative)

	## Linear stiffness matrix 
	Kg_free, _ = GeometryStiffnessMatrix(node0,member,support,N,ll0,tt)

	## Solve the eigenvalue problem
	eig_vals_comp, eig_vecs_comp = np.linalg.eig(np.dot(np.ascontiguousarray(-np.linalg.inv(Kg_free)),np.ascontiguousarray(Kl_free)))
	eig_vals = eig_vals_comp.real.astype(np.float64) # Extract real numbers
	eig_vecs = eig_vecs_comp.real.astype(np.float64) # Extract real numbers
	eig_modes = np.empty((len(eig_vals),nn,3),dtype=np.float64)
	for i in range(len(eig_vals)):
		eig_modes[i] = Reshape_freevec_to_n3(eig_vecs[:,i],free,nn)

	return eig_vals, eig_modes

@njit(Tuple((f8[:],f8[:,:]))(f8[:,:],i4[:,:],b1[:,:],f8[:,:],f8[:],f8[:],f8),cache=CACHE,parallel=PARALLEL)
def ElasticBucklingAnalysis(node0,member,fix,load,A,E,linear_buckling_factor):
	'''
	Elastic buckling analysis considering geometric nonlinearity.
	During analysis, nodal loads and displacements are gradually changed using delta.

	(input)
	node0[nn,3]<float> : Nodal locations (x,y coordinates) [mm]
	member[nm,3]<int>  : Member connectivity
	fix[nn,3]<bool>    : True if supported
	load[nn,3]<float>  : Nodal loads [N]
	A[nm]<float>       : Cross-sectional areas [mm^2]
	E[nm]<float>       : Elastic modulus [N/mm^2]

	(output)
	lambda_history[nstep]<float>      : Load intensity factors
	# node_history[nstep,nn,3]<float> : History of nodal locations
	'''
	## Organize input model
	nn = node0.shape[0] # number of nodes
	nm = member.shape[0] # number of members
	free = np.logical_not(fix.flatten()) # DOFs are True, otherwise False
	pp = load.flatten()[free].astype(np.float64)

	## Transformation matrices (tt) and initial lengths (ll0)
	tt,ll0,_ = TransformationMatrices(node0,member)
	tt = np.ascontiguousarray(tt)

	## Linear stiffness matrix 
	Kl_free, _ = LinearStiffnessMatrix(node0,member,fix,A,E,ll0,tt)
	Kt_free = np.copy(Kl_free) # tangent stiffness matrix = linear stiffness matrix at the beginning (geometry stiffness matrix is a zero-matrix at this point)

	## Solve the stiffness equation (Kl_free)(Up) = (pp) to obtain the deformation
	Up = np.linalg.solve(Kl_free,pp) # Compute displacement Up (size:nDOF), Error may occur at this point when numba is not in use
	# Up = sp_solve(Kl_free,pp,assume_a ='sym',check_finite=False) # Using this may improve the precision when numba is not in use
	U = np.zeros(nn*3,dtype=np.float64) # Displacement vector U (size:nn)
	U[free] = Up # Synchronize U to Up

	### Newton-Raphson
	eps = 1.e-6 # residual norm tolerance
	maxitr = 10000
	delta = linear_buckling_factor/500 # arc length parameter to adjust lamda (load amplifier) and U (nodal displacements)
	gamma = 1. # gamma: relative scaling parameter of stress increment to displacement increment (larger gamma -> disregard displacement)

	#node_history = [np.copy(node0)]
	lambda_history = np.zeros(1,dtype=np.float64)
	lamda = 0. # load amplifier
	done = False
	
	while not done: # lamda is iteratively updated to consider geometric non-linearlity (nodal location changes every step)
		## predictor (nodal location is modified)
		dlamda = delta/np.sqrt((np.dot(Up,Up)+gamma)) # lambda increment # Originally, a weight matrix is inserted between np.inner(Up,Up), which is ignored here
		dU = dlamda*Up # displacement increment
		U[free] += dU # Update nodal displacements
		lamda += dlamda # Update load amplifier
		node = node0 + U.reshape((nn,3)) # update nodal locations
		# node_history[step+1],append(np.copy(node))
		residual = np.dot(Kt_free,dU) - lamda*pp # residual = error between internal forces and external forces
		## corrector (Obtaining equilibrium state)
		'''
		delta_X: modification of the increment dX.
		Thanks to Chan's method, the values of dU and dlamda need not be preserved in the following iterations.
		'''
		done = True
		for j in range(maxitr): # Iterative computation to obtain eqilibrium state (0 residual)
			residual_norm = np.linalg.norm(residual)
			if np.isnan(residual_norm) or residual_norm > 1e12:
				# warnings.warn('Residual includes nan values at the last iteration.')
				break
			elif residual_norm <= eps:
				# print('step %i, lamda = %.5e, |residual| = %.5e' % (step,lamda,residual_norm))
				done = False
				lambda_history = np.append(lambda_history,lamda)
				disp_lasteq = U.reshape((nn,3))
				break
			else:
				delta_Ue = -np.linalg.solve(Kt_free,residual)
				delta_lamda = -np.dot(Up,delta_Ue)/np.dot(Up,Up) # Chan's method (TFC Chan, 1982)
				delta_U = delta_Ue + delta_lamda*Up
				U[free] += delta_U
				lamda += delta_lamda
				node = node0 + U.reshape((nn,3))

				tt,ll,tp = TransformationMatrices(node,member)
				tp = np.ascontiguousarray(tp)
				N = A*E/ll0*(ll-ll0) # axial forces
				ff = np.dot(tp.T,N) # nodal force vector equivalent to the internal force caused by specified displacements
				residual = ff[free] - lamda*pp # residual = error between internal forces and external forces

		'''
		If uncomment below   : Newton method (update tangent stiffness every step)
		If comment out below : Modified Newton method (use only initial stiffness)
		'''
		node = node0 + U.reshape((nn,3))
		tt,ll,tp = TransformationMatrices(node,member)
		N = A*E/ll0*(ll-ll0)
		Kg_free, _ = GeometryStiffnessMatrix(node0,member,fix,N,ll,tt)
		Kt_free = Kl_free+Kg_free
		Up = np.linalg.solve(Kt_free,pp) # Compute displacement vector based on updated tangent stiffness matrix

	return lambda_history, disp_lasteq


# model definition (2D truss)
# node0 = np.array([[-1000.,   0., 0. ], # initial nodal locations
#                  [    0., 100., 0. ],
#                  [ 1000.,   0., 0. ]])
# member = np.array([[0,1],[1,2]]) # connectivity

# load = np.zeros([node0.shape[0],3]) # nodal loads
# load[1,1] = -100.0 # apply -100 load at node 1 in y direction

# fix = np.zeros([node0.shape[0],3],dtype=bool) # support condition
# fix[0] = True
# fix[1,2] = True
# fix[2] = True



## model definition (3D truss)
# node0 = np.array([[-1000,0,0], # initial nodal locations
#                  [ 0,1000,0],
#                  [ 1000,0,0],
#                  [ 0,-1000,0],
#                  [ 0,0,100]],dtype=float)
# member = np.array([[0,4],[1,4],[2,4],[3,4]]) # connectivity

# load = np.zeros([node0.shape[0],3]) # nodal loads
# load[4,2] = -100.0 # apply -100 load at node 4 in z direction

# fix = np.zeros([node0.shape[0],3],dtype=bool) # support condition
# fix[0:4] = True
# fix[4,0:2] = True



# A = np.ones(member.shape[0])*100
# E = np.ones(member.shape[0])*2.0e5

# lambda_history = ElasticBucklingAnalysis(node0,member,fix,load,A,E)
# print(np.array(lambda_history,dtype=np.float16))
# a,b = LinearBucklingAnalysis(node0,member,fix,load,A,E)
# print(a)

# import plot
# plot.plot(node_history[:,:,1:3],member,skip=100)

