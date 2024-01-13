import numpy as np 
import colorsys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import art3d

def bernstein(t, n, i):
    if i == 0 or i > n:
        return 0
    cn, ci, cni = 1.0, 1.0, 1.0
    for k in range(2, n, 1):
        cn = cn * k
    for k in range(1, i, 1):
        if i == 1:
            break
        ci = ci * k
    for k in range(1, n - i + 1, 1):
        if n == i:
            break
        cni = cni * k
    j = t**(i - 1) * (1 - t)**(n - i) * cn / (ci * cni)
    return j


def d_bern(t, n, i):  
    cn, ci, cni = 1.0, 1.0, 1.0
    for k in range(2, n, 1):
        cn = cn * k
    for k in range(1, i, 1):
        if i == 1:
            break
        ci = ci * k
    for k in range(1, n - i + 1, 1):
        if n == i:
            break
        cni = cni * k
    j = t**(i - 2) * (1 - t)**(n - i - 1) * cn * \
        ((1 - n) * t + i - 1) / (ci * cni)
    return j

def bezierplot(uv,cp):
    '''
    uv[n_nodes,2]
    '''
    nodes = np.zeros([uv.shape[0], 3])
    for k in range(uv.shape[0]):
        for i in range(cp.shape[0]):
            bu = bernstein(uv[k,0], cp.shape[0], i+1)
            for j in range(cp.shape[1]):
                bv = bernstein(uv[k,1], cp.shape[1], j+1)
                nodes[k,:] += cp[i, j, :] * bu * bv
    return nodes

def bezierplot_grad(node_number, cp, var):
	xyz_g = np.zeros([3, len(var)])
	for i in range(cp.shape[0]):
		bu = bernstein(var[2*node_number], cp.shape[0], i+1) # var[2*ii] = u
		bu_g = cp.shape[0] * (bernstein(var[2*node_number], cp.shape[0]-1, i) - bernstein(var[2*node_number], cp.shape[0]-1, i+1))
		for j in range(cp.shape[1]):
			bv = bernstein(var[2*node_number+1], cp.shape[1], j+1) # var[2*ii+1] = v
			bv_g = cp.shape[1] * (bernstein(var[2*node_number+1], cp.shape[1]-1, j) - bernstein(var[2*node_number+1], cp.shape[1]-1, j+1))
			xyz_g[:,2*node_number] += cp[i,j,:] * bu_g * bv
			xyz_g[:,2*node_number+1] += cp[i,j,:] * bu * bv_g
	return xyz_g

def EGF(u, v, cp):
    z1, z2 = np.zeros(3), np.zeros(3)
    for i in range(cp.shape[0]):
        bu, dbu = bernstein(u, cp.shape[0], i+1), d_bern(u, cp.shape[0], i+1)
        for j in range(cp.shape[1]):
            bv, dbv = bernstein(v, cp.shape[1], j+1), d_bern(v, cp.shape[1], j+1)
            z1 += cp[i, j, :] * dbu * bv
            z2 += cp[i, j, :] * bu * dbv
    E, G, F = z1.dot(z1), z2.dot(z2), z1.dot(z2)
    return (abs(E * G - F**2))**0.5

def orthogonal_transformation(zfront, zback):
    a, b, c = 2 / (zfront - zback), -1 * (zfront + zback) / \
        (zfront - zback), zback
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, a, b], [0, 0, 0, c]])

def triangle_hue(cluster):
	# (cluster with lower number) red-orange-yellow-green-cyan-blue (cluster with higher number)
    n_cluster = max(cluster) + 1
    unit_hue = 0.7 / n_cluster
    hue = np.array(cluster).astype(float)
    for i in range(len(cluster)):
        hue[i] = unit_hue * hue[i]
    return hue

def plot_shape(cp, x, connectivity, cluster, limit): 
    proj3d.persp_transformation = orthogonal_transformation
    #surface points
    u, v = np.arange(0, 1 + 0.1, 0.1), np.arange(0, 1 + 0.1, 0.1)  
    s = bezierplot(u, v, cp)
    #triangular grid points
    gx = []
    gy = []
    gz = []
    for i in range(0,len(x),2):
        p = bezierplot([x[i]], [x[i+1]], cp)
        gx.append(p[0,0,0])
        gy.append(p[0,0,1])
        gz.append(p[0,0,2])
    #plotting figure
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_xlim(limit[0], limit[1])
    ax.set_ylim(limit[2], limit[3])
    ax.set_zlim(limit[4], limit[5])
    #surface
    #ax.plot_surface(s[:, :, 0], s[:, :, 1], s[:, :, 2], rstride=1, cstride=1, color='#cccccc') 
    ## trianglar meshes & plot
    n_d = []
    hue = triangle_hue(cluster)
    for i in range(len(connectivity)):
        n1 = connectivity[i][0][0]
        n2 = connectivity[i][0][1]
        n3 = connectivity[i][1][1]
        p1 = bezierplot([x[2*n1]], [x[2*n1+1]], cp)
        p2 = bezierplot([x[2*n2]], [x[2*n2+1]], cp)
        p3 = bezierplot([x[2*n3]], [x[2*n3+1]], cp)
        vtx = []
        vtx.append(p1[0,0])
        vtx.append(p2[0,0])
        vtx.append(p3[0,0])
        tri = art3d.Poly3DCollection([vtx])
        tri.set_color(colorsys.hsv_to_rgb(hue[i],1.0,1.0))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    #nodal points
    #ax.scatter3D(gx, gy, gz, c='#777777', s=25)
    plt.savefig("model_axonometric.png")
    ax.view_init(elev = 90 + 1.0e-10, azim = -90 + 1.0e-10)
    plt.savefig("model_plan.png")
    plt.show()
