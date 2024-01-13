import colorsys
from matplotlib import pyplot
from matplotlib.lines import Line2D
import numpy as np

def Draw(node, connectivity, node_color=[], node_vec=[], node_size=[], node_shape=[], node_zorder=[], front_node_index=[], edge_color=[], edge_size=[], edge_zorder=[], edge_annotation=[], name=0, show=False):
	"""
	node[nk,3]or[nk,2]  :(float) nodal coordinates
	connectivity[nm,2]	:(int)   connectivity to define member
	section[nm]			:(float) cross-sectional area of member
	edge_annotation[nm]	:(float) axial stress of member
	"""
	fig = pyplot.figure()
	pyplot.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

	if node.shape[1] == 3:

		def set_axes_equal(ax,Xmin,Xmax,Ymin,Ymax,Zmin,Zmax):

			plot_radius = 0.5*max([Xmax-Xmin, Ymax-Ymin, Zmax-Zmin])

			ax.set_xlim3d([(Xmax+Xmin)/2 - plot_radius, (Xmax+Xmin)/2 + plot_radius])
			ax.set_ylim3d([(Ymax+Ymin)/2 - plot_radius, (Ymax+Ymin)/2 + plot_radius])
			ax.set_zlim3d([(Zmax+Zmin)/2 - plot_radius, (Zmax+Zmin)/2 + plot_radius])

		# make space
		ax = pyplot.axes(projection='3d')
		ax.set_box_aspect((1,1,1))
		set_axes_equal(ax,node[:,0].min(),node[:,0].max(),node[:,1].min(),node[:,1].max(),node[:,2].min(),node[:,2].max())

		# axis label
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")

		if len(node_shape) == 0:
			node_shape = ["o" for i in range(node.shape[0])]
		if len(node_size) == 0:
			node_size = [3 for i in range(node.shape[0])]
		if len(node_color) == 0:
			node_color = [(0.0,0.0,0.0) for i in range(node.shape[0])]
		if len(edge_color) == 0:
			edge_color = [(0.5,0.5,0.5) for i in range(np.shape(connectivity)[0])]
		if len(edge_size) == 0:
			edge_size = [1 for i in range(np.shape(connectivity)[0])]

		objects = []
		# plot node
		for i in range(node.shape[0]):
			objects.extend(ax.plot(node[i,0],node[i,1],node[i,2], node_shape[i], color=node_color[i], ms=node_size[i]))#,zorder=node_zorder[i]))
			if node_shape[i] == "o":
				objects.extend(ax.plot(node[i,0],node[i,1],node[i,2], node_shape[i], color="white", ms=node_size[i]//1.5))#,zorder=edge_zorder[i]))

		# connect member
		for i in range(connectivity.shape[0]):
			x = [node[connectivity[i,0],0],node[connectivity[i,1],0]]
			y = [node[connectivity[i,0],1],node[connectivity[i,1],1]]
			z = [node[connectivity[i,0],2],node[connectivity[i,1],2]]
			objects.extend(ax.plot(x,y,z, linewidth=edge_size[i], color=edge_color[i],zorder=0))

		# # plot front node
		# if front_node_index is not None:
		#     for fni in front_node_index:
		#         ax.plot(node[fni,0],node[fni,1],node[fni,2], node_shape[fni], color=node_color[fni], ms=node_size[fni])

		ax.tick_params(labelbottom="off",bottom="off") # Delete x axis
		ax.tick_params(labelleft="off",left="off") # Delete y axis
		ax.set_xticklabels([]) 
		ax.axis("off") # Delete frame
	
		# view angle
		ax.set_proj_type('ortho')
		ax.view_init(elev=30, azim=225)
		pyplot.axis('off')
		pyplot.savefig(r'result/{0}.png'.format(name),dpi=150,transparent=True)
		if show:
			pyplot.show()
		for obj in objects:
			obj.remove()

		pyplot.close()
	else:
		raise Exception("Only 3D figure is supported.")

	return fig

def graph(y,index=0):
	x = np.linspace(0,len(y),len(y))
	pyplot.figure(figsize=(10,4))
	pyplot.plot(x,y,linewidth=1)

	# save figure
	pyplot.savefig(r"result/graph("+str(index)+").png")
	##pyplot.show()
	pyplot.close()

# node = np.zeros([25,2])
# for i in range(node.shape[0]):
# 	node[i,0] = i%5
# 	node[i,1] = i//5
# connectivity = np.zeros([40,2],dtype=int)
# for i in range(5):
# 	for j in range(4):
# 		connectivity[4*i+j] = [5*i+j,5*i+j+1]
# for i in range(4):
# 	for j in range(5):
# 		connectivity[20+5*i+j] = [5*i+j,5*(i+1)+j]
# section = np.ones(40)
# Draw(node,connectivity,section=section)

# edge_color = np.random.rand(40,3)*1
# Draw(node,connectivity,section=section,edge_color=edge_color)
