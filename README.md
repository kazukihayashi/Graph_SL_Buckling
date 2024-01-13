# Graph_SL_Buckling
Graph-based supervised learning for the prediction of knockdown factors for elastic buckling loads of lattice shells

## main.py
Main program to run the code.

## Agent.py
Graph embedding and machine learning classes

## Bezier.py
A program to obtain the coordinates of a point on a Bezier surface. The order of the Bezier surface is 3×3; In other words, the number of control points is 4×4. Among them, various shapes are generated by randomly changing the four points in the center.

## Environment.py
An intermediary program when generating supervison data and training the machine learning model.

## Plotter.py
Drawing program.

## StructuralAnalysis.py
Self-made linear and elastic buckling analysis program.
The numba module is used to make the program run tens of times faster.

## TrussEnv.py
A program that determines the initial shape of the lattice shell and compute graph information such as node attributes V, edge attributes W for graph embedding.
