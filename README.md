# 2D-beam-finite-element-mesh-analysis
1. Project Introduction
The script performs a linear elastic analysis on a structural body defined by nodes and elements. It follows the standard FEA pipeline: importing geometric and material data, constructing element stiffness matrices using numerical integration, assembling a global system of equations, and solving for nodal displacements.

The code is structured to handle either Plane Stress (suitable for thin plates) or Plane Strain (suitable for thick bodies) conditions, making it a versatile tool for basic 2D structural engineering problems.

2. Technical Project Details
A. Data Input & Management

The tool automates the import of structural data, typically exported from CAD/modeling software like Blender. It looks for four specific CSV files:

Vertices.csv: Global coordinates (x,y) for every node.

Elements.csv: Connectivity data defining which nodes form each quadrilateral element.

Restraint-Nodes & Restraint-DoF: Definitions for boundary conditions (fixed or pinned supports).

Force-Data: Locations and indices for applied external loads.

B. Material & Geometric Properties

The solver uses the following default mechanical properties, which can be manually adjusted:

Young’s Modulus (E): 200×10 
9
  Pa (Standard for Steel).

Poisson’s Ratio (ν): 0.3.

Thickness (t): 0.1 m.

Material Matrix (C): Automatically calculated based on the selected plane stress/strain state.

C. Numerical Integration (Gauss Quadrature)

To calculate the stiffness of the quadrilateral elements, the script employs a 2×2 Gauss Integration scheme.

It defines sampling points at ±0.57735 and corresponding weights.

It utilizes shape function derivatives to calculate the Jacobian matrix (J), which maps the local coordinate system to the global physical space.

D. System Assembly and Solution

Stiffness Assembly: Individual 8×8 element matrices are mapped to a global Kp matrix.

Boundary Conditions: The script "prunes" the global matrix by removing rows and columns associated with restrained degrees of freedom (DoF) to produce the reduced structure stiffness matrix (Ks).

Linear Solver: It solves the system U=Ks 
−1
 ⋅F to find unknown nodal displacements.

3. Visual Outputs
The project includes two primary visualization stages using matplotlib:

Initial Mesh Plot: Displays the undeformed structure, showing element boundaries and node locations to verify the mesh geometry.

Deflected Shape Plot: Overlays the original mesh with the deformed shape (scaled by a deformation factor, dFac) to show how the beam reacts under the assigned point loads.
