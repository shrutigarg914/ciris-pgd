#!/usr/bin/env python
# coding: utf-8


from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
import os
from pydrake.all import (
    LoadModelDirectives, ProcessModelDirectives, RevoluteJoint, 
    RationalForwardKinematics, CspaceFreePolytope, SeparatingPlaneOrder,
    RigidTransform, RotationMatrix
)
import numpy as np
# from pydrake.geometry.optimization_dev import (CspaceFreePolytope, SeparatingPlaneOrder)
from iris_plant_visualizer import IrisPlantVisualizer
from pydrake.geometry import Role


# In[3]:


#construct our robot
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parser = Parser(plant)

parser.package_map().Add("ciris_pgd", os.path.abspath(''))

directives_file = "/home/shrutigarg/ciris/models/bimanual_one_DOF.yaml"
directives = LoadModelDirectives(directives_file)
models = ProcessModelDirectives(directives, plant, parser)


# oneDOF_iiwa_file = "/home/sgrg/rlg/SUPERUROP/ciris/assets/oneDOF_iiwa7_with_box_collision.sdf"
# with open(oneDOF_iiwa_file, 'r') as f:
#     oneDOF_iiwa_string = f.read()
# box_asset_file = "/home/sgrg/rlg/SUPERUROP/ciris/assets/box_small.urdf"
# with open(box_asset_file, 'r') as f:
#     box_asset_string = f.read()


# In[4]:


plant.Finalize()


# In[5]:


idx = 0
q0 = [0.0, 0.0]
val = 1.7
q_low  = np.array([-val, -val])
q_high = np.array([val, val])
# set the joint limits of the plant
# for model in models:
for joint_index in plant.GetJointIndices():
    joint = plant.get_mutable_joint(joint_index)
    if isinstance(joint, RevoluteJoint):
        print("joint index: ", joint_index, "joint name: ", joint.name(), "IDX", idx)
        joint.set_default_angle(q0[idx])
        joint.set_position_limits(lower_limits= np.array([q_low[idx]]), upper_limits= np.array([q_high[idx]]))
        idx += 1


# In[6]:


Ratfk = RationalForwardKinematics(plant)

# the point about which we will take the stereographic projections
q_star = np.zeros(plant.num_positions())

do_viz = True

# The object we will use to perform our certification.
cspace_free_polytope = CspaceFreePolytope(plant, scene_graph, SeparatingPlaneOrder.kAffine, q_star)


# In[ ]:


# This line builds the visualization. Change the viz_role to Role.kIllustration if you
# want to see the plant with its illustrated geometry or to Role.kProximity if you want
# to see the plant with the collision geometries.
visualizer = IrisPlantVisualizer(plant, builder, scene_graph, cspace_free_polytope, viz_role=Role.kProximity)
visualizer.visualize_collision_constraint(factor = 1.2, num_points = 100)
visualizer.meshcat_cspace.Set2dRenderMode(RigidTransform(RotationMatrix.MakeZRotation(0), np.array([0,0,1])))
visualizer.meshcat_task_space.Set2dRenderMode(RigidTransform(RotationMatrix.MakeZRotation(0), np.array([1,0,0])))


# In[ ]:





# In[ ]:




