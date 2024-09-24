from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
import os
from pydrake.all import (
    LoadModelDirectives, ProcessModelDirectives, RevoluteJoint, 
    RationalForwardKinematics, CspaceFreePolytope, SeparatingPlaneOrder,
    RigidTransform, RotationMatrix, Rgba,
    AffineSubspace, MathematicalProgram, Solve,
    MeshcatVisualizer, StartMeshcat,
    PointCloud, RandomGenerator
)
import numpy as np
# from pydrake.geometry.optimization_dev import (CspaceFreePolytope, SeparatingPlaneOrder)
from iris_plant_visualizer import IrisPlantVisualizer
from pydrake.geometry import Role
from pydrake.geometry.optimization import IrisOptions, HPolyhedron, Hyperellipsoid, IrisInRationalConfigurationSpace, LoadIrisRegionsYamlFile, SaveIrisRegionsYamlFile
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions, ScsSolver
import time
from pydrake.all import ModelVisualizer

def visualise_IRIS(regions, plant, plant_context, seed=42, num_sample=10000, colors=None):       
    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("iiwa_frame_ee")

    rng = RandomGenerator(seed)

    # Allow caller to input custom colors
    if colors is None:
        colors = [
                    Rgba(0.5,0.0,0.0,0.5),
                    Rgba(0.0,0.5,0.0,0.5),
                    Rgba(0.0,0.0,0.5,0.5),
                    Rgba(0.5,0.5,0.0,0.5),
                    Rgba(0.5,0.0,0.5,0.5),
                    Rgba(0.0,0.5,0.5,0.5),
                    Rgba(0.2,0.2,0.2,0.5),
                    Rgba(0.5,0.2,0.0,0.5),
                    Rgba(0.2,0.5,0.0,0.5),
                    Rgba(0.5,0.0,0.2,0.5),
                    Rgba(0.2,0.0,0.5,0.5),
                    Rgba(0.0,0.5,0.2,0.5),
                    Rgba(0.0,0.2,0.5,0.5),
                ]

    for i in range(len(regions)):
        region = regions[i]

        xyzs = []  # List to hold XYZ positions of configurations in the IRIS region

        q_sample = region.UniformSample(rng)

        plant.SetPositions(plant_context, q_sample)
        xyzs.append(plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=ee_frame).translation())

        for _ in range(num_sample-1):
            prev_sample = q_sample
            q_sample = region.UniformSample(rng, prev_sample)

            plant.SetPositions(plant_context, q_sample)
            xyzs.append(plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=ee_frame).translation())

        # Create pointcloud from sampled point in IRIS region in order to plot in Meshcat
        xyzs = np.array(xyzs)
        pc = PointCloud(len(xyzs))
        pc.mutable_xyzs()[:] = xyzs.T
        meshcat.SetObject(f"{name}/region {i}", pc, point_size=0.025, rgba=colors[i % len(colors)])
    
    #construct our robot
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parser = Parser(plant)

parser.package_map().Add("ciris_pgd", os.path.abspath(''))

directives_file = "/home/sgrg/rlg/SUPERUROP/ciris/models/iiwa14_sphere_collision_complex_scenario.dmd.yaml"
directives = LoadModelDirectives(directives_file)
models = ProcessModelDirectives(directives, plant, parser)
plant.Finalize()
meshcat = StartMeshcat()
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
diagram = builder.Build()
q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetPositions(plant_context, q0)
diagram.ForcedPublish(context)

import shutil

# Define the source and destination directories
destination_dir = '/home/sgrg/rlg/SUPERUROP/ciris/regions_real/'

# os.makedirs(destination_dir, exist_ok=True)
regions_dict = dict()
# Iterate over all files in the source directory
for filename in os.listdir(destination_dir):
    regions_dict.update(LoadIrisRegionsYamlFile(f"/home/sgrg/rlg/SUPERUROP/ciris/regions_real/{filename}"))
    # Check if the file is in the name mapping dictionary
    print(f'File "{filename}" has been uploaded')

print('All files have been processed.')

for name in regions_dict:
    print(name)
    visualise_IRIS([regions_dict[name]], plant, plant_context)
    breakpoint()
#     meshcat.AddButton("Stop Animation", "Escape")
#     while meshcat.GetButtonClicks("Stop Animation") < 1:
#         time.sleep(0.01)
#     meshcat.DeleteButton("Stop Animation")