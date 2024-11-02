from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
import os
from pydrake.all import (
    LoadModelDirectives, ProcessModelDirectives, RevoluteJoint, 
    RationalForwardKinematics, CspaceFreePolytope, SeparatingPlaneOrder,
    RigidTransform, RotationMatrix, Rgba,
    AffineSubspace, MathematicalProgram, Solve,
    MeshcatVisualizer, StartMeshcat, MeshcatVisualizerParams,
    PointCloud, RandomGenerator, InverseKinematics
)
import numpy as np
# from pydrake.geometry.optimization_dev import (CspaceFreePolytope, SeparatingPlaneOrder)
from pydrake.geometry import Role
from pydrake.geometry.optimization import IrisOptions, HPolyhedron, Hyperellipsoid, IrisInRationalConfigurationSpace, LoadIrisRegionsYamlFile, SaveIrisRegionsYamlFile
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions, ScsSolver
import time
from pydrake.all import ModelVisualizer
from util import notebook_plot_connectivity

solver_options = SolverOptions()
# set this to 1 if you would like to see the solver output in terminal.
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 0)

os.environ["MOSEKLM_LICENSE_FILE"] = "/home/sgrg/mosek.lic"
with open(os.environ["MOSEKLM_LICENSE_FILE"], 'r') as f:
    contents = f.read()
    mosek_file_not_empty = contents != ''
print(mosek_file_not_empty)

solver_id = MosekSolver.id() if MosekSolver().available() and mosek_file_not_empty else ScsSolver.id()

import logging
dk_log = logging.getLogger("drake")
dk_log.setLevel(logging.DEBUG)
dk_log.getChild("Snopt").setLevel(logging.INFO)


# load all regions from folder

import os
from pydrake.all import LoadIrisRegionsYamlFile, RandomGenerator
regions_folder = '/home/sgrg/rlg/SUPERUROP/ciris/regions_bins/'

# os.makedirs(destination_dir, exist_ok=True)
regions_dict = dict()
# Iterate over all files in the regions directory
for filename in os.listdir(regions_folder):
    regions_dict.update(LoadIrisRegionsYamlFile(f"/home/sgrg/rlg/SUPERUROP/ciris/regions_bins/{filename}"))
    print(f'Region "{filename}" has been loaded')

print('All regions have been loaded.')
regions = list(regions_dict.values())
breakpoint()

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

        rq_sample = region.UniformSample(rng)
        q_sample = Ratfk.ComputeQValue(rq_sample, q_star)

        plant.SetPositions(plant_context, q_sample)
        xyzs.append(plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=ee_frame).translation())

        for _ in range(num_sample-1):
            prev_sample = rq_sample
            rq_sample = region.UniformSample(rng, prev_sample)
            q_sample = Ratfk.ComputeQValue(rq_sample, q_star)

            plant.SetPositions(plant_context, q_sample)
            xyzs.append(plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=ee_frame).translation())

        # Create pointcloud from sampled point in IRIS region in order to plot in Meshcat
        xyzs = np.array(xyzs)
        pc = PointCloud(len(xyzs))
        pc.mutable_xyzs()[:] = xyzs.T
        meshcat.SetObject(f"regions/region {i}", pc, point_size=0.025, rgba=colors[i % len(colors)])

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
diagram.ForcedPublish(context)
plant_context = plant.GetMyContextFromRoot(context)
plant.SetPositions(plant_context, q0)
q_low = np.array([-2.967060,-2.094395,-2.967060,-2.094395,-2.967060,-2.094395,-3.054326])
q_high = np.array([2.967060,2.094395,2.967060,2.094395,2.967060,2.094395,3.054326])
idx = 0
for joint_index in plant.GetJointIndices():
    joint = plant.get_mutable_joint(joint_index)
    if isinstance(joint, RevoluteJoint):
        joint.set_default_angle(q0[idx])
        joint.set_position_limits(lower_limits= np.array([q_low[idx]]), upper_limits= np.array([q_high[idx]]))
        print(joint)
        idx += 1 
Ratfk = RationalForwardKinematics(plant)
# the point about which we will take the stereographic projections
q_star = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
# The object we will use to perform our certification.
cspace_free_polytope = CspaceFreePolytope(plant, scene_graph, SeparatingPlaneOrder.kAffine, q_star)
regions = [regions_dict['left_bin'], regions_dict['middle_bin_jank']]
visualise_IRIS(regions, plant, plant_context)

breakpoint()

# build a list of randomly sampled points in these regions
# number_sample_pairs = 1
rng = RandomGenerator(42)
intersection = regions[0].Intersection(regions[1])
start = regions[0].UniformSample(rng)
if np.all(np.less_equal(intersection.A() @ start, intersection.b())):
    start = regions[0].UniformSample(rng, start)
goal = regions[1].UniformSample(rng)
if np.all(np.less_equal(intersection.A() @ goal, intersection.b())):
    goal = regions[1].UniformSample(rng, goal)
sample_start_end = (start, goal)

# find a trajectory from one to another.
from pgd import *
continuity = 1

gcs = GcsTrajectoryOptimization(7)
if continuity > 0:
    gcs.AddPathContinuityConstraints(continuity)
main_graph = gcs.AddRegions(regions, 3, h_min=0.1, h_max=100, name="")
start_graph = gcs.AddRegions([Point(start)], 0)
goal_graph = gcs.AddRegions([Point(goal)], 0)
gcs.AddEdges(start_graph, main_graph)
gcs.AddEdges(main_graph, goal_graph)

gcs.AddPathLengthCost()

options = GraphOfConvexSetsOptions()
options.max_rounding_trials = 1000
options.max_rounded_paths = 100
options.convex_relaxation = True

traj, result = gcs.SolvePath(start_graph, goal_graph, options)
if not result.is_success:
    print("ERROR NOT SUCCESS")
    breakpoint()

# gcs, result = generate_flows(start, goal, 3, regions, dim=7)
from util import plot_traj_end_effector_path
plot_traj_end_effector_path(traj, meshcat, InverseKinematics(plant, plant_context), name="path", color_rgba=(1, 0, 0, 1))
breakpoint()
# visualize superimposed trajectories + save metric
