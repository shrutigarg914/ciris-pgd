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
        meshcat.SetObject(f"regions/region {i}", pc, point_size=0.025, rgba=colors[i % len(colors)])

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
visualizer_params = MeshcatVisualizerParams()
visualizer_params.show_hydroelastic = True
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, visualizer_params)
diagram = builder.Build()
q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetPositions(plant_context, q0)
diagram.ForcedPublish(context)

Ratfk = RationalForwardKinematics(plant)

# the point about which we will take the stereographic projections
# q_star = np.zeros(plant.num_positions())
q_star = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
do_viz = True

# The object we will use to perform our certification.
cspace_free_polytope = CspaceFreePolytope(plant, scene_graph, SeparatingPlaneOrder.kAffine, q_star)

q_low = np.array([-2.967060,-2.094395,-2.967060,-2.094395,-2.967060,-2.094395,-3.054326])
q_high = np.array([2.967060,2.094395,2.967060,2.094395,2.967060,2.094395,3.054326])
idx = 0
# slider_names = 
meshcat.DeleteAddedControls()

for joint_index in plant.GetJointIndices():
    joint = plant.get_mutable_joint(joint_index)
    if isinstance(joint, RevoluteJoint):
        joint.set_default_angle(q0[idx])
        joint.set_position_limits(lower_limits= np.array([q_low[idx]]), upper_limits= np.array([q_high[idx]]))
        meshcat.AddSlider(joint.name(), value=0.0, min=q_low[idx], max=q_high[idx], step=0.01)
        idx += 1

iris_options = IrisOptions()
iris_options.require_sample_point_is_contained = True
iris_options.configuration_space_margin = 0.00001
iris_options.relative_termination_threshold = 0.001
iris_options.iteration_limit = 10

def in_collision(plant, scene_graph, context, print_collisions=False, thresh=1e-3):
    plant_context = plant.GetMyContextFromRoot(context)
    sg_context = scene_graph.GetMyContextFromRoot(context)
    query_object = plant.get_geometry_query_input_port().Eval(plant_context)
    inspector = scene_graph.get_query_output_port().Eval(sg_context).inspector()
    pairs = inspector.GetCollisionCandidates()
    dists = []
    for pair in pairs:
        dists.append(query_object.ComputeSignedDistancePairClosestPoints(pair[0], pair[1]).distance)
        if dists[-1] < thresh and print_collisions:
            print(inspector.GetName(inspector.GetFrameId(pair[0])),
                  inspector.GetName(inspector.GetFrameId(pair[1])))
    return np.min(dists) < thresh

def grow_region(seed):
    name = str(seed)    
    t1 = time.time()
    q = Ratfk.ComputeQValue(seed, q_star)
    plant.SetPositions(plant.GetMyMutableContextFromRoot(context), q)
    r = IrisInRationalConfigurationSpace(plant, 
                                         plant.GetMyContextFromRoot(context),
                                         q_star, iris_options)
    t2 = time.time()
    print("Region constructed in ~%d seconds." % int(t2 - t1))
    return r

# collecting seeds
meshcat.AddButton("Stop")
meshcat.AddButton("Plot Connectivity")
meshcat.AddButton("Grow IRIS Region")
meshcat.AddButton("Plot Region")
meshcat.AddButton("Add Seed and Region")
q = q0
regions = []
num_clicks_iris, num_clicks_connectivity = 0, 0
print("Ready to generate regions")

while meshcat.GetButtonClicks("Stop") < 1:
    for i in range(len(q)):
        q[i] = meshcat.GetSliderValue(f"iiwa_joint_{i+1}")
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q)
    diagram.ForcedPublish(context)
    if in_collision(plant, scene_graph, context):
        meshcat.AddButton("We're in Collision! Can't seed")
        grow = False
    else:
        try:
            meshcat.DeleteButton("We're in Collision! Can't seed")
        except:
            pass
        grow = True

    if meshcat.GetButtonClicks("Grow IRIS Region") > num_clicks_iris and grow:
        num_clicks_iris = meshcat.GetButtonClicks("Grow IRIS Region")
        region = grow_region(q)
        visualise_IRIS([region], plant, plant_context)
        regions.append(region)
        breakpoint() # Can remove to generate new regions

    if meshcat.GetButtonClicks("Plot Connectivity") > num_clicks_connectivity:
        num_clicks_connectivity = meshcat.GetButtonClicks("Plot Connectivity")
        if len(regions) > 0:
            notebook_plot_connectivity(regions)    
    time.sleep(0.01)