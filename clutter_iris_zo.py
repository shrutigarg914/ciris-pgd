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
    PointCloud, RandomGenerator, SceneGraphCollisionChecker,
    RobotDiagramBuilder, MeshcatVisualizerParams,
    IrisZo, HPolyhedron, IrisZoOptions,
    AffineBall
)
import numpy as np
# from pydrake.geometry.optimization_dev import (CspaceFreePolytope, SeparatingPlaneOrder)
from iris_plant_visualizer import IrisPlantVisualizer
from pydrake.geometry import Role
from pydrake.geometry.optimization import IrisOptions, HPolyhedron, Hyperellipsoid, LoadIrisRegionsYamlFile, SaveIrisRegionsYamlFile
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

# import logging
# dk_log = logging.getLogger("drake")
# dk_log.setLevel(logging.DEBUG)
# dk_log.getChild("Snopt").setLevel(logging.INFO)

regions_dict = LoadIrisRegionsYamlFile(f"/home/sgrg/rlg/SUPERUROP/ciris/1121/connected_regions_new.yaml")
print('All regions have been loaded.')
regions = list(regions_dict.values())

def examine_region(region):
    q_star = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    meshcat.AddButton("Stop")
    meshcat.AddButton("Use")
    meshcat.AddButton("Discard")
    use_clicks, discard_clicks = 0, 0
    rng = RandomGenerator(42)
    Rs = region.UniformSample(rng)
    q = Ratfk.ComputeQValue(Rs, q_star)
    # visualize it
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q)
    diagram.ForcedPublish(context)
    useful_qs = []

    while meshcat.GetButtonClicks("Stop") < 1:
        # compute q
        # if we hit use as start point, save as start q.
        if meshcat.GetButtonClicks("Use") > use_clicks:
            start_q = q
            print(start_q)
            start_s = Rs
            useful_qs.append(start_q)
            use_clicks = meshcat.GetButtonClicks("Use")
        elif meshcat.GetButtonClicks("Discard") > discard_clicks:
            # sample a point from lbin
            Rs = region.UniformSample(rng, Rs)
            q = Ratfk.ComputeQValue(Rs, q_star)
            # visualize it
            plant_context = plant.GetMyContextFromRoot(context)
            plant.SetPositions(plant_context, q)
            diagram.ForcedPublish(context)
            discard_clicks = meshcat.GetButtonClicks("Discard")
        time.sleep(0.01)

def visualise_IRIS(regions, plant, plant_context, seed=42, num_sample=10000, colors=None, tcspace=False):       
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

        sample = region.UniformSample(rng)
        if tcspace:
            q_sample = Ratfk.ComputeQValue(sample, q_star)
        else:
            q_sample = sample

        plant.SetPositions(plant_context, q_sample)
        xyzs.append(plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=ee_frame).translation())

        for _ in range(num_sample-1):
            prev_sample = sample
            sample = region.UniformSample(rng, prev_sample)
            if tcspace:
                q_sample = Ratfk.ComputeQValue(sample, q_star)
            else:
                q_sample = sample

            plant.SetPositions(plant_context, q_sample)
            xyzs.append(plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=ee_frame).translation())

        # Create pointcloud from sampled point in IRIS region in order to plot in Meshcat
        xyzs = np.array(xyzs)
        pc = PointCloud(len(xyzs))
        pc.mutable_xyzs()[:] = xyzs.T
        meshcat.SetObject(f"regions/region {i}", pc, point_size=0.025, rgba=colors[i % len(colors)])
    
meshcat = StartMeshcat()

#construct our robot
builder = RobotDiagramBuilder()
builder.parser().package_map().Add("ciris_pgd", os.path.abspath(''))

directives_file = "/home/sgrg/rlg/SUPERUROP/ciris/models/clutter_ciris_scenario.dmd.yaml"
directives = LoadModelDirectives(directives_file)
ProcessModelDirectives(directives, builder.plant(), builder.parser())

meshcat_visual_params = MeshcatVisualizerParams()
meshcat_visual_params.delete_on_initialization_event = False
meshcat_visual_params.role = Role.kIllustration
meshcat_visual_params.prefix = "visual"
meshcat_visual_params.visible_by_default = True
meshcat_visual = MeshcatVisualizer.AddToBuilder(
    builder.builder(), builder.scene_graph(), meshcat, meshcat_visual_params)
meshcat_collision_params = MeshcatVisualizerParams()
meshcat_collision_params.delete_on_initialization_event = False
meshcat_collision_params.role = Role.kProximity
meshcat_collision_params.prefix = "collision"
meshcat_collision_params.visible_by_default = False
meshcat_collision = MeshcatVisualizer.AddToBuilder(
    builder.builder(), builder.scene_graph(), meshcat, meshcat_collision_params)

diagram = builder.Build()

q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
plant = diagram.plant()
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetPositions(plant_context, q0)
diagram.ForcedPublish(context)

Ratfk = RationalForwardKinematics(plant)
q_star = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
q_low = np.array([-2.967060,-2.094395,-2.967060,-2.094395,-2.967060,-2.094395,-3.054326])
tc_low = Ratfk.ComputeSValue(q_low, q_star)
q_high = np.array([2.967060,2.094395,2.967060,2.094395,2.967060,2.094395,3.054326])
tc_high = Ratfk.ComputeSValue(q_high, q_star)
joint_limits = HPolyhedron.MakeBox(tc_low, tc_high)

model = diagram
robot_model_instances = [diagram.plant().GetModelInstanceByName("iiwa")]
edge_step_size = 0.01
collision_checker = SceneGraphCollisionChecker(model=model, robot_model_instances=robot_model_instances, edge_step_size=edge_step_size)

# def grow_region(start_polytope):
# start_polytope = AffineBall.MinimumVolumeCircumscribedEllipsoid([q0])
options = IrisZoOptions.CreateWithArctangentParametrization()
options.require_sample_point_is_contained = True
options.max_iterations = 1
# breakpoint()

# collecting seeds
meshcat.AddButton("Stop")
meshcat.AddButton("Plot Connectivity")
meshcat.AddButton("Grow IRIS Region")
meshcat.AddButton("Lower Bound")
meshcat.AddButton("Pdb")
num_clicks_iris, num_clicks_connectivity, num_clicks_pdb, num_clicks_lb = 0, 0, 0, 0
lb = None
print("we're set up kind of")

# visualise_IRIS(regions, plant, plant_context, tcspace=True)

idx = 0
for joint_index in plant.GetJointIndices():
    joint = plant.get_mutable_joint(joint_index)
    if isinstance(joint, RevoluteJoint):
        meshcat.AddSlider(joint.name(), value=0.0, min=q_low[idx]+0.01, max=q_high[idx]-0.01, step=0.01)
        idx += 1
new_dict = LoadIrisRegionsYamlFile("/home/sgrg/rlg/SUPERUROP/ciris/0205/simple_regions.yaml")
new_regions = list(new_dict.values())
region = new_dict['top_1']
q = q0
eps = 0.001
grow = True
while meshcat.GetButtonClicks("Stop") < 1:
    # breakpoint()
    for i in range(len(q)):
        q[i] = meshcat.GetSliderValue(f"iiwa_joint_{i+1}")
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q)
    diagram.ForcedPublish(context)
    if not collision_checker.CheckConfigCollisionFree(q):
        print("We're in Collision! Can't seed")
        grow = False
    else:
        if not grow:
            grow = True
            print("********")

    
    if meshcat.GetButtonClicks("Pdb") > num_clicks_pdb:
        num_clicks_pdb = meshcat.GetButtonClicks("Pdb")
        breakpoint()
    
    if meshcat.GetButtonClicks("Lower Bound") > num_clicks_lb:
        num_clicks_lb = meshcat.GetButtonClicks("Lower Bound")
        lb = [qj for qj in q]
        breakpoint()


    if meshcat.GetButtonClicks("Grow IRIS Region") > num_clicks_iris and grow:
        num_clicks_iris = meshcat.GetButtonClicks("Grow IRIS Region")
        svalue = Ratfk.ComputeSValue(q, q_star)
        if lb is not None:
            slb = Ratfk.ComputeSValue(lb, q_star)
            options.containment_points = np.vstack((slb, svalue)).T
            initial_region = Hyperellipsoid(AffineBall.MakeHypersphere(0.001, (slb + svalue)/2))
            breakpoint()
            lb = None
        else:
            initial_region = Hyperellipsoid(AffineBall.MakeHypersphere(0.01, svalue))
            options.containment_points = None
        
        if region:
            prev_center = region.ChebyshevCenter()
        else:
            prev_center = regions_dict['nominal'].ChebyshevCenter()
        

        Ratfk = RationalForwardKinematics(plant)
        q_star = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_low = np.array([-2.967060, -2.094395,-2.967060,-2.094395,-2.967060,-2.094395,-3.054326])
        tc_low = Ratfk.ComputeSValue(q_low, q_star)
        q_high = np.array([2.967060,2.094395,2.967060,2.094395,2.967060,2.094395,3.054326])
        tc_high = Ratfk.ComputeSValue(q_high, q_star)
        new_high = np.where(prev_center > svalue, (prev_center + svalue)/2, tc_high)
        new_low = np.where(prev_center < svalue, (prev_center + svalue)/2, tc_low)
        joint_limits = HPolyhedron.MakeBox(new_low, new_high)
        # breakpoint()
        
        
        # visualise_IRIS([], plant, plant_context)
        try:
            region = IrisZo(collision_checker, initial_region, joint_limits, options)
        except:
            joint_limits = HPolyhedron.MakeBox(tc_low, tc_high)
            region = IrisZo(collision_checker, initial_region, joint_limits, options)
            print("something weird happened try again")
        # region = grow_region(q)
        visualise_IRIS([region], plant, plant_context, tcspace=True)
        breakpoint()
        # simple_region = region.SimplifyByIncrementalFaceTranslation()
        # breakpoint()
        num_clicks_pdb = meshcat.GetButtonClicks("Pdb")
        num_clicks_lb = meshcat.GetButtonClicks("Lower Bound")
        num_clicks_iris = meshcat.GetButtonClicks("Grow IRIS Region")
        print("ready to go again")
        # simple_region = region.SimplifyByIncrementalFaceTranslation()
        # regions.append(region)
        # breakpoint() # Can remove to generate new regions
        # SaveIrisRegionsYamlFile("/home/sgrg/rlg/SUPERUROP/ciris/104/simple_regions.yaml", simple_dict)

    if meshcat.GetButtonClicks("Plot Connectivity") > num_clicks_connectivity:
        num_clicks_connectivity = meshcat.GetButtonClicks("Plot Connectivity")
        if len(regions) > 0:
            notebook_plot_connectivity(regions)    
    time.sleep(0.01)

breakpoint()