from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
import os
from pydrake.all import (
    LoadModelDirectives, ProcessModelDirectives, RevoluteJoint, 
    RationalForwardKinematics, CspaceFreePolytope, SeparatingPlaneOrder,
    RigidTransform, RotationMatrix, Rgba,
    AffineSubspace, MathematicalProgram, Solve,
    MeshcatVisualizer, StartMeshcat, InverseKinematics,
    PointCloud, RandomGenerator
)
import numpy as np
# from pydrake.geometry.optimization_dev import (CspaceFreePolytope, SeparatingPlaneOrder)
from iris_plant_visualizer import IrisPlantVisualizer
from pydrake.geometry import Role
from pydrake.geometry.optimization import IrisOptions, HPolyhedron, Hyperellipsoid, IrisInRationalConfigurationSpace, LoadIrisRegionsYamlFile, SaveIrisRegionsYamlFile
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions, ScsSolver
import time
import pickle
import logging
from manipulation import ConfigureParser
from manipulation.scenarios import AddPlanarIiwa, AddShape, AddWsg
from time import perf_counter
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

#construct our robot
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parser = Parser(plant)

parser.package_map().Add("ciris_pgd", os.path.abspath(''))

builder = DiagramBuilder()

plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parser = Parser(plant)
ConfigureParser(parser)
iiwa = AddPlanarIiwa(plant)
wsg = AddWsg(plant, iiwa, roll=0.0, welded=True)

bi = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
plant.WeldFrames(
    plant.world_frame(),
    plant.GetFrameByName("shelves_body", bi),
    RigidTransform([0.8, 0, 0.4]),
    )

plant.Finalize()
meshcat = StartMeshcat()
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
diagram = builder.Build()
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)

q0 = [-0.12, -1.75, 0.32]
gripper_frame = plant.GetFrameByName("body", wsg)

ik = InverseKinematics(plant, plant_context)
collision_constraint = ik.AddMinimumDistanceLowerBoundConstraint(0.001, 0.01)

prog = ik.get_mutable_prog()
q = ik.q()
prog.SetInitialGuess(q, q0)
result = Solve(ik.prog())
if not result.is_success():
    print("IK failed")

diagram.ForcedPublish(context)


q_low = np.array([-3.094395,-3.094395,-3.094395])
q_high = np.array([3.094395,3.094395,3.094395])

idx = 0
for joint_index in plant.GetJointIndices():
    joint = plant.get_mutable_joint(joint_index)
    if isinstance(joint, RevoluteJoint):
        joint.set_default_angle(q0[idx])
        joint.set_position_limits(lower_limits= np.array([q_low[idx]]), upper_limits= np.array([q_high[idx]]))
        idx += 1

Ratfk = RationalForwardKinematics(plant)

# the point about which we will take the stereographic projections
# q_star = np.zeros(plant.num_positions())
q_star = np.array([0.0,0.0,0.0])
do_viz = True

# The object we will use to perform our certification.
cspace_free_polytope = CspaceFreePolytope(plant, scene_graph, SeparatingPlaneOrder.kAffine, q_star)

# set up the certifier and the options for different search techniques
solver_options = SolverOptions()
# set this to 1 if you would like to see the solver output in terminal.
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 0)

# load the generated regions
regions_folder = '/home/sgrg/rlg/SUPERUROP/ciris/111/'

regions_dict = dict()
# Iterate over all files in the regions directory
for filename in os.listdir(regions_folder):
    if filename!="certified_regions_3.yaml":
        continue
    regions_dict.update(LoadIrisRegionsYamlFile(regions_folder+filename))
    print(f'Region "{filename}" has been loaded')
    break

print('All regions have been loaded.')
regions = list(regions_dict.values())
visualise_IRIS(regions, plant, plant_context)
breakpoint()
# # iris_regions = LoadIrisRegionsYamlFile("/home/shrutigarg/drake/ciris-pgd/ComplexScenarioRegions.yaml")
# # Some seedpoints
# list_regions = list(iris_regions.values())
# seed_points_q = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
# # regions_to_sample = [0, 13, 18, 19]
# names_regions = list(iris_regions.keys())
# names_to_save = ["origin"]
# for reg_idx in range(len(list_regions)):
#     if names_regions[reg_idx] == "RightBin-Above":
#         break
#     seed_q = list_regions[reg_idx].MaximumVolumeInscribedEllipsoid().center()
#     print(names_regions[reg_idx])
#     seed_points_q = np.append(seed_points_q, [seed_q], axis=0)
#     names_to_save.append(names_regions[reg_idx])

# seed_points = np.array([Ratfk.ComputeSValue(seed_points_q[idx], q_star)\
#                         for idx in range(seed_points_q.shape[0])])

# # generate C-IRIS regions with these seedpoints
# default_scale = 1e-2
# L1_ball = HPolyhedron.MakeL1Ball(7)
# Linf_ball = HPolyhedron.MakeBox(-np.ones(7), np.ones(7))

# template_C = np.vstack([L1_ball.A(), Linf_ball.A()])
# template_d = np.hstack([default_scale*L1_ball.b(), default_scale/np.sqrt(2)*Linf_ball.b()])


# def make_default_polytope_at_point(seed_point):
#     return HPolyhedron(template_C, template_d + template_C @ seed_point)

# colors to plot the region.
default_alpha = 0.2
colors_dict = {
    0: Rgba(0.565, 0.565, 0.565, default_alpha), # gray
    1: Rgba(0.118, 0.533, 0.898, default_alpha), # bluish
    2: Rgba(1,     0.757, 0.027, default_alpha), # gold
    3: Rgba(0,     0.549, 0.024, default_alpha), # green   
    4: Rgba(0.055, 0.914, 0.929, default_alpha), # teal 
}

# initial_regions = [make_default_polytope_at_point(s) for i, s in enumerate(seed_points)]
# The options for when we search for a new polytope given positivity certificates.
find_polytope_given_lagrangian_option = CspaceFreePolytope.FindPolytopeGivenLagrangianOptions()
find_polytope_given_lagrangian_option.solver_options = solver_options
find_polytope_given_lagrangian_option.ellipsoid_margin_cost = CspaceFreePolytope.EllipsoidMarginCost.kGeometricMean
find_polytope_given_lagrangian_option.search_s_bounds_lagrangians = True
find_polytope_given_lagrangian_option.ellipsoid_margin_epsilon = 1e-4
find_polytope_given_lagrangian_option.solver_id = solver_id

bilinear_alternation_options = CspaceFreePolytope.BilinearAlternationOptions()
bilinear_alternation_options.max_iter = 10 # Setting this to a high number will lead to more fill
bilinear_alternation_options.convergence_tol = 1e-3
bilinear_alternation_options.find_polytope_options = find_polytope_given_lagrangian_option

# The options for when we search for new planes and positivity certificates given the polytopes
# find_separation_certificate_given_polytope_options = CspaceFreePolytope.FindSeparationCertificateGivenPolytopeOptions()
# find_separation_certificate_given_polytope_options.num_threads = -1
# Parallelism 	parallelism {Parallelism::Max()}
bilinear_alternation_options.find_lagrangian_options.verbose = True
bilinear_alternation_options.find_lagrangian_options.solver_options = solver_options
bilinear_alternation_options.find_lagrangian_options.ignore_redundant_C = False
bilinear_alternation_options.find_lagrangian_options.solver_id = solver_id

binary_search_options = CspaceFreePolytope.BinarySearchOptions()
binary_search_options.scale_min = 1e-3
binary_search_options.scale_max = 1.0
binary_search_options.max_iter = 5
binary_search_options.find_lagrangian_options.verbose = True
binary_search_options.find_lagrangian_options.solver_options = solver_options
binary_search_options.find_lagrangian_options.ignore_redundant_C = False
binary_search_options.find_lagrangian_options.solver_id = solver_id
# binary_search_options.find_lagrangian_options = find_separation_certificate_given_polytope_options

binary_search_region_certificates_for_iris = dict.fromkeys([tuple(name) for name in regions_dict.keys()])
# # regions_dummy = [tup[0] for tup in initial_regions]

times_per_region = []
certified_regions = {}
for i, (name, initial_region) in enumerate(zip(regions_dict.keys(), regions)):
    print(f"starting seedpoint {i+1}/{len(regions)}")
    time.sleep(0.2)
    cur_s = initial_region.MaximumVolumeInscribedEllipsoid().center()
    bilinear_alternation_options.find_polytope_options.s_inner_pts = cur_s
    
    # We back off a little bit from the binary search solution so that we don't encounter numerical issues.
    initial_region = initial_region.Scale(0.9, cur_s)
    t_start = perf_counter()
    result = cspace_free_polytope.SearchWithBilinearAlternation(set(),
                                                                      initial_region.A(),
                                                                      initial_region.b(), 
                                                                      bilinear_alternation_options)
    t_end = perf_counter()
    if len(result)>0 and result[-1] is not None:
        cert = result[-1]
        certified_regions.update({name+"j": cert.certified_polytope()})
        visualise_IRIS([cert.certified_polytope(), initial_region], plant, plant_context)
        breakpoint()
        SaveIrisRegionsYamlFile(regions_folder+"grown_certified_region_3.yaml", certified_regions)
    else:
        print(f"COULDN'T FIND FOR {name}")

    print(t_end - t_start)

breakpoint()
