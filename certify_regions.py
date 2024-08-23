from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
import os
from pydrake.all import (
    LoadModelDirectives, ProcessModelDirectives, RevoluteJoint, 
    RationalForwardKinematics, CspaceFreePolytope, SeparatingPlaneOrder,
    RigidTransform, RotationMatrix, Rgba,
    AffineSubspace, MathematicalProgram, Solve,
    MeshcatVisualizer, StartMeshcat
)
import numpy as np
# from pydrake.geometry.optimization_dev import (CspaceFreePolytope, SeparatingPlaneOrder)
from iris_plant_visualizer import IrisPlantVisualizer
from pydrake.geometry import Role
from pydrake.geometry.optimization import IrisOptions, HPolyhedron, Hyperellipsoid, IrisInRationalConfigurationSpace, LoadIrisRegionsYamlFile, SaveIrisRegionsYamlFile
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions, ScsSolver
import time
import pickle

#construct our robot
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parser = Parser(plant)

parser.package_map().Add("ciris_pgd", os.path.abspath(''))

directives_file = "/home/shrutigarg/drake/ciris-pgd/models/iiwa14_sphere_collision_complex_scenario.dmd.yaml"
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
# q_star = np.zeros(plant.num_positions())
q_star = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
do_viz = True

# The object we will use to perform our certification.
cspace_free_polytope = CspaceFreePolytope(plant, scene_graph, SeparatingPlaneOrder.kAffine, q_star)

# set up the certifier and the options for different search techniques
solver_options = SolverOptions()
# set this to 1 if you would like to see the solver output in terminal.
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 0)

os.environ["MOSEKLM_LICENSE_FILE"] = "/home/shrutigarg/mosek.lic"
with open(os.environ["MOSEKLM_LICENSE_FILE"], 'r') as f:
    contents = f.read()
    mosek_file_not_empty = contents != ''
print(mosek_file_not_empty)

solver_id = MosekSolver.id() if MosekSolver().available() and mosek_file_not_empty else ScsSolver.id()


solver_id = MosekSolver.id() if MosekSolver().available() else ScsSolver.id()

# load Savva's regions
iris_regions = LoadIrisRegionsYamlFile("/home/shrutigarg/drake/ciris-pgd/ComplexScenarioRegions.yaml")
# Some seedpoints
list_regions = list(iris_regions.values())
seed_points_q = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
regions_to_sample = [0, 13, 18, 19]
for reg_idx in regions_to_sample:
    seed_q = list_regions[reg_idx].ChebyshevCenter()
    seed_points_q = np.append(seed_points_q, [seed_q], axis=0)

seed_points = np.array([Ratfk.ComputeSValue(seed_points_q[idx], q_star)\
                        for idx in range(seed_points_q.shape[0])])

# generate C-IRIS regions with these seedpoints
default_scale = 1e-2
L1_ball = HPolyhedron.MakeL1Ball(7)
Linf_ball = HPolyhedron.MakeBox(-np.ones(7), np.ones(7))

template_C = np.vstack([L1_ball.A(), Linf_ball.A()])
template_d = np.hstack([default_scale*L1_ball.b(), default_scale/np.sqrt(2)*Linf_ball.b()])


def make_default_polytope_at_point(seed_point):
    return HPolyhedron(template_C, template_d + template_C @ seed_point)

# colors to plot the region.
default_alpha = 0.2
colors_dict = {
    0: Rgba(0.565, 0.565, 0.565, default_alpha), # gray
    1: Rgba(0.118, 0.533, 0.898, default_alpha), # bluish
    2: Rgba(1,     0.757, 0.027, default_alpha), # gold
    3: Rgba(0,     0.549, 0.024, default_alpha), # green   
    4: Rgba(0.055, 0.914, 0.929, default_alpha), # teal 
}

initial_regions = [(make_default_polytope_at_point(s), colors_dict[i]) for i, s in enumerate(seed_points)]
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

start = time.perf_counter()
ciris_regions = []
ciris_ellipses = []

iris_options = IrisOptions()
iris_options.require_sample_point_is_contained = True
iris_options.configuration_space_margin = 1e-3
iris_options.relative_termination_threshold = 0.001

context_for_iris = context
for i, s in enumerate(seed_points):
    print("seed point ", i, " finished")
    q = Ratfk.ComputeQValue(s, q_star)
    plant.SetPositions(plant.GetMyMutableContextFromRoot(context_for_iris), q)
    r = IrisInRationalConfigurationSpace(plant, 
                                         plant.GetMyContextFromRoot(context_for_iris),
                                         q_star, iris_options)
    ciris_regions.append((r, colors_dict[i]))
    ciris_ellipses.append(r.MaximumVolumeInscribedEllipsoid())
end = time.perf_counter()
print("time taken ", end-start)

ciris_regions_proc = [elem[0] for elem in ciris_regions]
ciris_regions_proc_names = ["Origin", "AboveEverything", "RightBin-Above", "RightShelf-Front3", "RightShelf-Middle1"]
regions_dict = dict()
for n, r in zip(ciris_regions_proc_names, ciris_regions_proc):
    regions_dict[n] = r

SaveIrisRegionsYamlFile("/home/shrutigarg/drake/ciris-pgd/cirisregions3.yaml", regions_dict)

ciris_regions = LoadIrisRegionsYamlFile("/home/shrutigarg/drake/ciris-pgd/cirisregions3.yaml")
print(ciris_regions)

binary_search_region_certificates_for_iris = dict.fromkeys([tuple(name) for name in ciris_regions.keys()])
for i, (name, initial_region) in enumerate(zip(ciris_regions.keys(), ciris_regions.values())):
    print(f"starting seedpoint {i+1}/{len(ciris_regions)}")
    time.sleep(0.2)
    start = time.perf_counter()
    cert = cspace_free_polytope.BinarySearch(set(),
                                                    initial_region.A(),
                                                    initial_region.b(), 
                                                    initial_region.MaximumVolumeInscribedEllipsoid().center(), 
                                                    binary_search_options)
    binary_search_region_certificates_for_iris[name] = [(cert.certified_polytope() if cert is not None else None, cert)]
    end = time.perf_counter()
    print(end-start)

# Serialize the data to a file
with open('certficates.pkl', 'wb') as file:
    pickle.dump(binary_search_region_certificates_for_iris, file)
