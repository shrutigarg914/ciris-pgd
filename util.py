import numpy as np
import pydot
import scipy
import matplotlib.pyplot as plt
from IPython.display import display, Image
from pydrake.all import (
    IllustrationProperties,
    GeometrySet,
    Role,
    RoleAssign,
    VPolytope,
    MeshcatVisualizerParams,
    DiagramBuilder,
    LoadModelDirectives,
    ProcessModelDirectives,
    AddMultibodyPlantSceneGraph,
    Parser,
    MultibodyPlant,
    SceneGraph,
    MeshcatVisualizer,
    RationalForwardKinematics,
    FunctionHandleTrajectory,
    Toppra,
    PathParameterizedTrajectory
)
import os

def notebook_plot_connectivity(regions, plot_dot=True, plot_adj_mat=True):
    adj_mat = np.zeros((len(regions), len(regions)))
    graph = pydot.Dot("IRIS region connectivity")
    n = len(regions)
    for i in range(n):
        graph.add_node(pydot.Node(i))
    for i in range(n):
        for j in range(i+1,n):
            if regions[i].IntersectsWith(regions[j]):
                graph.add_edge(pydot.Edge(i, j, dir="both"))
                adj_mat[i,j] = adj_mat[j,i] = 1
    if plot_dot:
        display(Image(graph.create_png()))
    if plot_adj_mat:
        plt.imshow(adj_mat)
        plt.show()
    count, _ = scipy.sparse.csgraph.connected_components(adj_mat)
    print("Graph has %d connected components." % count)

def in_collision(plant, scene_graph, context, print_collisions=False, thresh=1e-12):
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

def notebook_plot_connectivity(regions, plot_dot=True, plot_adj_mat=True):
    adj_mat = np.zeros((len(regions), len(regions)))
    graph = pydot.Dot("IRIS region connectivity")
    n = len(regions)
    for i in range(n):
        graph.add_node(pydot.Node(i))
    for i in range(n):
        for j in range(i+1,n):
            if regions[i].IntersectsWith(regions[j]):
                graph.add_edge(pydot.Edge(i, j, dir="both"))
                adj_mat[i,j] = adj_mat[j,i] = 1
    if plot_dot:
        display(Image(graph.create_png()))
    if plot_adj_mat:
        plt.imshow(adj_mat)
        plt.show()
    count, _ = scipy.sparse.csgraph.connected_components(adj_mat)
    print("Graph has %d connected components." % count)

def visualize_state(q, plant, diagram, diagram_context):
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    plant.SetPositions(plant_context, q)
    diagram.ForcedPublish(diagram_context)

def lower_alpha(plant, scene_graph, alpha):
    inspector = scene_graph.model_inspector()
    for gid in inspector.GetGeometryIds(GeometrySet(inspector.GetAllGeometryIds()), Role.kIllustration):
        try:
            prop = inspector.GetIllustrationProperties(gid)
            new_props = IllustrationProperties(prop)
            phong = prop.GetProperty("phong", "diffuse")
            phong.set(phong.r(), phong.g(), phong.b(), alpha)
            new_props.UpdateProperty("phong", "diffuse", phong)
            scene_graph.AssignRole(plant.get_source_id(), gid, new_props, RoleAssign.kReplace)
        except:
            pass

def build_env(meshcat, directives_file, export_sg_input=False, suffix=""):
    builder = DiagramBuilder()
    
    if export_sg_input:
        scene_graph = builder.AddSystem(SceneGraph())
        plant = MultibodyPlant(time_step=0.0)
        plant.RegisterAsSourceForSceneGraph(scene_graph)
        builder.ExportInput(scene_graph.get_source_pose_port(plant.get_source_id()), "source_pose")
    else:
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    
    parser = Parser(plant)
    
    parser.package_map().Add("ciris_pgd", os.path.abspath(''))
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)

    plant.Finalize()
    
    meshcat_visual_params = MeshcatVisualizerParams()
    meshcat_visual_params.delete_on_initialization_event = False
    meshcat_visual_params.role = Role.kIllustration
    meshcat_visual_params.prefix = "visual" + suffix
    meshcat_visual = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, meshcat_visual_params)

    meshcat_collision_params = MeshcatVisualizerParams()
    meshcat_collision_params.delete_on_initialization_event = False
    meshcat_collision_params.role = Role.kProximity
    meshcat_collision_params.prefix = "collision" + suffix
    meshcat_collision_params.visible_by_default = False
    meshcat_collision = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, meshcat_collision_params)

    diagram = builder.Build()
    
    return diagram, plant, scene_graph

def display_iris(build_env, meshcat, directives_file, region, count=10, alpha=0.25, name_start=0, undistort=False, q_star=None):
    diagrams = []
    plants = []

    vpoly = VPolytope(region)

    for i in range(count):
        idx = np.random.randint(vpoly.vertices().shape[1])
        q = vpoly.vertices()[:,idx]
        diagram, plant, scene_graph = build_env(meshcat, directives_file, suffix=str(i + name_start))
        if undistort:
            q = 2*np.arctan2(q, np.ones_like(q))
        lower_alpha(plant, scene_graph, alpha)
        visualize_state(q, plant, diagram, diagram.CreateDefaultContext())
        diagrams.append(diagram)
        plants.append(plant)

    return diagrams, plants

def remap_and_toppra_trajectory(traj, q_to_q_full, plant, vel_limits=None, accel_limits=None, end_effector_accel_limits=None, n_grid_points=1000, vel_limit_rescale=1.0, accel_limit_rescale=0.25):
    # path should be the output from GcsTrajectoryOptimization
    # q_to_q_full should take in a configuration q and output q_full (both stored as numpy lists)

    if vel_limits is None:
        vel_limits = (plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits())
    else:
        vel_limit_rescale = 1.0
    if accel_limits is None:
        accel_limits = (plant.GetAccelerationLowerLimits(), plant.GetAccelerationUpperLimits())
    else:
        accel_limit_rescale = 1.0

    assert isinstance(vel_limits, tuple)
    assert len(vel_limits) == 2
    assert isinstance(accel_limits, tuple)
    assert len(accel_limits) == 2

    # Compute default grid points
    # options = CalcGridPointsOptions(
    #     max_err=1e-2,
    #     max_iter=100,
    #     max_seg_length=0.05,
    #     min_points=100
    # )
    # gridpoints = Toppra.CalcGridPoints(traj, options)
    gridpoints = np.linspace(traj.start_time(), traj.end_time(), n_grid_points)

    # Add grid points at boundaries between each trajectory segment
    n_segments = traj.get_number_of_segments()
    new_grid_points = [traj.end_time(i) for i in range(n_segments - 1)]
    gridpoints = np.sort(np.unique(np.append(gridpoints, new_grid_points)))

    # Create FunctionHandleTrajectory
    def remapped_traj_function(t):
        return q_to_q_full(traj.vector_values([t]).flatten()).reshape(-1, 1)

    remapped_traj = FunctionHandleTrajectory(func=remapped_traj_function,
                                             rows=7,
                                             cols=1,
                                             start_time=traj.start_time(),
                                             end_time=traj.end_time())

    # Construct Toppra instance
    toppra = Toppra(remapped_traj, plant, gridpoints)
    toppra.AddJointVelocityLimit(vel_limits[0] * vel_limit_rescale, vel_limits[1] * vel_limit_rescale)
    toppra.AddJointAccelerationLimit(accel_limits[0] * accel_limit_rescale, accel_limits[1] * accel_limit_rescale)

    if end_effector_accel_limits is not None:
        for model_instance_name in ["wsg_left", "wsg_right"]:
            model_instance = plant.GetModelInstanceByName(model_instance_name)
            body = plant.GetBodyByName("body", model_instance)
            end_effector_frame = body.body_frame()
            toppra.AddFrameAccelerationLimit(end_effector_frame, -end_effector_accel_limits, end_effector_accel_limits)

    time_traj = toppra.SolvePathParameterization()
    return PathParameterizedTrajectory(remapped_traj, time_traj)


def plot_traj_end_effector_path(q_path, meshcat, IK_obj, name="path", color_rgba=(1, 0, 0, 1)):
	points_local = np.array([IK_obj.FK(q_full[:7])[:-1,-1] for q_full in q_full_path])

	pointcloud = PointCloud(len(points_local))
	pointcloud.mutable_xyzs()[:] = points_local.T
	meshcat.SetObject("paths/" + name, pointcloud, 0.015, rgba=Rgba(*color_rgba))