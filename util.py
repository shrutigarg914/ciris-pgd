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
    RationalForwardKinematics
)

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
