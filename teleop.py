import numpy as np
from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    MultibodyPlant,
    ApplyMultibodyPlantConfig,
    Parser,
    ProcessModelDirectives,
    ModelDirectives,
    TrajectorySource,
    PiecewisePolynomial,
    ConstantVectorSource,
    AddMultibodyPlantSceneGraph,
    LoadModelDirectives,
    MeshcatVisualizerParams,
    Role,
    MeshcatVisualizer,
    Simulator
)
from manipulation.station import MakeHardwareStation, MakeHardwareStationInterface, load_scenario
from manipulation.scenarios import WsgPositions

if __name__ == '__main__':
    meshcat = StartMeshcat()
    scenario_file = "iiwa_standard_setup.yaml"
    scenario = load_scenario(filename=scenario_file, scenario_name="Demo")
    directives_file = "iiwa.yaml"
    package_file = "package.xml"
    
    
    #hardware diagram
    #################################
    #################################
    hardware_builder = DiagramBuilder()
    real_station = hardware_builder.AddNamedSystem(
        "real_station",
        MakeHardwareStationInterface(
            scenario,
            meshcat=meshcat
        )
    )
    fake_station = hardware_builder.AddNamedSystem(
        "fake_station",
        MakeHardwareStation(
            scenario,
            meshcat=meshcat,
        )
    )
    
    hardware_plant = MultibodyPlant(scenario.plant_config.time_step)
    ApplyMultibodyPlantConfig(scenario.plant_config, hardware_plant)
    parser = Parser(hardware_plant)
    # parser.package_map().AddPackageXml(package_file)
    
    ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        plant=hardware_plant,
        parser=parser
    )
    hardware_plant = fake_station.GetSubsystemByName("plant")
    
    default_positions = hardware_plant.GetPositions(hardware_plant.CreateDefaultContext())
    # position_trajectory = hardware_builder.AddNamedSystem(
	# 	"position_trajectory", TrajectorySource(PiecewisePolynomial(default_positions))
	# )
 
    joint_initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_middle = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi/8, 0.0])
    joint_end = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
    qs = np.array([joint_initial, joint_middle, joint_end])
    ts = np.array([0.0, 5.0, 10.0])
    
    traj = PiecewisePolynomial.FirstOrderHold(ts, qs.T)
    position_trajectory = hardware_builder.AddNamedSystem(
        "position_trajectory", TrajectorySource(traj)
    )
    torque_output = hardware_builder.AddNamedSystem(
        "torque_source", ConstantVectorSource(np.zeros(7))
    )
    
    hardware_builder.Connect(position_trajectory.get_output_port(), fake_station.GetInputPort("iiwa.position"))
    hardware_builder.Connect(position_trajectory.get_output_port(), real_station.GetInputPort("iiwa.position"))
    hardware_builder.Connect(torque_output.get_output_port(), fake_station.GetInputPort("iiwa.feedforward_torque"))
    hardware_builder.Connect(torque_output.get_output_port(), real_station.GetInputPort("iiwa.feedforward_torque"))
    
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa.position_commanded"), "iiwa.position_commanded"
    )
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa.position_measured"), "iiwa.position_measured"
    )
    hardware_diagram = hardware_builder.Build()
    #################################
    #################################
    
    
    
    #visualization diagram
    #################################
    #################################
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    # run simulator.set_target_realtime_rate(1.0)
    parser = Parser(plant)
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)
    plant.Finalize()
    
    meshcat_visual_params = MeshcatVisualizerParams()
    meshcat_visual_params.delete_on_initialization_event = False
    meshcat_visual_params.role = Role.kIllustration
    meshcat_visual_params.prefix = "visual"
    meshcat_visual = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, meshcat_visual_params)
    
    meshcat_collision_params = MeshcatVisualizerParams()
    meshcat_collision_params.delete_on_initialization_event = False
    meshcat_collision_params.role = Role.kProximity
    meshcat_collision_params.prefix = "collision"
    meshcat_collision_params.visible_by_default = False
    meshcat_collision = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, meshcat_collision_params)
    diagram = builder.Build()
    #################################
    #################################
    
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    
    ts = np.array([0.0, 20.0])
    
    curr_q = hardware_diagram.GetOutputPort("iiwa.position_commanded").Eval(context)
    des_q = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, np.pi/6, 0.0])
    # des_q = np.zeros(7)
    
    qs = np.array([curr_q, des_q])
    traj = PiecewisePolynomial.FirstOrderHold(ts, qs.T)
    position_trajectory = hardware_diagram.GetSubsystemByName("position_trajectory")
    position_trajectory.UpdateTrajectory(traj)
    
    run = True
    if run:    
        simulator = Simulator(hardware_diagram)
        simulator.set_target_realtime_rate(1.0)
        simulator.AdvanceTo(40.0)
    else:
        print("Joint Values: ", curr_q * 180 / np.pi) 
    print("Done")