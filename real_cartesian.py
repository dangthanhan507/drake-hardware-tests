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
    Simulator,
    LeafSystem,
    Context
)
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import SnoptSolver
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.all import StartMeshcat, PiecewisePose, Quaternion, PiecewisePolynomial, TrajectorySource
from manipulation.station import MakeHardwareStation, MakeHardwareStationInterface, load_scenario
JOINT0 = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, np.pi/6, 0.0])
CARTESIAN_ENDTIME = 20.0
JOINT_STEPS = 100

def solveIK(plant: MultibodyPlant, target_pose: RigidTransform, frame_name: str, q0=1e-10*np.ones(7)):
    ik = InverseKinematics(plant, with_joint_limits=True)
    ik.AddPositionConstraint(plant.GetFrameByName(frame_name),
                             np.array([0,0,0]),
                             plant.world_frame(),
                             target_pose.translation(),
                             target_pose.translation()
    )
    ik.AddOrientationConstraint(
        plant.GetFrameByName(frame_name),
        RotationMatrix(RollPitchYaw(0,0,0)),
        plant.world_frame(),
        target_pose.rotation(),
        0.0
    )
    prog = ik.get_mutable_prog()
    q = ik.q()
    if np.abs(q0).sum() > 1e-4:
        prog.AddQuadraticErrorCost(np.eye(7), q0, q)
    prog.SetInitialGuess(q, q0)
    solver = SnoptSolver()
    result = solver.Solve(prog)
    if not result.is_success():
        raise ValueError("Inverse Kinematics Failed")
    return result.GetSolution()

def create_traj(plant: MultibodyPlant, ee_pose0: RigidTransform, q0: np.ndarray, endtime=CARTESIAN_ENDTIME, steps=JOINT_STEPS):
    ee_pose1 = RigidTransform(ee_pose0.rotation(), ee_pose0.translation() + np.array([0.1,0,0]))
    ee_pose2 = RigidTransform(ee_pose1.rotation(), ee_pose1.translation() + np.array([0.0,0.1,0]))
    ee_pose3 = RigidTransform(ee_pose2.rotation(), ee_pose2.translation() + np.array([-0.1,0.0,0]))
    ee_pose4 = RigidTransform(ee_pose3.rotation(), ee_pose3.translation() + np.array([0.0,-0.1,0]))
    ee_poses = [ee_pose0, ee_pose1, ee_pose2, ee_pose3, ee_pose4]
    ts_ee = np.linspace(0,endtime,5)
    piecewise_ee = PiecewisePose.MakeLinear(ts_ee, ee_poses)
    ee_pos_traj = piecewise_ee.get_position_trajectory()
    ee_ori_traj = piecewise_ee.get_orientation_trajectory()
    
    ts_joints = np.linspace(0, endtime, steps)
    curr_q = q0.copy()
    qs = []
    for ts_joint in ts_joints:
        pose = RigidTransform(Quaternion(ee_ori_traj.value(ts_joint)), ee_pos_traj.value(ts_joint))
        q = solveIK(plant, pose, "iiwa_link_7", curr_q)
        qs.append(q)
        curr_q = q
    qs = np.array(qs)
    q_traj = PiecewisePolynomial.FirstOrderHold(ts_joints, qs.T)
    return q_traj

def get_hardware_setup(hardware_builder, scenario, meshcat, ):
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
    
    ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        plant=hardware_plant,
        parser=parser
    )
    return real_station, fake_station

def create_visual_diagram(directives_file: str):
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
    return diagram

class BenchmarkController(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        
        
        self._target = self.DeclareVectorInputPort("target", 7)
        self._measured = self.DeclareVectorInputPort("measure", 7)
        self.DeclarePeriodicPublishEvent(period_sec=1e-2, offset_sec=0.0, publish=self.Publish)
        
        self.ts = []
        self.q_targets = []
        self.q_measures = []
        self.positions = []
        self.des_positions = []
    def Publish(self, context: Context):
        self.ts.append(context.get_time())
        q_target = self._target.Eval(context)
        q_measure = self._measured.Eval(context)
        
        self._plant.SetPositions(self._plant_context, q_measure)
        ee_pose = self._plant.CalcRelativeTransform(self._plant_context, self._plant.world_frame(), self._plant.GetBodyByName("iiwa_link_7").body_frame())
        position = ee_pose.translation()
        
        self._plant.SetPositions(self._plant_context, q_target)
        ee_pose_des = self._plant.CalcRelativeTransform(self._plant_context, self._plant.world_frame(), self._plant.GetBodyByName("iiwa_link_7").body_frame())
        position_des = ee_pose_des.translation()
        
        
        self.q_measures.append(q_measure)
        self.q_targets.append(q_target)
        self.positions.append(position)
        self.des_positions.append(position_des)
        

if __name__ == '__main__':
    meshcat = StartMeshcat()
    scenario_file = "iiwa_standard_setup.yaml"
    scenario = load_scenario(filename=scenario_file, scenario_name="Demo")
    directives_file = "iiwa.yaml"
    
    hardware_builder = DiagramBuilder()
    real_station, fake_station = get_hardware_setup(hardware_builder, scenario, meshcat)
    
    hardware_plant = fake_station.GetSubsystemByName("plant")
    default_positions = hardware_plant.GetPositions(hardware_plant.CreateDefaultContext())
    
    
    #### create trajectory
    q0 = JOINT0
    plant_context = hardware_plant.CreateDefaultContext()
    hardware_plant.SetPositions(plant_context, q0)
    ee_pose0 = hardware_plant.CalcRelativeTransform(plant_context, hardware_plant.world_frame(), hardware_plant.GetBodyByName("iiwa_link_7").body_frame())
    q_traj = create_traj(hardware_plant, ee_pose0, q0, endtime=CARTESIAN_ENDTIME, steps=JOINT_STEPS)
    ##############################
    
    
    position_trajectory = hardware_builder.AddNamedSystem("position_trajectory", TrajectorySource(q_traj))
    torque_output = hardware_builder.AddNamedSystem("torque_source", ConstantVectorSource(np.zeros(7)))
    record_dataguy = BenchmarkController(hardware_plant)
    rec = hardware_builder.AddSystem(record_dataguy)
    hardware_builder.Connect(position_trajectory.get_output_port(), real_station.GetInputPort("iiwa.position"))    
    hardware_builder.Connect(torque_output.get_output_port(), real_station.GetInputPort("iiwa.feedforward_torque"))
    hardware_builder.Connect(real_station.GetOutputPort("iiwa.position_commanded"), rec.get_input_port(0))
    hardware_builder.Connect(real_station.GetOutputPort("iiwa.position_measured"), rec.get_input_port(1))
    
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa.position_commanded"), "iiwa.position_commanded"
    )
    hardware_builder.ExportOutput(
        real_station.GetOutputPort("iiwa.position_measured"), "iiwa.position_measured"
    )
    hardware_diagram = hardware_builder.Build()
    
    diagram = create_visual_diagram(directives_file)
    
    input("Press Enter to Start.")
    simulator = Simulator(hardware_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(CARTESIAN_ENDTIME + 3.0)
    
    # done running
    import matplotlib.pyplot as plt
    q_targets = np.array(record_dataguy.q_targets)
    q_measures = np.array(record_dataguy.q_measures)
    positions = np.array(record_dataguy.positions)
    des_positions = np.array(record_dataguy.des_positions)
    ts = np.array(record_dataguy.ts)
    plt.figure()
    plt.plot(ts, positions[:,2], c='r', label='meas')
    plt.plot(ts, des_positions[:,2], c='b', label='des')
    plt.legend()
    
    fig, axs = plt.subplots(7, 1)
    for i in range(7):
        if i == 0:
            axs[i].plot(ts, q_targets[:,i], c='r',label='des')
            axs[i].plot(ts, q_measures[:,i], c='b', label='meas')
            axs[i].legend()
        else:
            axs[i].plot(ts, q_targets[:,i], c='r')
            axs[i].plot(ts, q_measures[:,i], c='b')
    plt.show()