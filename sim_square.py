import numpy as np
from pydrake.multibody.parsing import LoadModelDirectives, ProcessModelDirectives
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlantConfig, AddMultibodyPlant
from pydrake.geometry import SceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import SnoptSolver
from pydrake.systems.framework import DiagramBuilder
from pydrake.all import StartMeshcat, PiecewisePose, Quaternion, PiecewisePolynomial, TrajectorySource, LeafSystem, Context
from pydrake.systems.analysis import Simulator
from pydrake.visualization import AddDefaultVisualization
import os
JOINT0 = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, np.pi/6, 0.0])
CARTESIAN_ENDTIME = 10.0
JOINT_STEPS = 50

def load_iiwa_setup(plant: MultibodyPlant, scene_graph: SceneGraph = None):
    parser = Parser(plant, scene_graph)
    # directive_path = "med.yaml"
    # directives = LoadModelDirectives(directive_path)
    # models = ProcessModelDirectives(directives, plant, parser)
    parser.AddModels("./med.urdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
    

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

class BenchmarkController(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        
        
        self._target = self.DeclareVectorInputPort("target", 7)
        self._measured = self.DeclareVectorInputPort("measure", 14)
        self.DeclarePeriodicPublishEvent(period_sec=1e-2, offset_sec=0.0, publish=self.Publish)
        
        self.ts = []
        self.q_targets = []
        self.q_measures = []
        self.positions = []
        self.des_positions = []
    def Publish(self, context: Context):
        self.ts.append(context.get_time())
        q_target = self._target.Eval(context)
        q_measure = self._measured.Eval(context)[:7]
        
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
    
    config = MultibodyPlantConfig()
    config.time_step = 1e-3
    config.penetration_allowance = 1e-9
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlant(config, builder)
    plant: MultibodyPlant = plant # for help w/ vscode :) 
    load_iiwa_setup(plant, scene_graph)
    plant.Finalize()
    
    print(plant.num_positions())
    print(plant.GetPositionNames())
    
    num_joints = plant.num_positions()
    # kp = np.array([800, 600, 600, 600, 400, 200, 200])
    kp = 400*np.ones(7)
    ki = np.zeros(7)
    kd = np.ones(7)
    controller_block = builder.AddSystem(InverseDynamicsController(plant, kp, ki, kd, False))
    
    q0 = JOINT0

    # make it go in a square +0.1x -> +0.1y -> -0.1x -> -0.1y
    plant_context = plant.CreateDefaultContext()
    plant.SetPositions(plant_context, q0)
    ee_pose0 = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetBodyByName("iiwa_link_7").body_frame())
    q_traj = create_traj(plant, ee_pose0, q0, endtime=CARTESIAN_ENDTIME, steps=JOINT_STEPS)
    
    traj_block = builder.AddSystem(TrajectorySource(q_traj,output_derivative_order=1))
    traj0_block = builder.AddSystem(TrajectorySource(q_traj, output_derivative_order=0))
    result = BenchmarkController(plant)
    result_block = builder.AddSystem(result)
    builder.Connect(traj_block.get_output_port(), controller_block.get_input_port_desired_state())
    builder.Connect(plant.get_state_output_port(), controller_block.get_input_port_estimated_state())
    builder.Connect(controller_block.get_output_port(), plant.get_actuation_input_port())
    builder.Connect(traj0_block.get_output_port(), result_block.GetInputPort("target"))
    builder.Connect(plant.get_state_output_port(), result_block.GetInputPort("measure"))
    
    
    AddDefaultVisualization(builder, meshcat)
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("med"), JOINT0)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    meshcat.StartRecording()
    simulator.AdvanceTo(CARTESIAN_ENDTIME+5.0)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    
    import matplotlib.pyplot as plt
    # make 7 subplots
    fig, axs = plt.subplots(7, 1, figsize=(10, 20))
    for i in range(7):
        axs[i].plot(result.ts, np.array(result.q_targets)[:,i], label="target")
        axs[i].plot(result.ts, np.array(result.q_measures)[:,i], label="measured")
        axs[i].set_title(f"Joint {i}")
        axs[i].legend()
        
    plt.figure() # plot z positions
    positions = np.array(result.positions)
    des_positions = np.array(result.des_positions)
    plt.plot(result.ts, positions[:,2], label="z")
    plt.plot(result.ts, des_positions[:,2], label="z_des")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Z Position (m)")
    plt.show()