from planning import create_traj, GravityCompensation
from diagrams import create_hardware_diagram_plant, create_visual_diagram
from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    RigidTransform,
    ConstantVectorSource
)
import numpy as np
from data_record import BenchmarkController
JOINT0 = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, np.pi/6, 0.0])
ENDTIME = 20.0

if __name__ == '__main__':
    meshcat = StartMeshcat()
    
    # scenario_file = "iiwa_setup_impedance.yaml"
    # directives_file = "iiwa.yaml"
    scenario_file = "med_setup_impedance.yaml"
    directives_file = "med.yaml"
    
    root_builder = DiagramBuilder()
    
    hardware_diagram, hardware_plant = create_hardware_diagram_plant(scenario_filepath=scenario_file, meshcat=meshcat, position_only=False)
    vis_diagram = create_visual_diagram(directives_filepath=directives_file, meshcat=meshcat)
    
    plant_context = hardware_plant.CreateDefaultContext()
    hardware_plant.SetPositions(plant_context, JOINT0)
    ee_pose0 = hardware_plant.CalcRelativeTransform(plant_context, hardware_plant.world_frame(), hardware_plant.GetBodyByName("iiwa_link_7").body_frame())
    ee_pose1 = RigidTransform(ee_pose0.rotation(), ee_pose0.translation() + np.array([0.1,0,0]))
    ee_pose2 = RigidTransform(ee_pose1.rotation(), ee_pose1.translation() + np.array([0.0,0.1,0]))
    ee_pose3 = RigidTransform(ee_pose2.rotation(), ee_pose2.translation() + np.array([-0.1,0.0,0]))
    ee_pose4 = RigidTransform(ee_pose3.rotation(), ee_pose3.translation() + np.array([0.0,-0.1,0]))
    ee_poses = [ee_pose0, ee_pose1, ee_pose2, ee_pose3, ee_pose4]
    
    traj = TrajectorySource(create_traj(hardware_plant, ee_poses, JOINT0, endtime=ENDTIME, steps=100), output_derivative_order=0)
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    traj_block = root_builder.AddSystem(traj)
    record_dataguy = BenchmarkController(hardware_plant)
    record_block = root_builder.AddSystem(record_dataguy)
    
    grav_block = root_builder.AddSystem(GravityCompensation(hardware_plant))
    
    root_builder.Connect(traj_block.get_output_port(), hardware_block.GetInputPort("iiwa.position"))
    root_builder.Connect(traj_block.get_output_port(), hardware_block.GetInputPort("iiwa_fake.position"))
    
    root_builder.Connect(hardware_block.GetOutputPort("iiwa.position_measured"), grav_block.get_input_port(0))
    root_builder.Connect(grav_block.get_output_port(), hardware_block.GetInputPort("iiwa.feedforward_torque"))
    root_builder.Connect(grav_block.get_output_port(), hardware_block.GetInputPort("iiwa_fake.feedforward_torque"))
    
    # zero_torque_block = root_builder.AddSystem(ConstantVectorSource(np.zeros(7)))
    # root_builder.Connect(zero_torque_block.get_output_port(), hardware_block.GetInputPort("iiwa.feedforward_torque"))
    # root_builder.Connect(zero_torque_block.get_output_port(), hardware_block.GetInputPort("iiwa_fake.feedforward_torque"))
    
    
    root_builder.Connect(hardware_block.GetOutputPort("iiwa.position_commanded"), record_block.GetInputPort("target"))
    root_builder.Connect(hardware_block.GetOutputPort("iiwa.position_measured"), record_block.GetInputPort("measure"))
    
    root_diagram = root_builder.Build()
    
    # run simulation
    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(ENDTIME + 2.0)
    
    # done running
    import matplotlib.pyplot as plt
    q_targets = np.array(record_dataguy.q_targets)
    q_measures = np.array(record_dataguy.q_measures)
    positions = np.array(record_dataguy.positions)
    des_positions = np.array(record_dataguy.des_positions)
    ts = np.array(record_dataguy.ts)
    fig, axs = plt.subplots(3, 1)
    for i in range(3):
        axs[i].plot(ts, positions[:,i], c='r',label='meas')  
        axs[i].plot(ts, des_positions[:,i], c='b',label='des')
        if i == 0:
          axs[i].legend()
        # axs[i].plot(ts, np.abs(positions[:,i] - des_positions[:,i]), c='r')
    
    fig, axs = plt.subplots(7, 1)
    for i in range(7):
        axs[i].plot(ts, q_targets[:,i], c='r',label='des')
        axs[i].plot(ts, q_measures[:,i], c='b', label='meas')
        if i == 0:
            axs[i].legend()
    plt.show()