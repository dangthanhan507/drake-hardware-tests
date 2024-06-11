from planning import create_traj
from diagrams import create_hardware_diagram_plant, create_visual_diagram
from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    RigidTransform,
    LeafSystem,
    Context
)
import numpy as np
JOINT0 = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, np.pi/6, 0.0])
ENDTIME = 20.0

if __name__ == '__main__':
    meshcat = StartMeshcat()
    scenario_file = "iiwa_standard_setup.yaml"
    directives_file = "iiwa.yaml"
    
    root_builder = DiagramBuilder()
    
    hardware_diagram, hardware_plant = create_hardware_diagram_plant(scenario_filepath=scenario_file, meshcat=meshcat, position_only=True)
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
    
    root_builder.Connect(traj_block.get_output_port(), hardware_block.GetInputPort("iiwa.position"))
    root_builder.Connect(traj_block.get_output_port(), hardware_block.GetInputPort("iiwa_fake.position"))
    
    root_diagram = root_builder.Build()
    
    # run simulation
    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(ENDTIME + 2.0)