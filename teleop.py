from diagrams import create_hardware_diagram_plant, create_visual_diagram
from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    TrajectorySource,
    PiecewisePolynomial
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
    
    hardware_block = root_builder.AddSystem(hardware_diagram)
    
    ## make a plan from current position to desired position
    context = hardware_diagram.CreateDefaultContext()
    hardware_diagram.ExecuteInitializationEvents(context)
    curr_q = hardware_diagram.GetOutputPort("iiwa.position_commanded").Eval(context)
    des_q = JOINT0
    ts = np.array([0.0, ENDTIME])
    qs = np.array([curr_q, des_q])
    traj = PiecewisePolynomial.FirstOrderHold(ts, qs.T)
    
    traj_block = root_builder.AddSystem(TrajectorySource(traj))
    root_builder.Connect(traj_block.get_output_port(), hardware_block.GetInputPort("iiwa.position"))
    root_builder.Connect(traj_block.get_output_port(), hardware_block.GetInputPort("iiwa_fake.position"))
    
    root_diagram = root_builder.Build()
    
    # run simulation
    simulator = Simulator(root_diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(ENDTIME + 2.0)