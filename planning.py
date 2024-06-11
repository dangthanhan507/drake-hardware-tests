import numpy as np
from pydrake.multibody.all import MultibodyPlant
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import SnoptSolver
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.trajectories import PiecewisePolynomial, PiecewisePose
from pydrake.common.eigen_geometry import Quaternion
from typing import List

from pydrake.systems.framework import LeafSystem
from pydrake.all import Context, MultibodyPlant
class GravityCompensation(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        
        self._measured = self.DeclareVectorInputPort("measure", 7)
        self.DeclareVectorOutputPort("feedforward_torques", 7, self.CalcOutput)
    def CalcOutput(self, context: Context, output):
        q = self.get_input_port(0).Eval(context)
        self._plant.SetPositions(self._plant_context, q)
        tau_g = self._plant.CalcGravityGeneralizedForces(self._plant_context)
        output.SetFromVector(-tau_g)

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

def create_traj(plant: MultibodyPlant, ee_poses: List[RigidTransform], q0: np.ndarray, endtime=20.0, steps=100):
    ts_ee = np.linspace(0,endtime,len(ee_poses))
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

