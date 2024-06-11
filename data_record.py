from pydrake.all import (
    MultibodyPlant,
    LeafSystem,
    Context
)
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