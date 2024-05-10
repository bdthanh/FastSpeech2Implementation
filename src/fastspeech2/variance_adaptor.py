from torch.nn import Module
from components.variance_predictor import VariancePredictor
from components.length_regulator import LengthRegulator

class VarianceAdaptor(Module):
    #TODO: To be implemented
    def __init__(self) -> None:
        super().__init__()
        self.duration_predictor = VariancePredictor()
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor()
        self.energy_predictor = VariancePredictor()
    
    def forward(self, x):
        pass