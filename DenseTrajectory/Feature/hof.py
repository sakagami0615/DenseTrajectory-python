import cv2


class HofFeature:
	def __init__(self):
		self.PATCH_SIZE = 32
		self.XY_CELL_NUM = 2
		self.T_CELL_NUM = 3
		self.MIN_FLOW_THRESH = 0.4
		self.BIN_NUM = 9
		self.DIM = self.BIN_NUM*self.XY_CELL_NUM*self.XY_CELL_NUM
	
	def Compute(self, flow):
		x_flow, y_flow = cv2.split(flow)
		return None
	
	def Extract(self, integral, point):
		return None
