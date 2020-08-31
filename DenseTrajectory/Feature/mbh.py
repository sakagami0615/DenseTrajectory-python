import cv2


class MbhFeature:
	def __init__(self):
		self.PATCH_SIZE = 32
		self.XY_CELL_NUM = 2
		self.T_CELL_NUM = 3
		self.BIN_NUM = 8
		self.DIM = self.BIN_NUM*self.XY_CELL_NUM*self.XY_CELL_NUM
	
	def Compute(self, flow):
		x_flow, y_flow = cv2.split(flow)
		
		x_flow_x_edge = cv2.Sobel(x_flow, cv2.CV_64F, 1, 0, ksize=1)
		x_flow_y_edge = cv2.Sobel(x_flow, cv2.CV_64F, 0, 1, ksize=1)
		y_flow_x_edge = cv2.Sobel(y_flow, cv2.CV_64F, 1, 0, ksize=1)
		y_flow_y_edge = cv2.Sobel(y_flow, cv2.CV_64F, 0, 1, ksize=1)

		return None, None
	
	def Extract(self, desc, point):
		return None
