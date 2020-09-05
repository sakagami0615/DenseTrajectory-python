import cv2


class TrajectoryFeature:
	def __init__(self):
		self.DIM = 2
	
	
	def Extract(self, flow, point, scale):
		x_pos = int(round(point[0]))
		y_pos = int(round(point[1]))
		
		feature = flow[x_pos, y_pos]/scale
		return feature
