import cv2
import numpy


class TrajectoryFeature:
	
	def __init__(self):
		self.DIM = 2
	
	
	def Extract(self, flow, point, scale):
		x_pos = int(round(point[0]))
		y_pos = int(round(point[1]))
		
		feature = flow[x_pos, y_pos]/scale
		return feature
	

	def Normalize(self, feature):
		feature_normalize = feature/numpy.sum(feature)
		feature_normalize = feature_normalize.reshape(-1)
		return feature_normalize
