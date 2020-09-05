import cv2
import numpy


class MbhFeature:
	
	def __init__(self):
		self.PATCH_SIZE = 32
		self.XY_CELL_NUM = 2
		self.T_CELL_NUM = 3
		self.BIN_NUM = 8
		self.DIM = self.BIN_NUM*self.XY_CELL_NUM*self.XY_CELL_NUM
		self.EPSILON = 0.05
	

	def __ComputeIntegralFeatures(self, x_edge, y_edge):
		# Split xyedge to magnitude and angle 
		edge_mag = numpy.sqrt(x_edge*x_edge + y_edge*y_edge)
		edge_ang_rad = numpy.arctan2(y_edge, x_edge)

		# Calc two adjacent bins
		bin_origin = edge_ang_rad * self.BIN_NUM/(2*numpy.pi)
		bin_floor = numpy.floor(bin_origin).astype(numpy.int64)
		bin_ceil = numpy.ceil(bin_origin).astype(numpy.int64) % self.BIN_NUM

		# Calc adjacent bins mask
		bin_masks_floor = [(bin_floor == i).astype(numpy.int64) for i in range(self.BIN_NUM)]
		bin_masks_ceil = [(bin_ceil == i).astype(numpy.int64) for i in range(self.BIN_NUM)]
		
		# Split the magnitude to two adjacent bins
		mag_floor = (bin_origin - bin_floor)*edge_mag
		mag_ceil = (edge_mag - mag_floor)

		# Add adjacent magnitudes and convert integral features
		features = [(mag_floor*mask_floor) + (mag_ceil*mask_ceil) for(mask_floor, mask_ceil) in zip(bin_masks_floor, bin_masks_ceil)]
		integral_features = [cv2.integral(feature) for feature in features]

		return integral_features
	

	def Compute(self, flow):
		x_flow, y_flow = cv2.split(flow)
		
		x_flow_x_edge = cv2.Sobel(x_flow, cv2.CV_64F, 1, 0, ksize=1)
		x_flow_y_edge = cv2.Sobel(x_flow, cv2.CV_64F, 0, 1, ksize=1)
		y_flow_x_edge = cv2.Sobel(y_flow, cv2.CV_64F, 1, 0, ksize=1)
		y_flow_y_edge = cv2.Sobel(y_flow, cv2.CV_64F, 0, 1, ksize=1)

		integral_x_features = self.__ComputeIntegralFeatures(x_flow_x_edge, x_flow_y_edge)
		integral_x_features = self.__ComputeIntegralFeatures(y_flow_x_edge, y_flow_y_edge)

		return integral_x_features, integral_x_features
	
	def Extract(self, integral_features, point, img_size):

		block_size = int(self.PATCH_SIZE/2)
		top_right_x = int(min(max(point[0] - block_size, 0), img_size[1] - self.PATCH_SIZE))
		top_right_y = int(min(max(point[1] - block_size, 0), img_size[0] - self.PATCH_SIZE))
		
		rect_points = [[(top_right_y + block_size*idx_y,           top_right_x + block_size*idx_x),
						(top_right_y + block_size*idx_y,           top_right_x + block_size*(idx_x + 1) - 1),
						(top_right_y + block_size*(idx_y + 1) - 1, top_right_x + block_size*idx_x),
						(top_right_y + block_size*(idx_y + 1) - 1, top_right_x + block_size*(idx_x + 1) - 1)] for idx_y in range(self.XY_CELL_NUM) for idx_x in range(self.XY_CELL_NUM)]

		feature = numpy.array([integral[points[3]] + integral[points[0]] - integral[points[1]] - integral[points[2]] for points in rect_points for integral in integral_features])
		feature[feature < 0] = 0
		feature += self.EPSILON

		feature_normalize = numpy.sqrt(feature/numpy.sum(feature))
		return feature_normalize
	
	
	def Normalize(self, feature):
		split_features = numpy.split(feature, self.T_CELL_NUM)

		feature = [numpy.sum(split_feature, axis=0) for split_feature in split_features]
		feature = numpy.array(feature).reshape(-1)
		
		normalize = numpy.floor(feature.shape[0]/self.T_CELL_NUM)
		feature_normalize = feature/normalize

		return feature_normalize
