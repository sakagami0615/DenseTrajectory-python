import cv2


class HogFeature:
	def __init__(self):
		self.PATCH_SIZE = 32
		self.XY_CELL_NUM = 2
		self.T_CELL_NUM = 3
		self.BIN_NUM = 8
		self.DIM = self.BIN_NUM*self.XY_CELL_NUM*self.XY_CELL_NUM
	
	def Compute(self, gray):
		x_edge = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
		y_edge = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)

		mag, ang = cv2.cartToPolar(x_edge, y_edge)
		return None
	
	def Extract(self, integral, point):
		import numpy
		return numpy.zeros((1, self.DIM))