import cv2
import numpy


class OpticalflowWrapper:
	
	def __init__(self):
		self.pyr_scale = 0.5
		self.levels = 3
		self.winsize = 15
		self.iterations = 3
		self.poly_n = 5
		self.poly_sigma = 1.2
	
	
	def ExtractFlow(self, prev, curr):
		flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
											self.pyr_scale,
											self.levels,
											self.winsize,
											self.iterations,
											self.poly_n,
											self.poly_sigma,
											cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
		return flow
	
	
	def DrawFlow(self, base_img, flow):
		hsv = numpy.zeros_like(base_img)
		hsv[...,1] = 255

		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		hsv[...,0] = ang*180/numpy.pi/2
		hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
		return flow_rgb
