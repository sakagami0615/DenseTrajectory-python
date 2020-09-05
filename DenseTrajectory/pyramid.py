import cv2
import numpy


class PyramidImageCreator:

	def __init__(self, image_shape, min_size, scale_stride, scale_num):
		row = image_shape[0]
		col = image_shape[1]
		short_side = min(row, col)

		calc_scale_num = self.__CalcScaleNum(short_side, min_size, scale_stride, scale_num)

		rows = [row*numpy.power(scale_stride, idx) for idx in range(calc_scale_num)]
		cols = [col*numpy.power(scale_stride, idx) for idx in range(calc_scale_num)]

		self.image_num = calc_scale_num
		self.image_sizes = [(int(numpy.round(col)), int(numpy.round(row))) for col, row in zip(cols, rows)]
		self.image_scales = [1*numpy.power(1/scale_stride, idx) for idx in range(calc_scale_num)]
	

	def __CalcScaleNum(self, short_side, min_size, scale_stride, scale_num):
		sizes = [short_side*numpy.power(scale_stride, idx) for idx in range(scale_num)]
		sizes = [a for a in sizes if a > min_size]
		scale_num = len(sizes)
		if scale_num == 0: scale_num = 1
		return scale_num
	

	def Create(self, image):
		pyramid_images = [cv2.resize(image, size, interpolation=cv2.INTER_LINEAR) for size in self.image_sizes]
		return pyramid_images
