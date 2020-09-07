import numpy


class DenseSampleParameter:
	QUALITY = 0.001
	MIN_DIST = 5
	EIGEN_BLICK_SIZE = 3
	EIGEN_APERTURE_SIZE = 3


class TrajectoryParameter:
	TRACK_LENGTH = 15
	REJECT_MIN_STD = numpy.sqrt(3)
	REJECT_MAX_STD = 50
	REJECT_MAX_DIST = 20


class FlowKeypointParameter:
	MAX_COUNT = 1000
	QUALITY = 0.01
	MIN_DIST = 10


class SurfParameter:
	HESSIAN_THRESH = 200
	MATCH_MASK_THRESH = 25


class PyramidImageParameter:
	MIN_SIZE = 32
	PYRAMID_SCALE_STRIDE = 1/numpy.sqrt(2)
	PYRAMID_SCALE_NUM = 8


class HomographyParameter:
	KEYPOINT_THRESH = 50
	MATCH_MASK_THRESH = 25
	RANSAC_REPROJECT_ERROR_THRESH = 1
