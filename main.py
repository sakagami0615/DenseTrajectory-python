
from DenseTrajectory.dense import DenseTrajectory


if __name__ == '__main__':

	#VIDEO_PATH = 'TestData/boxing.avi'
	VIDEO_PATH = 'TestData/beginner_001.avi'

	extractor = DenseTrajectory()

	hog_feature, hof_feature, mbhx_feature, mbhy_feature, trj_feature = extractor.compute(VIDEO_PATH)

	print(hog_feature.shape)
	print(hof_feature.shape)
	print(mbhx_feature.shape)
	print(mbhy_feature.shape)
	print(trj_feature.shape)
