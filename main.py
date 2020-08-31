
from DenseTrajectory.dense import DenseTrajectory


if __name__ == '__main__':

	#VIDEO_PATH = 'TestData/boxing.avi'
	VIDEO_PATH = 'TestData/beginner_001.avi'

	extractor = DenseTrajectory()

	extractor.compute(VIDEO_PATH)
