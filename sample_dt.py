import os
import glob
import numpy
import itertools
from DenseTrajectory.dense import DenseTrajectory


def GetVideoPaths_KTH():
	label_names = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

	file_path_list = [glob.glob('./dataset/KTH/{}/*avi'.format(label_name)) for label_name in label_names]

	file_paths = list(itertools.chain.from_iterable(file_path_list))
	return file_paths

	
def ExtractDenseTrajectoryFeatures(file_paths, save_folder_path):
	save_feature_dir = '{}/feature'.format(save_folder_path)
	save_video_dir = '{}/video'.format(save_folder_path)

	if not os.path.isdir(save_feature_dir):
		os.makedirs(save_feature_dir)
	if not os.path.isdir(save_video_dir):
		os.makedirs(save_video_dir)
	
	extractor = DenseTrajectory()

	save_feature_path = '{}/{}_{}.csv'
	save_video_path = '{}/{}.avi'

	for (idx, file_path) in enumerate(file_paths):
		print('[process : {}/{}]'.format(idx + 1, len(file_paths)))
		file_name = os.path.splitext(os.path.basename(file_path))[0]
		
		hog_feature, hof_feature, mbhx_feature, mbhy_feature, trj_feature = extractor.compute(file_path, save_video_path.format(save_video_dir, file_name))
		numpy.savetxt(save_feature_path.format(save_feature_dir, file_name, 'HOG'),  hog_feature,  delimiter=',')
		numpy.savetxt(save_feature_path.format(save_feature_dir, file_name, 'HOF'),  hof_feature,  delimiter=',')
		numpy.savetxt(save_feature_path.format(save_feature_dir, file_name, 'MBHx'), mbhx_feature, delimiter=',')
		numpy.savetxt(save_feature_path.format(save_feature_dir, file_name, 'MBHy'), mbhy_feature, delimiter=',')
		numpy.savetxt(save_feature_path.format(save_feature_dir, file_name, 'TRJ'),  trj_feature,  delimiter=',')


if __name__ == '__main__':

	file_paths = GetVideoPaths_KTH()
	ExtractDenseTrajectoryFeatures(file_paths, 'result/KTH')
