import os
import glob
import numpy
import itertools
from DenseTrajectory.dense import DenseTrajectory


def GetVideoPaths_KTH():
	
	label_names = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

	file_path_list = [glob.glob('./dataset/KTH/{}/*avi'.format(label_name)) for label_name in label_names]

	file_paths = itertools.chain.from_iterable(file_path_list)
	return file_paths

	
def ExtractDenseTrajectoryFeatures(file_paths, save_folder_path):
	if not os.path.isdir(save_folder_path):
		os.makedirs(save_folder_path)
	
	save_file_path = '{}/{}_{}.csv'
	for file_path in file_paths:
		file_name = os.path.splitext(os.path.basename(filepath))[0]
		
		hog_feature, hof_feature, mbhx_feature, mbhy_feature, trj_feature = extractor.compute(file_path)
		numpy.savetxt(save_file_path.format(save_folder_path, file_name, 'HOG'),  hog_feature,  delimiter=',')
		numpy.savetxt(save_file_path.format(save_folder_path, file_name, 'HOF'),  hof_feature,  delimiter=',')
		numpy.savetxt(save_file_path.format(save_folder_path, file_name, 'MBHx'), mbhx_feature, delimiter=',')
		numpy.savetxt(save_file_path.format(save_folder_path, file_name, 'MBHy'), mbhy_feature, delimiter=',')
		numpy.savetxt(save_file_path.format(save_folder_path, file_name, 'TRJ'),  trj_feature,  delimiter=',')


if __name__ == '__main__':

	file_paths = GetVideoPaths_KTH()
	ExtractDenseTrajectoryFeatures(file_paths, 'result/KTH')

