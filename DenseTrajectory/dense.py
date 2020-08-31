import os
import sys
import cv2
import numpy
import copy
from tqdm import tqdm
from .pyramid import PyramidImageCreator
from .track import TrackList
from .flow import OpticalflowWrapper
from .Feature.hog import HogFeature
from .Feature.hof import HofFeature
from .Feature.mbh import MbhFeature
from .Feature.trajectory import TrajectoryFeature
from . import param


import time
class SpeedStack:
	def __init__(self):
		self.data_dict = {}
	
	def Add(self, name, speed_score):	
		if list(self.data_dict.keys()).count(name) == 0:
			self.data_dict[name] = 0
	
		self.data_dict[name] = self.data_dict[name] + speed_score


SPEED_STACK = SpeedStack()


class DenseTrajectory:

	def __init__(self):
		# Create parameter object
		self.DENSE_SAMPLE_PARAM = param.DenseSampleParameter()
		self.TRJ_PARAM = param.TrajectoryParameter()
		self.PYRAMID_PARAM = param.PyramidImageParameter()
		self.SURF_PARAM = param.SurfParameter()
		self.FLOW_KYPT_PARAM = param.FlowKeypointParameter()
		self.HOMO_PARAM = param.HomographyParameter()

		# Create frature object
		#self.surf_create = cv2.xfeatures2d.SURF_create(self.HESSIAN_THRESH)
		self.surf_create = cv2.SIFT_create()
		self.flow_create = OpticalflowWrapper()
		self.hog_create = HogFeature()
		self.hof_create = HofFeature()
		self.mbh_create = MbhFeature()
		self.trj_create = TrajectoryFeature()


	# TODO:処理速度遅いため、改修する
	def __DrawTrack(self, frame, track_list, image_scale):
		START = time.time()
		import math
		for track in track_list.tracks:
			points = track.points[:track.track_num]*image_scale
			if points != numpy.array([]):
				indexs_1 = numpy.array(range(points.shape[0] - 1)).tolist()
				indexs_2 = (numpy.array(range(points.shape[0] - 1)) + 1).tolist()
				darks = [math.floor(255.0*(idx + 1.0)/(track.track_num + 1.0)) for idx in indexs_2]
				
				for (index_1, index_2, dark) in zip(indexs_1, indexs_2, darks):
					pt1 = points[index_1].astype(numpy.int64)
					pt2 = points[index_2].astype(numpy.int64)
					#cv2.line(frame, (pt1[1], pt1[0]), (pt2[1], pt2[0]), (0, dark, 0), 2, 8)
					cv2.line(frame, (pt1[1], pt1[0]), (pt2[1], pt2[0]), (0, dark, 0), 1, 8)
				
				pt = points[track.track_num - 1].astype(numpy.int64)
				#cv2.circle(frame, (pt[1], pt[0]), 2, (0, 0, 255), -1, 8)
				cv2.circle(frame, (pt[1], pt[0]), 1, (0, 0, 255), -1, 8)
		SPEED_STACK.Add('__DrawTrack', time.time() - START)

	
	def __DenseSample(self, gray_frame, prev_points=None):
		# Prepare usage parameters
		width = int(gray_frame.shape[0]/self.DENSE_SAMPLE_PARAM.MIN_DIST)
		height = int(gray_frame.shape[1]/self.DENSE_SAMPLE_PARAM.MIN_DIST)
		x_max = int(self.DENSE_SAMPLE_PARAM.MIN_DIST*width)
		y_max = int(self.DENSE_SAMPLE_PARAM.MIN_DIST*height)
		offset = int(self.DENSE_SAMPLE_PARAM.MIN_DIST/2.0)

		# Prepare sampling points
		START_1 = time.time()
		all_points = numpy.array([[w, h] for w in range(width) for h in range(height)])
		SPEED_STACK.Add('__DenseSample:Prepare sampling points', time.time() - START_1)
		
		if prev_points != numpy.array([]):
			# Floor and cast current feature points
			START_2 = time.time()
			cast_prev_points = numpy.floor(prev_points).astype(numpy.int64)
			SPEED_STACK.Add('__DenseSample:Floor and cast', time.time() - START_2)

			# Get previous feature point within the boundary
			START_3 = time.time()
			enable_prev_flg = ((cast_prev_points[:,0] < x_max) & (cast_prev_points[:,1] < y_max))
			prev_points = cast_prev_points[enable_prev_flg]/self.DENSE_SAMPLE_PARAM.MIN_DIST
			SPEED_STACK.Add('__DenseSample:Get previous feature point', time.time() - START_3)

			# Get points that do not match previous feature points as candidates
			# TODO:処理速度遅いため、改修する
			START_4 = time.time()
			prev_point_list = prev_points.tolist()
			enable_point_flg = [True if prev_point_list.count(a) > 0 else True for a in all_points.tolist()]
			enable_points = all_points[enable_point_flg]*self.DENSE_SAMPLE_PARAM.MIN_DIST + offset
			SPEED_STACK.Add('__DenseSample:do not match previous', time.time() - START_4)
			
		else:
			# Get feature point candidates
			enable_points = all_points*self.DENSE_SAMPLE_PARAM.MIN_DIST + offset
				
		START_5 = time.time()
		# Calculate the smallest eigenvalue of the gradient matrix
		eigen_mat = cv2.cornerMinEigenVal(gray_frame, self.DENSE_SAMPLE_PARAM.EIGEN_BLICK_SIZE, self.DENSE_SAMPLE_PARAM.EIGEN_APERTURE_SIZE)
		# Calculate the eigenvalue threshold for corner detection
		max_value = cv2.minMaxLoc(eigen_mat)[1]
		eigen_thresh = max_value*self.DENSE_SAMPLE_PARAM.QUALITY
		SPEED_STACK.Add('__DenseSample:eigenvalue', time.time() - START_5)
		
		# Extract corner points
		START_6 = time.time()
		enable_point_eigen = eigen_mat[enable_points[:,0], enable_points[:,1]]
		corner_eigen_flg = (enable_point_eigen > eigen_thresh)
		corner_points = enable_points[corner_eigen_flg]
		SPEED_STACK.Add('__DenseSample:Extract corner', time.time() - START_6)
		return corner_points


	def __windowedMatchingMask(self, kypts_1, kypts_2, max_x_diff, max_y_diff):
		START = time.time()
		if ( not kypts_1) or ( not kypts_2):
			return None
		
		# Convert keypoint data to point data
		pts_1 = numpy.array([kypt.pt for kypt in kypts_1])
		pts_2 = numpy.array([kypt.pt for kypt in kypts_2])

		# Create grid data in (N rows, 2 columns) for each xy of point 1 and point 2
		x_pts_21 = numpy.vstack(numpy.stack(numpy.meshgrid(pts_2[:,0], pts_1[:,0]), axis=-1))
		y_pts_21 = numpy.vstack(numpy.stack(numpy.meshgrid(pts_2[:,1], pts_1[:,1]), axis=-1))

		# Calculate the difference for each xy of point 1 and point 2
		x_diffs = numpy.abs(x_pts_21[:,0] - x_pts_21[:,1])
		y_diffs = numpy.abs(y_pts_21[:,0] - y_pts_21[:,1])

		# Create a (kypts_2_num, kypts_1_num) mask matrix that does not exceed the difference threshold
		mask = ((x_diffs < max_x_diff) & (y_diffs < max_y_diff)).astype(numpy.uint8)
		mask = mask.reshape([len(kypts_2), len(kypts_1)])
		SPEED_STACK.Add('__windowedMatchingMask', time.time() - START)
		return mask


	def __KeypointMatching(self, prev_kypts, prev_descs, curr_kypts, curr_descs):
		START = time.time()
		# Keypoint matching with Brute-force
		mask = self.__windowedMatchingMask(prev_kypts, curr_kypts, self.SURF_PARAM.MATCH_MASK_THRESH, self.SURF_PARAM.MATCH_MASK_THRESH)
		matcher = cv2.BFMatcher(cv2.NORM_L2)
		matches = matcher.match(curr_descs, prev_descs, mask)
		
		# Convert keypoint data to point data
		prev_surf_pts = numpy.array([[prev_kypts[match.trainIdx].pt] for match in matches])
		curr_surf_pts = numpy.array([[curr_kypts[match.queryIdx].pt] for match in matches])
		SPEED_STACK.Add('__KeypointMatching', time.time() - START)
		return prev_surf_pts, curr_surf_pts


	def __DetectFlowKeypoint(self, prev_gray, flow):
		START = time.time()
		width = prev_gray.shape[0]
		height = prev_gray.shape[1]

		# Detect previous frame corner points
		original_prev_points = cv2.goodFeaturesToTrack(prev_gray, self.FLOW_KYPT_PARAM.MAX_COUNT, self.FLOW_KYPT_PARAM.QUALITY, self.FLOW_KYPT_PARAM.MIN_DIST)
		if original_prev_points is None:
			return None, None

		# Floor and cast current feature points
		prev_points = numpy.round(original_prev_points).astype(numpy.int64)
	
		# Feature points saturation
		prev_points[:,0,0] = numpy.clip(prev_points[:,0,0], 0, None)
		prev_points[:,0,0] = numpy.clip(prev_points[:,0,0], None, width - 1)
		prev_points[:,0,1] = numpy.clip(prev_points[:,0,1], 0, None)
		prev_points[:,0,1] = numpy.clip(prev_points[:,0,1], None, height - 1)

		# Generate feature points by adding flow to the corner points of the previous frame
		flow_points = numpy.array([[flow[point[0][0], point[0][1]]] for point in prev_points.tolist()])
		curr_points = prev_points + flow_points
		SPEED_STACK.Add('__DetectFlowKeypoint', time.time() - START)
		return prev_points, curr_points


	def __UnionPoint(self, prev_points_1, curr_points_1, prev_points_2, curr_points_2):
		START = time.time()
		# Combine feature points vertically
		union_prev_points = numpy.vstack([prev_points_1, prev_points_2])
		union_curr_points = numpy.vstack([curr_points_1, curr_points_2])
		SPEED_STACK.Add('__UnionPoint', time.time() - START)
		return union_prev_points, union_curr_points
	


	def compute(self, vieo_path):
		capture = cv2.VideoCapture(vieo_path)
		if not capture.isOpened():
			error_message = '{} is not exist.'.format(vieo_path)
			raise Exception('{}:{}():{}'.format(os.path.basename(__file__), sys._getframe().f_code.co_name, error_message))

		width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		frame_rate = int(capture.get(cv2.CAP_PROP_FPS))
		frame_size = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
		capture_frames = [capture.read()[1] for a in range(frame_size)]
		print('VideoPath:{}'.format(vieo_path))
		print('width:{}, height:{}, fps:{}, frame:{}'.format(width, height, frame_rate, frame_size))
		
		# Preparation Video Writer
		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		writer = cv2.VideoWriter('ResultData/out.mp4', fourcc, frame_rate, (width, height))

		# Create Pyramid Image Generator
		pyr_img_creator = PyramidImageCreator((height, width), self.PYRAMID_PARAM.MIN_SIZE,
															   self.PYRAMID_PARAM.PYRAMID_SCALE_STRIDE,
															   self.PYRAMID_PARAM.PYRAMID_SCALE_NUM)
		
		# Create track list
		pyr_track_list = [TrackList(self.TRJ_PARAM.TRACK_LENGTH, self.hog_create.DIM,
																 self.hof_create.DIM,
																 self.mbh_create.DIM,
																 self.mbh_create.DIM) for idx in range(pyr_img_creator.image_num)]

		# ----------------------------------------------------------------------------------------------------
		# Init Process Begin
		# ----------------------------------------------------------------------------------------------------
		progress = tqdm(total=len(capture_frames))
		# Grayscale Conversion
		gray_frame = cv2.cvtColor(capture_frames[0], cv2.COLOR_BGR2GRAY)
		# Pyramid Image Generation
		prev_pyr_gray_frame = pyr_img_creator.Create(gray_frame)
		# Find keypoints and descriptors
		prev_surf_kypts, prev_surf_descs = self.surf_create.detectAndCompute(prev_pyr_gray_frame[0], None)
		# Dense Sampling
		pyr_dense_pts = [self.__DenseSample(a) for a in prev_pyr_gray_frame]
		# Tracking points store
		[track_list.ResistTrack(pts[idx]) for (track_list, pts) in zip(pyr_track_list, pyr_dense_pts) for idx in range(pts.shape[0])]
		progress.update(1)
		# ----------------------------------------------------------------------------------------------------
		# Init Process End
		# ----------------------------------------------------------------------------------------------------

		# ----------------------------------------------------------------------------------------------------
		# Compute Process Begin
		# ----------------------------------------------------------------------------------------------------
		for capture_frame in capture_frames[1:]:
			# Grayscale Conversion
			gray_frame = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2GRAY)
			# Pyramid Image Generation
			curr_pyr_gray_frame = pyr_img_creator.Create(gray_frame)
			# Find keypoints and descriptors
			curr_surf_kypts, curr_surf_descs = self.surf_create.detectAndCompute(curr_pyr_gray_frame[0], None)
			# SURF feature matching
			prev_surf_pts, curr_surf_pts = self.__KeypointMatching(prev_surf_kypts, prev_surf_descs, curr_surf_kypts, curr_surf_descs)
			# Compute Optical Flow
			pyr_flow = [self.flow_create.ExtractFlow(curr_gray_frame, prev_gray_frame) for (curr_gray_frame, prev_gray_frame) in zip(curr_pyr_gray_frame, prev_pyr_gray_frame)]
			# Find Flow keypoints
			prev_flow_pts, curr_flow_pts = self.__DetectFlowKeypoint(prev_pyr_gray_frame[0], pyr_flow[0])
			# SURF and Flow Point combination
			prev_pts, curr_pts = self.__UnionPoint(prev_surf_pts, curr_surf_pts, prev_flow_pts, curr_flow_pts)
			
			# Calculation homography matrix
			H = numpy.eye(3)
			if (not curr_pts is None) and (curr_pts.shape[0] > self.HOMO_PARAM.KEYPOINT_THRESH):
				M, match_mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, self.HOMO_PARAM.RANSAC_REPROJECT_ERROR_THRESH)
				if numpy.count_nonzero(match_mask) > self.HOMO_PARAM.MATCH_MASK_THRESH:
					H = numpy.copy(M)
		
			# WarpPerspective
			prev_pyr_gray_warp_frame = [cv2.warpPerspective(a, numpy.linalg.inv(H), (a.shape[1], a.shape[0])) for a in prev_pyr_gray_frame]

			# Farneback OpticalFlow
			START_ExtractFlow = time.time()
			pyr_flow_warp = [self.flow_create.ExtractFlow(prev, curr) for (prev, curr) in zip(prev_pyr_gray_warp_frame, curr_pyr_gray_frame)]
			SPEED_STACK.Add('flow_create.ExtractFlow', time.time() - START_ExtractFlow)
			
			for (prev_gray_frame, flow, flow_warp, track_list) in zip(curr_pyr_gray_frame, pyr_flow, pyr_flow_warp, pyr_track_list):

				width = prev_gray_frame.shape[0]
				height = prev_gray_frame.shape[1]

				if track_list.tracks:
					prev_pts = numpy.array([track.points[track.track_num,:] for track in track_list.tracks])

					# Calcurate track points
					index = numpy.round(numpy.copy(prev_pts)).astype(numpy.int64)
					index[:,0] = numpy.clip(index[:,0], 0, None)
					index[:,0] = numpy.clip(index[:,0], None, width - 1)
					index[:,1] = numpy.clip(index[:,1], 0, None)
					index[:,1] = numpy.clip(index[:,1], None, height - 1)
					track_pts = prev_pts + flow[index[:,0], index[:,1]]
					
					# Remove points outside the range of frame
					START_RemoveTrack = time.time()
					enable_track_flg = ((track_pts[:,0] > 0) & (track_pts[:,0] < width - 1) & (track_pts[:,1] > 0) & (track_pts[:,1] < height - 1))
					enable_track_pts = track_pts[enable_track_flg]
					track_list.RemoveTrack(enable_track_flg)
					SPEED_STACK.Add('track_list.RemoveTrack', time.time() - START_RemoveTrack)

					# Compute feature description
					hog_integral = self.hog_create.Compute(prev_gray_frame)
					hof_integral = self.hof_create.Compute(flow_warp)
					mbhx_integral, mbhy_integral = self.mbh_create.Compute(flow_warp)

					# Extract features
					hog_descs = self.hog_create.Extract(hog_integral, enable_track_pts)
					hof_descs = self.hof_create.Extract(hof_integral, enable_track_pts)
					mbhx_descs = self.mbh_create.Extract(mbhx_integral, enable_track_pts)
					mbhy_descs = self.mbh_create.Extract(mbhy_integral, enable_track_pts)
					#[track.ResistDescriptor(hog_descs[idx,:] ,hof_descs[idx,:] ,mbhx_descs[idx,:] ,mbhy_descs[idx,:]) for (idx, track) in enumerate(track_list.tracks)]
					
					# Tracking points store
					START_AddPoint = time.time()
					[track.AddPoint(enable_track_pts[idx,:]) for (idx, track) in enumerate(track_list.tracks)]
					SPEED_STACK.Add('track.AddPoint', time.time() - START_AddPoint)
					
					continue_track_flg = [track.CheckEnable() for track in track_list.tracks]
					track_list.RemoveTrack(continue_track_flg)
					# TODO:下記に特徴抽出したデータの保管処理を


			
			# Draw Tracking points
			START_DrawTrack = time.time()
			self.__DrawTrack(capture_frame, pyr_track_list[0], pyr_img_creator.image_scales[0])
			SPEED_STACK.Add('__DrawTrack', time.time() - START_DrawTrack)
			
			# store new feature points
			START_ResistTrack = time.time()
			pyr_curr_pts = [numpy.array([track.points[track.track_num] for track in track_list.tracks]) for track_list in pyr_track_list]
			SPEED_STACK.Add('track_list.ResistTrack', time.time() - START_ResistTrack)
			pyr_dense_pts = [self.__DenseSample(gray_frame, curr_pts) for (gray_frame, curr_pts) in zip(prev_pyr_gray_frame, pyr_curr_pts)]
			START_ResistTrack = time.time()
			[track_list.ResistTrack(dense_pts[idx]) for (track_list, dense_pts) in zip(pyr_track_list, pyr_dense_pts) for idx in range(dense_pts.shape[0])]
			SPEED_STACK.Add('track_list.ResistTrack', time.time() - START_ResistTrack)

			#flow_img = self.flow_create.DrawFlow(capture_frame, pyr_flow_warp[0])
			#flow_img = self.flow_create.DrawFlow(capture_frame, pyr_flow[0])
			#capture_frame = numpy.copy(flow_img)
			writer.write(capture_frame)


			def keypoints_deepcopy(f):
				return [cv2.KeyPoint(x = k.pt[0], y = k.pt[1], 
						_size = k.size, _angle = k.angle, 
						_response = k.response, _octave = k.octave, 
						_class_id = k.class_id) for k in f]

			# Update
			prev_pyr_gray_frame = copy.deepcopy(curr_pyr_gray_frame)
			#prev_surf_kypts = curr_surf_kypts
			prev_surf_kypts = keypoints_deepcopy(curr_surf_kypts)
			prev_surf_descs = numpy.copy(curr_surf_descs)

			progress.update(1)
		# ----------------------------------------------------------------------------------------------------
		# Compute Process End
		# ----------------------------------------------------------------------------------------------------

		import csv
		with open('ResultData/log_speed.csv', 'w') as f:
			writer = csv.writer(f)
			for key in SPEED_STACK.data_dict.keys():
				print(key,SPEED_STACK.data_dict[key],SPEED_STACK.data_dict[key]/len(capture_frames))
				writer.writerow([key,SPEED_STACK.data_dict[key],SPEED_STACK.data_dict[key]/len(capture_frames)])