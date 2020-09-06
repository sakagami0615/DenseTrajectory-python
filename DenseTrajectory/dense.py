import os
import sys
import cv2
import numpy
import copy
import itertools
from tqdm import tqdm
from .pyramid import PyramidImageCreator
from .track import TrackList
from .flow import OpticalflowWrapper
from .Feature.hog import HogFeature
from .Feature.hof import HofFeature
from .Feature.mbh import MbhFeature
from .Feature.trajectory import TrajectoryFeature
from . import param


class DenseTrajectory:

	def __init__(self):
		# Create parameter object
		self.DENSE_SAMPLE_PARAM = param.DenseSampleParameter()
		self.TRJ_PARAM = param.TrajectoryParameter()
		self.PYRAMID_PARAM = param.PyramidImageParameter()
		self.SURF_PARAM = param.SurfParameter()
		self.FLOW_KYPT_PARAM = param.FlowKeypointParameter()
		self.HOMO_PARAM = param.HomographyParameter()

		# TODO:xfeatures2dが使用できないため、Dockerfileを見直す（暫定でSIFTを代用）
		# Create frature object
		#self.surf_create = cv2.xfeatures2d.SURF_create(self.HESSIAN_THRESH)
		self.surf_create = cv2.SIFT_create()
		self.flow_create = OpticalflowWrapper()
		self.hog_create = HogFeature()
		self.hof_create = HofFeature()
		self.mbh_create = MbhFeature()
		self.trj_create = TrajectoryFeature()


	def __DrawTrack(self, frame, track_list, image_scale):
		def GetLineData(track, image_scale):
			track_pts = (track.points[:track.track_num + 1]*image_scale).astype(numpy.int64)

			if track_pts.shape[0] > 1:
				line_begin_idxs = numpy.array(range(track_pts.shape[0] - 1)).tolist()
				line_end_idxs = (numpy.array(range(track_pts.shape[0] - 1)) + 1).tolist()
				line_begin_pts = [(track_pts[i][1], track_pts[i][0]) for i in line_begin_idxs]
				line_end_pts = [(track_pts[j][1], track_pts[j][0]) for j in line_end_idxs]
				line_colors = [(0, numpy.floor(255.0*(i + 1.0)/(track.track_num + 1.0)), 0) for i in line_end_idxs]
			elif track_pts.shape[0] == 1:
				line_end_pts = [(track_pts[0][1], track_pts[0][0])]
				line_begin_pts = []
				line_colors = []
			else:
				line_begin_pts = []
				line_end_pts = []
				line_colors = []
			
			return line_begin_pts, line_end_pts, line_colors
		
		def DrawTrackLine(frame, line_begin_pts, line_end_pts, line_colors):
			[cv2.line(frame, begin_pt, end_pt, line_color, 2, 8) for (begin_pt, end_pt, line_color) in zip(line_begin_pts, line_end_pts, line_colors)]
			if line_end_pts:
				cv2.circle(frame, line_end_pts[-1], 2, (0, 0, 255), -1, 8)

		line_datas = [GetLineData(track, image_scale) for track in track_list.tracks]
		[DrawTrackLine(frame, data[0], data[1], data[2]) for data in line_datas]
		
	
	def __DenseSample(self, gray_frame, prev_points=None):
		# Prepare usage parameters
		width = int(gray_frame.shape[0]/self.DENSE_SAMPLE_PARAM.MIN_DIST)
		height = int(gray_frame.shape[1]/self.DENSE_SAMPLE_PARAM.MIN_DIST)
		x_max = int(self.DENSE_SAMPLE_PARAM.MIN_DIST*width)
		y_max = int(self.DENSE_SAMPLE_PARAM.MIN_DIST*height)
		offset = int(self.DENSE_SAMPLE_PARAM.MIN_DIST/2.0)

		# Prepare sampling points
		all_points = numpy.array([[w, h] for w in range(width) for h in range(height)])
		
		if prev_points != numpy.array([]):
			# Floor and cast current feature points
			cast_prev_points = numpy.floor(prev_points).astype(numpy.int64)

			# Get previous feature point within the boundary
			enable_prev_flg = ((cast_prev_points[:,0] < x_max) & (cast_prev_points[:,1] < y_max))
			prev_points = (cast_prev_points[enable_prev_flg]/self.DENSE_SAMPLE_PARAM.MIN_DIST).astype(numpy.int64)

			# Get points that do not match previous feature points as candidates
			prev_point_list = prev_points.tolist()
			enable_point_flg = [True if prev_point_list.count(a) == 0 else False for a in all_points.tolist()]
			enable_points = all_points[enable_point_flg]*self.DENSE_SAMPLE_PARAM.MIN_DIST + offset
		else:
			# Get feature point candidates
			enable_points = all_points*self.DENSE_SAMPLE_PARAM.MIN_DIST + offset
				
		# Calculate the smallest eigenvalue of the gradient matrix
		eigen_mat = cv2.cornerMinEigenVal(gray_frame, self.DENSE_SAMPLE_PARAM.EIGEN_BLICK_SIZE, self.DENSE_SAMPLE_PARAM.EIGEN_APERTURE_SIZE)
		# Calculate the eigenvalue threshold for corner detection
		max_value = cv2.minMaxLoc(eigen_mat)[1]
		eigen_thresh = max_value*self.DENSE_SAMPLE_PARAM.QUALITY
		
		# Extract corner points
		enable_point_eigen = eigen_mat[enable_points[:,0], enable_points[:,1]]
		corner_eigen_flg = (enable_point_eigen > eigen_thresh)
		corner_points = enable_points[corner_eigen_flg]
		return corner_points


	def __windowedMatchingMask(self, kypts_1, kypts_2, max_x_diff, max_y_diff):
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
		return mask


	def __KeypointMatching(self, prev_kypts, prev_descs, curr_kypts, curr_descs):
		# Keypoint matching with Brute-force
		mask = self.__windowedMatchingMask(prev_kypts, curr_kypts, self.SURF_PARAM.MATCH_MASK_THRESH, self.SURF_PARAM.MATCH_MASK_THRESH)
		matcher = cv2.BFMatcher(cv2.NORM_L2)
		matches = matcher.match(curr_descs, prev_descs, mask)
		
		# Convert keypoint data to point data
		prev_surf_pts = numpy.array([[prev_kypts[match.trainIdx].pt] for match in matches])
		curr_surf_pts = numpy.array([[curr_kypts[match.queryIdx].pt] for match in matches])
		return prev_surf_pts, curr_surf_pts


	def __DetectFlowKeypoint(self, prev_gray, flow):
		width = prev_gray.shape[0]
		height = prev_gray.shape[1]

		# Detect previous frame corner points
		original_prev_points = cv2.goodFeaturesToTrack(prev_gray, self.FLOW_KYPT_PARAM.MAX_COUNT, self.FLOW_KYPT_PARAM.QUALITY, self.FLOW_KYPT_PARAM.MIN_DIST)
		if original_prev_points is None:
			prev_points = None
			curr_points = None
		else:
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
		
		return prev_points, curr_points


	def __UnionPoint(self, prev_points_1, curr_points_1, prev_points_2, curr_points_2):
		# Combine feature points vertically
		union_prev_points = numpy.vstack([prev_points_1, prev_points_2])
		union_curr_points = numpy.vstack([curr_points_1, curr_points_2])
		return union_prev_points, union_curr_points
	

	def __ExtractFeatureDescs(self, prev_gray_frame, flow_warp, track_list, image_scale):
		# Compute feature description
		hog_integral = self.hog_create.Compute(prev_gray_frame)
		hof_integral = self.hof_create.Compute(flow_warp)
		mbhx_integral, mbhy_integral = self.mbh_create.Compute(flow_warp)

		# Extract features
		hog_descs  = [self.hog_create.Extract(hog_integral,  track.points[track.track_num], prev_gray_frame.shape) for track in track_list.tracks]
		hof_descs  = [self.hof_create.Extract(hof_integral,  track.points[track.track_num], prev_gray_frame.shape) for track in track_list.tracks]
		mbhx_descs = [self.mbh_create.Extract(mbhx_integral, track.points[track.track_num], prev_gray_frame.shape) for track in track_list.tracks]
		mbhy_descs = [self.mbh_create.Extract(mbhy_integral, track.points[track.track_num], prev_gray_frame.shape) for track in track_list.tracks]
		trj_descs  = [self.trj_create.Extract(flow_warp, track.points[track.track_num], image_scale) for track in track_list.tracks]
		[track.ResistDescriptor(hog, hof, mbhx, mbhy, trj)
			for (track, hog, hof, mbhx, mbhy, trj) in zip(track_list.tracks, hog_descs, hof_descs, mbhx_descs, mbhy_descs, trj_descs)]

	
	def __AddTrackPoints(self, flow, track_list, image_size):
		if track_list.tracks:
			prev_pts = numpy.array([track.points[track.track_num,:] for track in track_list.tracks])

			# Calcurate track points
			index = numpy.round(numpy.copy(prev_pts)).astype(numpy.int64)
			index[:,0] = numpy.clip(index[:,0], 0, None)
			index[:,0] = numpy.clip(index[:,0], None, image_size[1] - 1)
			index[:,1] = numpy.clip(index[:,1], 0, None)
			index[:,1] = numpy.clip(index[:,1], None, image_size[0] - 1)
			track_pts = prev_pts + flow[index[:,0], index[:,1]]
			
			# Remove points outside the range of frame
			enable_track_flg = ((track_pts[:,0] > 0) & (track_pts[:,0] < image_size[1] - 1) & (track_pts[:,1] > 0) & (track_pts[:,1] < image_size[0] - 1))
			enable_track_pts = track_pts[enable_track_flg]
			track_list.RemoveTrack([not flg for flg in enable_track_flg])

			# Tracking points store
			[track.AddPoint(enable_track_pts[idx,:]) for (idx, track) in enumerate(track_list.tracks)]
	

	def __ResistTracks(self, prev_gray_frame, track_list):
		track_pts = numpy.array([track.points[track.track_num] for track in track_list.tracks])
		dense_pts = self.__DenseSample(prev_gray_frame, track_pts)
		[track_list.ResistTrack(dense_pts[idx]) for idx in range(dense_pts.shape[0])]

	
	def __RemoveTracks(self, track_list):
		remove_track_flg = [track.CheckRemove() for track in track_list.tracks]
		remove_tracks = track_list.RemoveTrack(remove_track_flg)
		return remove_tracks

	
	def __ExtractTrackFeature(self, pyr_remove_tracks, pyr_scales):
		def ExtractTrackFeature(remove_tracks, scale):
			valid_track_flg = [track.CheckValidTrajectory(scale) for track in remove_tracks]
			motion_track_flg = [track.CheckNotCameraMotion() for track in remove_tracks]

			hog_feature  = [self.hog_create.Normalize(track.hog_descs)
							for (track, valid_flg, motion_flg) in zip(remove_tracks, valid_track_flg, motion_track_flg) if valid_flg and motion_flg]
			hof_feature  = [self.hof_create.Normalize(track.hof_descs)
							for (track, valid_flg, motion_flg) in zip(remove_tracks, valid_track_flg, motion_track_flg) if valid_flg and motion_flg]
			mbhx_feature = [self.mbh_create.Normalize(track.mbhx_descs)
							for (track, valid_flg, motion_flg) in zip(remove_tracks, valid_track_flg, motion_track_flg) if valid_flg and motion_flg]
			mbhy_feature = [self.mbh_create.Normalize(track.mbhy_descs)
							for (track, valid_flg, motion_flg) in zip(remove_tracks, valid_track_flg, motion_track_flg) if valid_flg and motion_flg]
			trj_feature  = [self.trj_create.Normalize(track.trj_descs)
							for (track, valid_flg, motion_flg) in zip(remove_tracks, valid_track_flg, motion_track_flg) if valid_flg and motion_flg]

			return hog_feature, hof_feature, mbhx_feature, mbhy_feature, trj_feature

		pyr_features = [ExtractTrackFeature(remove_tracks, scale) for (remove_tracks, scale) in zip(pyr_remove_tracks, pyr_scales) if remove_tracks]

		hog_features  = itertools.chain.from_iterable([features[0] for features in pyr_features])
		hof_features  = itertools.chain.from_iterable([features[1] for features in pyr_features])
		mbhx_features = itertools.chain.from_iterable([features[2] for features in pyr_features])
		mbhy_features = itertools.chain.from_iterable([features[3] for features in pyr_features])
		trj_features  = itertools.chain.from_iterable([features[4] for features in pyr_features])
		return hog_features, hof_features, mbhx_features, mbhy_features, trj_features


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
		
		hog_feature_store  = []
		hof_feature_store  = []
		mbhx_feature_store = []
		mbhy_feature_store = []
		trj_feature_store  = []
		
		#TODO
		"""
		# Preparation Video Writer
		if self.TRJ_PARAM.DRAW_TRACK_FLG:
			fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
			writer = cv2.VideoWriter('ResultData/out.mp4', fourcc, frame_rate, (width, height))
		"""

		# Create Pyramid Image Generator
		pyr_img_creator = PyramidImageCreator((height, width), self.PYRAMID_PARAM.MIN_SIZE,
															   self.PYRAMID_PARAM.PYRAMID_SCALE_STRIDE,
															   self.PYRAMID_PARAM.PYRAMID_SCALE_NUM)
		
		# Create track list
		pyr_track_list = [TrackList(self.hog_create.DIM,
									self.hof_create.DIM,
									self.mbh_create.DIM,
									self.mbh_create.DIM,
									self.trj_create.DIM) for idx in range(pyr_img_creator.image_num)]

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
			pyr_flow_warp = [self.flow_create.ExtractFlow(prev, curr) for (prev, curr) in zip(prev_pyr_gray_warp_frame, curr_pyr_gray_frame)]
			
			# Extract feature descriptor
			[self.__ExtractFeatureDescs(prev_gray_frame, flow_warp, track_list, image_scale)
				for (prev_gray_frame, flow_warp, track_list, image_scale) in zip(curr_pyr_gray_frame, pyr_flow_warp, pyr_track_list, pyr_img_creator.image_scales)]			
			
			# Add track points
			[self.__AddTrackPoints(flow, track_list, image_size) for (flow, track_list, image_size) in zip(pyr_flow, pyr_track_list, pyr_img_creator.image_sizes)]
			
			# Draw Tracking points
			"""
			if self.TRJ_PARAM.DRAW_TRACK_FLG:
				self.__DrawTrack(capture_frame, pyr_track_list[0], pyr_img_creator.image_scales[0])
				#TODO
				# writer.write(capture_frame)
				cv2.imshow('DrawTrack', capture_frame)
				cv2.waitKey(1)
			"""
			
			# Remove tracking ended track datas
			pyr_remove_tracks = [self.__RemoveTracks(track_list) for track_list in pyr_track_list]

			# Extract tracking ended track features
			hog_features, hof_features, mbhx_features, mbhy_features, trj_features = self.__ExtractTrackFeature(pyr_remove_tracks, pyr_img_creator.image_scales)
			hog_feature_store.extend(hog_features)
			hof_feature_store.extend(hof_features)
			mbhx_feature_store.extend(mbhx_features)
			mbhy_feature_store.extend(mbhy_features)
			trj_feature_store.extend(trj_features)

			# Regist new points in track data
			[self.__ResistTracks(prev_gray_frame, track_list) for (prev_gray_frame, track_list) in zip(prev_pyr_gray_frame, pyr_track_list)]
			
			# Update current to previous
			prev_pyr_gray_frame = copy.deepcopy(curr_pyr_gray_frame)
			prev_surf_kypts = curr_surf_kypts
			prev_surf_descs = numpy.copy(curr_surf_descs)

			progress.update(1)
		# ----------------------------------------------------------------------------------------------------
		# Compute Process End
		# ----------------------------------------------------------------------------------------------------

		hog_feature_store  = numpy.array(hog_feature_store)
		hof_feature_store  = numpy.array(hof_feature_store)
		mbhx_feature_store = numpy.array(mbhx_feature_store)
		mbhy_feature_store = numpy.array(mbhy_feature_store)
		trj_feature_store  = numpy.array(trj_feature_store)

		return hog_feature_store, hof_feature_store, mbhx_feature_store, mbhy_feature_store, trj_feature_store
