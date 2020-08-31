# pylint: disable=maybe-no-member
import os
import sys
import cv2
import numpy
import copy
from tqdm import tqdm
from .pyramid import PyramidImage
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

		# Create frature object
		#self.surf_create = cv2.xfeatures2d.SURF_create(self.HESSIAN_THRESH)
		self.surf_create = cv2.SIFT_create()
		self.flow_create = OpticalflowWrapper()
		self.hog_create = HogFeature()
		self.hof_create = HofFeature()
		self.mbh_create = MbhFeature()
		self.trj_create = TrajectoryFeature()


	def __DrawTrack(self, image, points):
		[cv2.circle(image, (p[1], p[0]), 2, (0,0,255), -1) for p in list(points)]

	
	def __DenseSample(self, gray_frame, src_points=None):
		# Prepare usage parameters
		width = int(gray_frame.shape[0]/self.DENSE_SAMPLE_PARAM.MIN_DIST)
		height = int(gray_frame.shape[1]/self.DENSE_SAMPLE_PARAM.MIN_DIST)
		x_max = int(self.DENSE_SAMPLE_PARAM.MIN_DIST*width)
		y_max = int(self.DENSE_SAMPLE_PARAM.MIN_DIST*height)
		offset = int(self.DENSE_SAMPLE_PARAM.MIN_DIST/2.0)

		# Prepare sampling points
		all_points = numpy.array([[w, h] for w in range(width) for h in range(height)])
		
		if src_points:
			# Floor and cast current feature points
			cast_src_points = numpy.floor(src_points).astype(numpy.int64)

			# Get previous feature point within the boundary
			enable_prev_flg = ((cast_src_points[:,0] < x_max) & (cast_src_points[:,1] < y_max))
			prev_points = cast_src_points[enable_prev_flg]/self.DENSE_SAMPLE_PARAM.MIN_DIST

			# Get points that do not match previous feature points as candidates
			enable_point_flg = ~((all_points[:,0] == prev_points[:,0]) & (all_points[:,1] == prev_points[:,1]))
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
		points_1 = numpy.array([kypt.pt for kypt in kypts_1])
		points_2 = numpy.array([kypt.pt for kypt in kypts_2])

		# Create grid data in (N rows, 2 columns) for each xy of point 1 and point 2
		x_points_21 = numpy.vstack(numpy.stack(numpy.meshgrid(points_2[:,0], points_1[:,0]), axis=-1))
		y_points_21 = numpy.vstack(numpy.stack(numpy.meshgrid(points_2[:,1], points_1[:,1]), axis=-1))

		# Calculate the difference for each xy of point 1 and point 2
		x_diffs = numpy.abs(x_points_21[:,0] - x_points_21[:,1])
		y_diffs = numpy.abs(y_points_21[:,0] - y_points_21[:,1])

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
		prev_surf_points = numpy.array([[prev_kypts[match.trainIdx].pt] for match in matches])
		curr_surf_points = numpy.array([[curr_kypts[match.queryIdx].pt] for match in matches])
		return prev_surf_points, curr_surf_points

	def __DetectFlowKeypoint(self, prev_gray, flow):
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
		return prev_points, curr_points

	def __UnionPoint(self, prev_points_1, curr_points_1, prev_points_2, curr_points_2):
		# Combine feature points vertically
		union_prev_points = numpy.vstack([prev_points_1, prev_points_2])
		union_curr_points = numpy.vstack([curr_points_1, curr_points_2])
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
		pyramid_generate = PyramidImage((height, width), self.PYRAMID_PARAM.MIN_SIZE,
														 self.PYRAMID_PARAM.PYRAMID_SCALE_STRIDE,
														 self.PYRAMID_PARAM.PYRAMID_SCALE_NUM)
		
		# Create track list
		pyramid_track_list = [TrackList(self.TRJ_PARAM.TRACK_LENGTH, self.hog_create.DIM,
																	 self.hof_create.DIM,
																	 self.mbh_create.DIM,
																	 self.mbh_create.DIM)
																	 for idx in range(pyramid_generate.image_num)]

		# ----------------------------------------------------------------------------------------------------
		# Init Process Begin
		# ----------------------------------------------------------------------------------------------------
		# Grayscale Conversion
		prev_gray = cv2.cvtColor(capture_frames[0], cv2.COLOR_BGR2GRAY)
		# Pyramid Image Generation
		prev_pyramid_gray = pyramid_generate.Create(prev_gray)
		# Dense Sampling at each scale
		scale_points = [self.__DenseSample(gray) for gray in prev_pyramid_gray]
		# Find keypoints and descriptors directly
		prev_surf_kypts, prev_surf_descs = self.surf_create.detectAndCompute(prev_gray, None)
		# ----------------------------------------------------------------------------------------------------
		# Init Process End
		# ----------------------------------------------------------------------------------------------------
		
		for capture_frame in tqdm(capture_frames[1:]):
			# Dummy Data
			#capture_frame = numpy.zeros((capture_frame.shape[0],capture_frame.shape[1],capture_frame.shape[2]),numpy.uint8)

			# Grayscale Conversion
			curr_gray = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2GRAY)
			# Pyramid Image Generation
			curr_pyramid_gray = pyramid_generate.Create(curr_gray)
			# Dense Sampling at each scale
			scale_points = [self.__DenseSample(gray) for gray in curr_pyramid_gray]
			
			# Find SURE keypoints and descriptors
			curr_surf_kypts, curr_surf_descs = self.surf_create.detectAndCompute(curr_gray, None)
			prev_surf_points, curr_surf_points = self.__KeypointMatching(prev_surf_kypts, prev_surf_descs, curr_surf_kypts, curr_surf_descs)
			# Find Flow keypoints
			pyramid_flow = [self.flow_create.ExtractFlow(prev, curr) for (prev, curr) in zip(prev_pyramid_gray, curr_pyramid_gray)]
			prev_flow_points, curr_flow_points = self.__DetectFlowKeypoint(prev_gray, pyramid_flow[0])
			# SURF and Flow Point combination
			prev_points, curr_points = self.__UnionPoint(prev_surf_points, curr_surf_points, prev_flow_points, curr_flow_points)
			
			# Calculation homography matrix
			H = numpy.eye(3)
			if (not curr_points is None) and (curr_points.shape[0] > self.HOMO_PARAM.KEYPOINT_THRESH):
				M, match_mask = cv2.findHomography(prev_points, curr_points, cv2.RANSAC, self.HOMO_PARAM.RANSAC_REPROJECT_ERROR_THRESH)
				if numpy.count_nonzero(match_mask) > self.HOMO_PARAM.MATCH_MASK_THRESH:
					H = numpy.copy(M)
		
			# WarpPerspective
			prev_gray_warp = cv2.warpPerspective(prev_gray, numpy.linalg.inv(H), prev_gray.shape)
			prev_pyramid_gray_warp = pyramid_generate.Create(prev_gray_warp)
			# Farneback OpticalFlow
			pyramid_flow_warp = [self.flow_create.ExtractFlow(prev, curr) for (prev, curr) in zip(prev_pyramid_gray_warp, curr_pyramid_gray)]
			# Compute Track Feature

			#xyScaleTracksは今までの追跡点を格納する配列
			#→ピラミッド画像ごとに格納している
			#　下記のコードでアドレスを取得し、trackを更新するとxyScaleTracksも更新されるようにしている
			#　std::list<Track>& tracks = xyScaleTracks[iScale];

			def Test(track, flow):
				width = flow.shape[0]
				height = flow.shape[1]
				prev_point = track.points[track.track_num,:]
				
				x_pos = int(min(max(round(prev_point[0]), 0), width - 1))
				y_pos = int(min(max(round(prev_point[1]), 0), height - 1))

				x_point = prev_point[0] + flow[x_pos, y_pos][0]
				y_point = prev_point[1] + flow[x_pos, y_pos][1]

				if (x_point <= 0) or (x_point >= width) or (y_point <= 0) or (y_point >= height):
					return False

				return True

			pyramid_enable_track_flg = [
											[
												Test(track, flow) for track in track_list.tracks
												] 
											for (track_list, flow) in zip(pyramid_track_list, pyramid_flow)
										]
			
			[
				track_list.RemoveTrack(track_flg) 
				for (track_list, track_flg) in zip(pyramid_track_list, pyramid_enable_track_flg)
			]
			pyramid_enable_track_list = copy.deepcopy(pyramid_track_list)
			
			def ComputeDescriptor(prev_gray, flow, flow_warp):
				hog_desc = self.hog_create.Compute(prev_gray)
				hof_desc = self.hof_create.Compute(flow)
				mbhx_desc, mbhy_desc = self.mbh_create.Compute(flow)
				return hog_desc, hof_desc, mbhx_desc, mbhy_desc

			
			def ExtractFeature(tracks, hog_desc, hof_desc, mbhx_desc, mbhy_desc):
				tracks.hog_descs = self.hog_create.Extract(hog_desc, tracks.points)
				tracks.hof_descs = self.hof_create.Extract(hof_desc, tracks.points)
				tracks.mbhx_descs = self.mbh_create.Extract(mbhx_desc, tracks.points)
				tracks.mbhy_descs = self.mbg_create.Extract(mbhy_desc, tracks.points)
				#tracks.trj_descs = self.trj_create.Extract(flow, tracks.points)
				return tracks

			def TrackRun(prev_gray, flow, flow_warp, track_list):
				hog_desc, hof_desc, mbhx_desc, mbhy_desc = ComputeDescriptor(prev_gray, flow, flow_warp)

				if track_list.tracks:
					track_list.tracks = [ExtractFeature(tracks, hog_desc, hof_desc, mbhx_desc, mbhy_desc) for tracks in track_list.tracks]

				return track_list
			
			pyramid_enable_track_list = [TrackRun(prev_gray, flow, flow_warp, track_list)
											for (prev_gray, flow, flow_warp, track_list) in zip(prev_pyramid_gray, pyramid_flow, pyramid_flow_warp, pyramid_enable_track_list)]

			"""
			// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];
			"""

			self.__DrawTrack(capture_frame, scale_points[0])
			#flow_img = self.flow_create.DrawFlow(capture_frame, pyramid_flow_warp[0])
			#capture_frame = numpy.copy(flow_img)
			writer.write(capture_frame)

			
			prev_gray = numpy.copy(curr_gray)
			prev_pyramid_gray = copy.deepcopy(curr_pyramid_gray)
			prev_surf_kypts = curr_surf_kypts
			prev_surf_descs = copy.deepcopy(curr_surf_descs)

		# ------------------------------------------------------------
		writer.release()
		# ------------------------------------------------------------
