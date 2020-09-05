import numpy
from .param import TrajectoryParameter

class Track:

	def __init__(self, init_point, hog_dim, hof_dim, mbhx_dim, mbhy_dim, trj_dim):
		self.PARAM = TrajectoryParameter()
		self.track_num = 0
		self.hog_descs = numpy.empty((self.PARAM.TRACK_LENGTH, hog_dim))
		self.hof_descs = numpy.empty((self.PARAM.TRACK_LENGTH, hof_dim))
		self.mbhx_descs = numpy.empty((self.PARAM.TRACK_LENGTH, mbhx_dim))
		self.mbhy_descs = numpy.empty((self.PARAM.TRACK_LENGTH, mbhy_dim))
		self.trj_descs = numpy.empty((self.PARAM.TRACK_LENGTH, trj_dim))

		self.points = numpy.empty((self.PARAM.TRACK_LENGTH + 1, 2))
		self.points[self.track_num,:] = init_point
	
	
	def AddPoint(self, point):
		self.track_num += 1
		self.points[self.track_num,:] = point
		
	
	def ResistDescriptor(self, hog_desc, hof_desc, mbhx_desc, mbhy_desc, trj_desc):
		self.hog_descs[self.track_num,:] = hog_desc
		self.hof_descs[self.track_num,:] = hof_desc
		self.mbhx_descs[self.track_num,:] = mbhx_desc
		self.mbhy_descs[self.track_num,:] = mbhy_desc
		self.trj_descs[self.track_num,:] = trj_desc
	
	
	def CheckRemove(self):
		if self.PARAM.TRACK_LENGTH > self.track_num:
			return False
		return True
	

	def CheckValidTrajectory(self, scale):
		points_x = self.points[:,0]/scale
		points_y = self.points[:,1]/scale

		std_x = numpy.std(points_x)
		std_y = numpy.std(points_y)

		# Remove static trajectory
		if (std_x < self.PARAM.REJECT_MIN_STD) and (std_y < self.PARAM.REJECT_MIN_STD):
			return False
		# Remove random trajectory
		if (std_x > self.PARAM.REJECT_MAX_STD) or (std_y > self.PARAM.REJECT_MAX_STD):
			return False
		
		mag = numpy.sqrt(points_x*points_x + points_y*points_y)
		max_mag = numpy.amax(mag)
		sum_mag = numpy.sum(mag)

		if (max_mag > self.PARAM.REJECT_MAX_DIST) and (max_mag > sum_mag*0.7):
			return False

		return True
	
	
	def CheckNotCameraMotion(self):
		mag = numpy.sqrt(self.trj_descs[:,0]*self.trj_descs[:,0] + self.trj_descs[:,1]*self.trj_descs[:,1])
		if numpy.amax(mag) <= 1:
			return False
		return True



class TrackList:
	
	def __init__(self, hog_dim, hof_dim, mbhx_dim, mbhy_dim, trj_dim):
		self.tracks = []
		self.hog_dim = hog_dim
		self.hof_dim = hof_dim
		self.mbhx_dim = mbhx_dim
		self.mbhy_dim = mbhy_dim
		self.trj_dim = trj_dim
	
	
	def ResistTrack(self, point):
		track = Track(point, self.hog_dim, self.hof_dim, self.mbhx_dim, self.mbhy_dim, self.trj_dim)
		self.tracks.append(track)
		
	
	def RemoveTrack(self, remove_flg):
		self.tracks = numpy.array(self.tracks)
		remove_tracks = numpy.copy(self.tracks[remove_flg])
		remove_tracks = remove_tracks.tolist()
		self.tracks = self.tracks[[not flg for flg in remove_flg]]
		self.tracks = self.tracks.tolist()
		
		return remove_tracks
