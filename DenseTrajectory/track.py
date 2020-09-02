import numpy


class Track:
	def __init__(self, init_point, track_length, hog_dim, hof_dim, mbhx_dim, mbhy_dim):
		self.track_num = 0
		self.track_length = track_length
		self.hog_descs = numpy.empty((track_length, hog_dim))
		self.hof_descs = numpy.empty((track_length, hof_dim))
		self.mbhx_descs = numpy.empty((track_length, mbhx_dim))
		self.mbhy_descs = numpy.empty((track_length, mbhy_dim))

		self.points = numpy.empty((track_length + 1, 2))
		self.points[self.track_num,:] = init_point
	
	def AddPoint(self, point):
		self.track_num += 1
		self.points[self.track_num,:] = point
		
	
	def ResistDescriptor(self, hog_desc, hof_desc, mbhx_desc, mbhy_desc):
		self.hog_descs[self.track_num,:] = hog_desc
		self.hof_descs[self.track_num,:] = hof_desc
		self.mbhx_descs[self.track_num,:] = mbhx_desc
		self.mbhy_descs[self.track_num,:] = mbhy_desc
	
	def CheckEnable(self):
		if self.track_length > self.track_num:
			return True
		return False


class TrackList:
	def __init__(self, track_length, hog_dim, hof_dim, mbhx_dim, mbhy_dim):
		self.tracks = []
		self.track_length = track_length
		self.hog_dim = hog_dim
		self.hof_dim = hof_dim
		self.mbhx_dim = mbhx_dim
		self.mbhy_dim = mbhy_dim
	
	def ResistTrack(self, point):
		track = Track(point, self.track_length, self.hog_dim, self.hof_dim, self.mbhx_dim, self.mbhy_dim)
		self.tracks.append(track)
	
	def RemoveTrack(self, enable_flg):
		self.tracks = numpy.array(self.tracks)
		self.tracks = self.tracks[enable_flg]
		self.tracks = self.tracks.tolist()
