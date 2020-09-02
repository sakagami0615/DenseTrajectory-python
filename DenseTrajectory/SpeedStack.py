import time


class SpeedStack:
	def __init__(self):
		self.measure_time = {}
		self.start_time = {}
	
	def TimerBegin(self, name):
		self.start_time[name] = time.time()

	def TimerEnd(self, name):
		if list(self.measure_time.keys()).count(name) == 0:
			self.measure_time[name] = 0
		calc_time = time.time() - self.start_time[name]
		self.measure_time[name] = self.measure_time[name] + calc_time