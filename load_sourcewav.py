#coding:utf-8

# load wav file (16bit mono) as source

import os
import numpy as np
from scipy.io.wavfile import read as wavread
from matplotlib import pyplot as plt

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1
#  scipy 1.0.0

class Class_WavSource(object):
	def __init__(self, path0):  # , sampling_rate=48000):
		# initalize
		
		if os.path.isfile(path0):
			sr, y = wavread(path0)
			self.yg= y / (2 ** 15)
			self.sr= sr
			print ('sampling rate ', sr)
			self.rtcode= True
		else:
			print ('error: file is not exist', path0)
			self.rtcode= False
	
	def make_N_repeat(self, repeat_num=1):
		yg_repeat=np.zeros( len(self.yg) * repeat_num)
		for loop in range( repeat_num):
			yg_repeat[len(self.yg)*loop:len(self.yg)*(loop+1)]= self.yg
		return  yg_repeat
	
	def fone(self, f):
		# calculate one point of frequecny response
		xw= 2.0 * np.pi * f / self.sr
		yi=0.0
		yb=0.0
		for v in range (len(self.yg)):
			yi+=  self.yg[v] * np.exp(-1j * xw * v)
			yb+=  self.yg[v]
		val= yi/yb
		return np.sqrt(val.real ** 2 + val.imag ** 2)
	
	def H0(self, freq_low=100, freq_high=5000, Band_num=256):
		# get Log scale frequecny response, from freq_low to freq_high, Band_num points
		amp=[]
		freq=[]
		bands= np.zeros(Band_num+1)
		fcl=freq_low * 1.0    # convert to float
		fch=freq_high * 1.0   # convert to float
		delta1=np.power(fch/fcl, 1.0 / (Band_num)) # Log Scale
		bands[0]=fcl
		#print ("i,band = 0", bands[0])
		for i in range(1, Band_num+1):
			bands[i]= bands[i-1] * delta1
			#print ("i,band =", i, bands[i]) 
		for f in bands:
			amp.append(self.fone(f) )
		return   np.log10(amp) * 20, bands # = amp value, freq list

if __name__ == '__main__':
	
	# instance
	glo=Class_WavSource('s_noise.wav')

	# draw
	fig = plt.figure()
	# draw one waveform
	plt.subplot(3,1,1)
	plt.xlabel('mSec')
	plt.ylabel('level')
	plt.title('Waveform')
	plt.plot( (np.arange(len(glo.yg)) * 1000.0 / glo.sr) , glo.yg)
	
	# draw frequecny response
	plt.subplot(3,1,2)
	plt.xlabel('Hz')
	plt.ylabel('dB')
	plt.title('frequecny response')
	amp, freq=glo.H0(freq_high=5000, Band_num=256)
	plt.plot(freq, amp)
	
	# draw repeated waveform
	yg_repeat=glo.make_N_repeat(repeat_num=1)
	plt.subplot(3,1,3)
	plt.xlabel('mSec')
	plt.ylabel('level')
	plt.title('repeated Waveform')
	plt.plot( (np.arange(len(yg_repeat)) * 1000.0 / glo.sr) , yg_repeat)
	#
	fig.tight_layout()
	plt.show()
	
#This file uses TAB

