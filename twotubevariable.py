#coding:utf-8

#
# variable Two Tube Model
#

import math
import numpy as np
from matplotlib import pyplot as plt

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 


class Class_TwoTube_variable(object):
	def __init__(self, tube_x, rg0=0.95, rl0=0.9 ,sampling_rate=48000):
		# initalize
		self.total_tube_length = tube_x.total_tube_length
		self.sr= sampling_rate # for precision computation, higher sampling_rate is better
		C0=35000.0  # speed of sound in air, round 35000 cm/second
		self.tu= self.total_tube_length / C0   # whole delay time
		self.tu1= tube_x.L1 / C0
		self.tu2= tube_x.L2 / C0
		self.M= round( self.tu * self.sr )
		print ('total_tube_delay_points', self.M)
		self.M1= np.round((tube_x.L1 / C0) * self.sr)
		self.M2= self.M - self.M1
		#print ('self.M1', self.M1)
		#print ('self.M2', self.M2)
		self.r1=( tube_x.A2 - tube_x.A1) / ( tube_x.A2 + tube_x.A1)  # reflection coefficient between 1st tube and 2nd tube
		# print (self.r1)
		self.rg0=rg0 # rg is reflection coefficient between glottis and 1st tube
		self.rl0=rl0 # reflection coefficient between 3rd tube and mouth
		
	def fone(self, xw):
		# calculate frequecny response at target position (last position only)
		tu1=self.tu1[-1]
		tu2=self.tu2[-1]
		r1=self.r1[-1]
		yi= 0.5 * ( 1.0 + self.rg0 ) * ( 1.0 + r1)  * ( 1.0 + self.rl0 ) * np.exp( -1.0j * ( tu1 + tu2 ) * xw) 
		yb= 1.0 + r1 * self.rg0 *  np.exp( -2.0j * tu1 * xw )  + r1 * self.rl0 * np.exp( -2.0j * tu2 * xw )  + self.rl0 * self.rg0 * np.exp( -2.0j * (tu1 + tu2) * xw )
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
			amp.append(self.fone(f * 2.0 * np.pi))
		return   np.log10(amp) * 20, bands # = amp value, freq list

	def process(self, yg ):
		# process reflection transmission of resonance tube: yg is input, y2tm is output
		# two serial resonance tube
		#                         -------------------------
		#                         |                        |
		#   ----------------------                         |
		#   |                                              |
		#   |                                              |
		#   ----------------------                         |
		#                         |                        |
		#                         -------------------------
		# reflection ratio
		#   rg                    r1                       rl0
		#   [0]----(forward)--[M1][M1+1]------(forward)---[M]
		#   [M]----(backward)-[M-M1+1][M-M1]--(backward)--[0]
		# input yg                                 output y2tm
		# 
		#
		
		if len(yg) > len(self.M1):
			print ('warning: Tube duration is less than yg duration. Data will be extended as target position(last position).') 
			y2tm=np.zeros(len(yg))
		else:
			print ('warning: Yg duration is less than Tube duration. Only Yg duration portion will be computed.')
			print ('Yg duration, Tube duration ', len(yg), len(self.M1) )
			y2tm=np.zeros(len(yg))
		
		FWD0= np.zeros(self.M+1)  # FWD0[0] is present, FWD0[1] is 1 delay, ..., FWD0[M+1] is whole delyed
		FWD1= np.zeros(self.M+1)
		BWD0= np.zeros(self.M+1)
		BWD1= np.zeros(self.M+1)
		
		for tc0 in range(len(y2tm)):
			# process one step ahead
			FWD1= np.roll(FWD0,1)
			BWD1= np.roll(BWD0,1)
			
			if tc0 < len(self.M1):
				m1=int(self.M1[tc0])
				m2=int(self.M - self.M1[tc0])
				r1=self.r1[tc0]
			else: # if yg is longer than tube duration, rest portion is set to value at target position (last one).
				m1=int(self.M1[-1])
				m2=int(self.M - self.M1[-1])
				r1=self.r1[-1]
			
			# compute reflection
			FWD1[0]= ((1. + self.rg0 ) / 2.) * yg[tc0] + self.rg0 * BWD0[-1]
			#ya1[0]= ((1. + self.rg0 ) / 2.) * yg[tc0] + self.rg0 * ya2[-1]
			
			if (m1 >0 and  m1 < self.M)  or (m2 >0 and  m2 < self.M):
				
				BWD1[m2+1]= -1. * r1 *  FWD0[m1]  +  ( 1. - r1 ) * BWD0[m2]
				#ya2[0]= -1. * self.r1 *  ya1[-1]  +  ( 1. - self.r1 ) * yb2[-1]
				
				FWD1[m1+1] = ( 1 + r1 ) * FWD0[m1] + r1 * BWD0[m2]
				#yb1[0]= ( 1 + self.r1 ) * ya1[-1] + self.r1 * yb2[-1]
			
			BWD1[0]=  -1. * self.rl0  * FWD0[-1]
			#yb2[0]=  -1. * self.rl0  * yb1[-1]
			
			y2tm[tc0]= (1 + self.rl0) * FWD0[-1]
			#y2tm[tc0]= (1 + self.rl0) * yb1[-1]
			
			# load FWD1/BWD1 to FWD0/BWD0 
			FWD0 = np.copy(FWD1)
			BWD0 = np.copy(BWD1)

		return y2tm

if __name__ == '__main__':
	
	from tube_A1 import *
	
	# insatnce
	A=Class_A(transient_time=30)
	# A.f_show_all()
	tube=Class_TwoTube_variable( A )
	
	# draw
	fig = plt.figure()
	# draw frequecny response
	plt.subplot(2,1,1)
	plt.xlabel('Hz')
	plt.ylabel('dB')
	plt.title('frequecny response at target')
	amp, freq=tube.H0(freq_high=5000, Band_num=256)
	plt.plot(freq, amp)
	#
	fig.tight_layout()
	plt.show()
	
#This file uses TAB

