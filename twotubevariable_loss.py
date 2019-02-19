#coding:utf-8

# variable Two Tube Model
# use passing loss of tube, to weak resonance effect
# it's assumed that loss depends on cross-section length

import math
import numpy as np
from matplotlib import pyplot as plt

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 


class Class_TwoTube_variable_loss(object):
	def __init__(self, tube_x, rg0=0.95, rl0=0.9 , passing_loss_ratio = 0.01, standard_area=3.0,  sampling_rate=48000):
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
		self.tube_A1= tube_x.A1
		self.tube_A2= tube_x.A2
		
		# print (self.r1)
		self.rg0=rg0 # rg is reflection coefficient between glottis and 1st tube
		self.rl0=rl0 # reflection coefficient between 3rd tube and mouth
		
		# passing loss of tube
		self.loss_ratio = passing_loss_ratio  # no loss when loss ratio is 0.0. It means passing loss ratio per one sampling point
		self.standard_area= standard_area     # no effcet when cross- section area is equal to standard_area
		
		
	def fone(self, xw):
		# calculate frequecny response at target position (last position only) when loss is zero
		tu1=self.tu1[-1]
		tu2=self.tu2[-1]
		r1=self.r1[-1]
		A1_loss_factor= self.get_loss_factor(self.tube_A1[-1])
		A2_loss_factor= self.get_loss_factor(self.tube_A2[-1])
		beta1=np.power(A1_loss_factor , self.M1[-1])
		beta2=np.power(A2_loss_factor , self.M2[-1])
		#print ('beta1, beta2', beta1, beta2)
		yi= 0.5 * ( 1.0 + self.rg0 ) * ( 1.0 + r1)  * ( 1.0 + self.rl0 ) * np.exp( -1.0j * ( tu1 + tu2 ) * xw) * beta1 * beta2
		yb= 1.0 + r1 * self.rg0 *  np.exp( -2.0j * tu1 * xw ) * beta1  + r1 * self.rl0 * np.exp( -2.0j * tu2 * xw ) * beta2 + self.rl0 * self.rg0 * np.exp( -2.0j * (tu1 + tu2) * xw ) * beta1 * beta2
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
		
		self.A1_loss_list= np.zeros(len(y2tm))  # loss per A1,L1 tube at every tc0
		self.A2_loss_list= np.zeros(len(y2tm))  # loss per A2,L2 tube at every tc0
		
		for tc0 in range(len(y2tm)):
			# process one step ahead
			FWD1= np.roll(FWD0,1)
			BWD1= np.roll(BWD0,1)
			
			# m1 1st tube length, m2 2nd tube length
			if tc0 < len(self.M1):
				m1=int(self.M1[tc0])
				m2=int(self.M - self.M1[tc0])
				r1=self.r1[tc0]
				A1=self.tube_A1[tc0]
				A2=self.tube_A2[tc0]
			else: # if yg is longer than tube duration, rest portion is set to value at target position (last one).
				m1=int(self.M1[-1])
				m2=int(self.M - self.M1[-1])
				r1=self.r1[-1]
				A1=self.tube_A1[-1]
				A2=self.tube_A2[-1]
			
			# apply loss factor
			A1_loss_factor= self.get_loss_factor(A1)
			A2_loss_factor= self.get_loss_factor(A2)
			
			self.A1_loss_list[tc0]= 1.0 - np.power( A1_loss_factor, m1)
			self.A2_loss_list[tc0]= 1.0 - np.power( A1_loss_factor, m2)
			
			
			for i in range(len(FWD1)):
				if i < m1:
					FWD1[i]= FWD1[i] * A1_loss_factor  # in A1
				else:
					FWD1[i]= FWD1[i] * A2_loss_factor  # in A2
			
			for i in range(len(BWD1)):
				if i < m2:
					BWD1[i]= BWD1[i] * A2_loss_factor  # in A2
				else:
					BWD1[i]= BWD1[i] * A1_loss_factor  # in A1
			
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

	def get_loss_factor(self, xin_area):
		# no effcet when cross- section area is equal to standard_area 
		return 1.0 - (self.loss_ratio *  np.sqrt(self.standard_area) / np.sqrt(xin_area))  # get length dim via sqrt area

if __name__ == '__main__':
	
	# instance
	from tube_1A import *
	A=Class_neutral2A( transient_time=50, start_hold_time=13, tc=0.1)  # simulation of mouth movement for /ra/
	#A.draw_cross_section_area()
	#A.f_show_all()
	
	# instance variable two tube model
	tube0=Class_TwoTube_variable_loss( A, passing_loss_ratio = 0.0)
	amp0, freq=tube0.H0(freq_high=5000, Band_num=256)
	
	tube1=Class_TwoTube_variable_loss( A, passing_loss_ratio = 0.01)
	amp1, freq=tube1.H0(freq_high=5000, Band_num=256)
	
	# draw
	fig = plt.figure()
	plt.xlabel('Hz')
	plt.ylabel('dB')
	plt.title('two tube frequency response (passing_loss_ratio, green: 0, red: 0.01)')
	plt.plot(freq, amp0,color='g')
	plt.plot(freq, amp1,color='r')
	fig.tight_layout()
	plt.show()
	
#This file uses TAB
