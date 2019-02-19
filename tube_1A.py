#coding:utf-8

#  simulated opening mouth from neutral state to /a/
#  Two Tube model definition, to specify start status and target status
#
#  After start_hold_time (initial resonance time) at first time, initial start status will be kept.
#  
#  Prerequisite:
#     Total length of tubes, that is L1 + L2, must be always a fixed value.

import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1

class Class_neutral2A(object):
	def __init__(self, target=None, start=None, transient_time=50, start_hold_time=13,  sampling_rate=48000, tc=0.1):
		# if transient_time is too long, hear as independend phonemes combonation
		# initalize
		if target is None:
			# /a/
			# target
			L1_a_t=9.0    # set list of 1st tube's length by unit is [cm]
			A1_a_t=2.0 #1.0    # set list of 1st tube's area by unit is [cm^2]
			L2_a_t=8.0    # set list of 2nd tube's length by unit is [cm]
			A2_a_t=self.get_A2( A1_a_t, 0.75)
			target=[L1_a_t, A1_a_t, L2_a_t, A2_a_t]
			
		if start is None:
			# start neutral(r1=-0.4) and low resonance effect(A1=0.5, passing_loss_ratio=0.01) for unvoiced sound
			L1_a_s=9.0    # set list of 1st tube's length by unit is [cm]
			A1_a_s=0.5   # set list of 1st tube's area by unit is [cm^2]
			L2_a_s=8.0    # set list of 2nd tube's length by unit is [cm]
			A2_a_s= self.get_A2( A1_a_s, -0.4) 
			start= [L1_a_s, A1_a_s, L2_a_s, A2_a_s]
		
		self.sampling_rate= sampling_rate
		self.transient_time= transient_time # unit [msec]
		self.start_hold_time= start_hold_time # unit [msec]
		self.tc= tc # rise shape factor of exp acceleration
		self.transient_steps= round( self.sampling_rate * ( self.transient_time / 1000.0) )
		self.start_hold_steps= round( self.sampling_rate * ( self.start_hold_time / 1000.0) )
		self.total_tube_length = target[0] + target[2]
		if self.total_tube_length != (start[0] + start[2]):
			print (' error: Total length is not a fixed value')
			sys.exit()
		
		A1_start_hold= np.ones(self.start_hold_steps) * A1_a_s
		A2_start_hold= np.ones(self.start_hold_steps) * A2_a_s
		L1_start_hold= np.ones(self.start_hold_steps) * L1_a_s
		L2_start_hold= np.ones(self.start_hold_steps) * L2_a_s
		
		A1_transient= self.exp_x(start[1], target[1], self.transient_steps)
		A2_transient= self.exp_x(start[3], target[3], self.transient_steps)
		L2_transient= self.exp_x(start[2], target[2], self.transient_steps)
		L1_transient= np.ones(self.transient_steps) * (target[0] + target[2])
		L1_transient= L1_transient - L2_transient
		
		self.A1= np.concatenate( (A1_start_hold, A1_transient))
		self.A2= np.concatenate( (A2_start_hold, A2_transient))
		self.L1= np.concatenate( (L1_start_hold, L1_transient))
		self.L2= np.concatenate( (L2_start_hold, L2_transient))
		
		
	def get_A2(self,  A1, r1):
		# return cross section area A2 to meet the reflection ratio r1
		return ((1.0 + r1)/(1.0 - r1)) * A1
		
	def linear_x(self, s, t, steps):
		# linear interpolation
		x=np.linspace(s, t, num=steps)
		return x
		
	def exp_x(self, s, t, steps ):
		# 1-exp(-t) curve interpolation
		x=np.linspace(0, 1.0, num=steps)
		return  (t - s) * ( 1.0 -  np.exp( -self.tc * x)) / (1.0 - np.exp(- self.tc)) + s
		
	def f_show_all(self,):
		# show x curve
		fig = plt.figure()
		
		plt.subplot(4,1,1)
		plt.title( 'A1' )
		plt.xlabel('mSec')
		#plt.ylabel('level')
		plt.plot( np.arange(len(self.A1)) / (self.sampling_rate / 1000.0)  , self.A1)
		
		plt.subplot(4,1,2)
		plt.title( 'A2' )
		plt.xlabel('mSec')
		#plt.ylabel('level')
		plt.plot( np.arange(len(self.A2)) / (self.sampling_rate / 1000.0) , self.A2)
		
		plt.subplot(4,1,3)
		plt.title( 'L1' )
		plt.xlabel('mSec')
		#plt.ylabel('level')
		plt.plot( np.arange(len(self.L1)) / (self.sampling_rate / 1000.0) , self.L1)
		
		plt.subplot(4,1,4)
		plt.title( 'L2' )
		plt.xlabel('mSec')
		#plt.ylabel('level')
		plt.plot( np.arange(len(self.L2)) / (self.sampling_rate / 1000.0) , self.L2)
		
		fig.tight_layout()
		plt.show()
		
	def draw_cross_section_area(self,):
		# draw cross_section_area vs length at start[0] and target[-]1
		fig = plt.figure()
		# start
		plt.subplot(2,1,1)
		ax1=fig.add_subplot(2,1,1)
		plt.title( 'start tube shape' )
		plt.xlabel('length')
		plt.ylabel('cross section area')
		ax1.add_patch( patches.Rectangle((0, -0.5* self.A1[0]), self.L1[0], self.A1[0], hatch='/', fill=False))
		ax1.add_patch( patches.Rectangle((self.L1[0], -0.5* self.A2[0]), self.L2[0], self.A2[0], hatch='/', fill=False, color='blue'))
		ax1.set_xlim([0, 20])
		ax1.set_ylim([-10, 10])
		# target(last)
		plt.subplot(2,1,2)
		ax1=fig.add_subplot(2,1,2)
		plt.title( 'target tube shape (Mouth open toward /a/)' )
		plt.xlabel('length')
		plt.ylabel('cross section area')
		ax1.add_patch( patches.Rectangle((0, -0.5* self.A1[-1]), self.L1[-1], self.A1[-1], hatch='/', fill=False))
		ax1.add_patch( patches.Rectangle((self.L1[-1], -0.5* self.A2[-1]), self.L2[-1], self.A2[-1], hatch='/', fill=False, color='blue'))
		ax1.set_xlim([0, 20])
		ax1.set_ylim([-10, 10])
		#
		fig.tight_layout()
		plt.show()
		
if __name__ == '__main__':
	
	# instance
	A=Class_neutral2A()
	
	# draw
	A.draw_cross_section_area()
	A.f_show_all()
	
	
#This file uses TAB


