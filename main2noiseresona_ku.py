#coding:utf-8

#
# apply two tube model resonance effect to noise sound
#

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy.io.wavfile import write as wavwrite

from twotube import *
from load_sourcewav import *
from HPF import *

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1
#  scipy 1.0.0


def plot_freq_res(twotube, label, glo, hpf):
	plt.xlabel('Hz')
	plt.ylabel('dB')
	plt.title(label)
	amp0, freq=glo.H0(freq_high=10000, Band_num=512)
	amp1, freq=twotube.H0(freq_high=10000, Band_num=512)
	amp2, freq=hpf.H0(freq_high=10000, Band_num=512)
	plt.plot(freq, (amp0+amp1+amp2))

def plot_waveform(twotube, label, glo, hpf):
	# you can get longer input source to set bigger repeat_num 
	yg_repeat=glo.make_N_repeat(repeat_num=1) # input source of two tube model
	y2tm=twotube.process(yg_repeat)
	yout=hpf.iir1(y2tm)
	plt.xlabel('mSec')
	plt.ylabel('level')
	plt.title(label)
	plt.plot( (np.arange(len(yout)) * 1000.0 / glo.sr) , yout)
	return yout

def save_wav( yout, wav_path, sampling_rate=48000):
	wavwrite( wav_path, sampling_rate, ( yout * 2 ** 15).astype(np.int16))
	print ('save ', wav_path) 
	
def f_show( twotube, i):
	plt.subplot(MAX_PATTERNS, 1 ,i+1)
	#plt.subplot(MAX_PATTERNS * 2, 1 ,int(2*i+1))
	yout=plot_waveform(twotube, 'Noise waveform with resonance (resona_' + str(i) + ')', glo, hpf)
	#plt.subplot(MAX_PATTERNS * 2 ,1, int(2*i+2))
	#plot_freq_res(twotube, 'Frequency response (whole span) '+str(i), glo, hpf)
	return yout
	
def get_A2( A1, r1):
		# return cross section area A2 to meet the reflection ratio r1
		return ((1.0 + r1)/(1.0 - r1)) * A1

def make_zero(duration, sampling_rate=48000):
	# duration unit is [msec]
	return np.zeros( int((duration / 1000.) * sampling_rate) )

if __name__ == '__main__':
	
	import itertools
	
	MAX_PATTERNS=2 #1  # number of display patterns
	
	# set some initial value
	"""
	# /u/ reference
	L1_u=10.0   # set list of 1st tube's length by unit is [cm]
	A1_u=7.0    # set list of 1st tube's area by unit is [cm^2]
	L2_u=7.0    # set list of 2nd tube's length by unit is [cm]
	A2_u=3.0    # set list of 2nd tube's area by unit is [cm^2]
	r1= -0.4
	"""
	L1=np.ones(MAX_PATTERNS) * 10.0
	A1=np.ones(MAX_PATTERNS) * 7.0
	L2=np.ones(MAX_PATTERNS) * 7.0
	A2=np.ones(MAX_PATTERNS) * 3.0
	
	# set cross section area to specify reflection ratio r1
	for i in range(MAX_PATTERNS): 
		A2[i]= get_A2( A1[i], -0.4)  # neutral (r1=-0.4)
	
	# set varied tube's length, (varied resonance frequecny)
	L1_list=[ 7.0, 7.0, 7.0, 7.0]
	L2_list=[ 1.5, 2.5, 3.5, 4.5]
	for i in range(MAX_PATTERNS): 
		L1[i]= L1_list[i]
		L2[i]= L2_list[i]
	
	# load a wav file as source source
	blast_impulse_duration=[40]  # [40, 50]
	start_frequency=[600]  # [600, 800]
	Perlin_Noise_no=[0] # [0, 1]
	clist=list(itertools.product(blast_impulse_duration,start_frequency,Perlin_Noise_no ))
	print (clist)
	
	for i in range(len(clist)):
		# instance blast impulse
		len0=clist[i][0]
		f_start0=clist[i][1]
		PerlinNoisen_no0=clist[i][2]
		
		glo=Class_WavSource('k_noise' + str(PerlinNoisen_no0) + '_i' + str(len0)  + '_s' + str(f_start0) + '.wav') # instance noise sound
		if not glo.rtcode: # if the file is not exist, skip 
			continue
		
		hpf=Class_HPF()       # instance for mouth radiation effect
		
		# draw
		fig = plt.figure()
		
		yz10=make_zero(10)
		yz150=make_zero(150)
		# 
		for j in range(MAX_PATTERNS): 
			twotube=  Class_TwoTube(L1[j],L2[j],A1[j],A2[j])
			print ('r1, L1, L2', twotube.r1, L1[j], L2[j])
			yout=f_show( twotube, j)
			
			wavname='k_noise' + str(PerlinNoisen_no0) +  '_i' + str(len0) + '_s' +  str(f_start0) + '_resona_' + str(j)+ '.wav'
			save_wav(yout, wavname)  # save generated waveform as a wav file
			
			# make long version to listen
			yout_long= np.concatenate( (yz10, yout,  yz150 ) )
			wavname='k_noise' + str(PerlinNoisen_no0) +  '_i' + str(len0) + '_s' +  str(f_start0) + '_resona_' + str(j)+ '_long.wav'
			save_wav(yout_long, wavname)  # save generated waveform as a wav file
		
		fig.tight_layout()
		plt.show()
	
	
#This file uses TAB

