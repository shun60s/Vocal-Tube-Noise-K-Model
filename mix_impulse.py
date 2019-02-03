#coding:utf-8

# mix blast impulse waveform with noise waveform
#

import os
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from matplotlib import pyplot as plt
from blast_impulse import *

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1
#  scipy 1.0.0

def load_wav(path0):
	if os.path.isfile(path0):
		sr, y = wavread(path0)
		y = y / (2 ** 15)
		print ('loading , sampling rate ', path0, sr)
		return y
	else:
		print ('error: file is not exist', path0)
		return None
	
def save_wav( yout, wav_path, sampling_rate=48000):
	wavwrite( wav_path, sampling_rate, ( yout * 2 ** 15).astype(np.int16))
	print ('save ', wav_path) 
	
def plot_waveform(y1, label1, y2, label2, sampling_rate=48000):
	fig = plt.figure()
	plt.subplot(2,1,1)
	plt.xlabel('mSec')
	plt.ylabel('level')
	plt.title( label1+ '(blue)' + ' and  ' + label2 + '(green)' )
	plt.plot( (np.arange(len(y1)) * 1000.0 / sampling_rate) , y1)
	#plt.subplot(2,1,2)
	#plt.xlabel('mSec')
	#plt.ylabel('level')
	#plt.title( label2 )
	plt.plot( (np.arange(len(y2)) * 1000.0 / sampling_rate) , y2, color='g')
	fig.tight_layout()
	plt.show()


def mix_wav(y1, yi, ratio=1.0):
	# multiply yi to its maximum value will be same as y1 maximum value multiplied by the ratio.
	# add it to y1
	# return length is shorter one
	len0= np.amin( (len(y1), len(yi)) )
	y1_max0= np.max( np.abs(y1[0:len0]) )
	yi_max0= np.max( np.abs(yi[0:len0]) )
	if yi_max0 > 0.0:
		factor0= (y1_max0 * ratio) / yi_max0
	else:
		factor0=1.0
	print ('multiply value ', factor0)
	yout=np.zeros(len0)
	for i in range( len0 ):
		yout[i]=y1[i]+ yi[i] * factor0
	
	return yout

def mix_wav_length(y1, yi, length=30, ratio=1.0, sampling_rate=48000):
	# return length [msec], if y1 or yi is less than length, append zero data.
	# multiply yi to its maximum value will be same as y1 maximum value multiplied by the ratio.
	# add it to y1
	nstep0= int((length / 1000.0) * sampling_rate)
	
	if len(y1) >= nstep0:
		y1b= np.copy(y1[0:nstep0])
	else:
		y1b= np.concatenate(y1, np.zeros( nstep0 - len(y1)) )
	
	if len(yi) >= nstep0:
		yib= np.copy(yi[0:nstep0])
	else:
		y0= np.zeros( int(nstep0 - len(yi)))
		yib= np.concatenate((yi, y0)  )
	
	#
	len0= np.amin( (len(y1b), len(yib)) )
	y1_max0= np.max( np.abs(y1b[0:len0]) )
	yi_max0= np.max( np.abs(yib[0:len0]) )
	if yi_max0 > 0.0:
		factor0= (y1_max0 * ratio) / yi_max0
	else:
		factor0=1.0
	print ('multiply value ', factor0)
	yout=np.zeros(len0)
	for i in range( len0 ):
		yout[i]=y1b[i]+ yib[i] * factor0
	
	return yout, factor0
	
def make_zero(duration, sampling_rate=48000):
	# duration unit is [msec]
	return np.zeros( int((duration / 1000.) * sampling_rate) )

if __name__ == '__main__':
	
	import itertools
	
	# blast impulse duration list
	blast_impulse_duration= [40] # [40, 50]
	start_frequency= [800]  # [600, 800]
	Perlin_Noise_no=[0] # [0, 1]
	clist=list(itertools.product(blast_impulse_duration,start_frequency, Perlin_Noise_no ))
	print (clist)
	
	for i in range(len(clist)):
		# instance blast impulse
		len0=clist[i][0]
		f_start0=clist[i][1]
		PerlinNoise_no0=clist[i][2]
		
		# load noise source
		yn=load_wav( 'k_noise' + str(PerlinNoise_no0) + '.wav' )
		if yn is None: # if the file is not exist, skip 
			continue
		
		impulse1=Class_BlastImpulse(length=len0, f_start=f_start0)
		yi=np.copy(impulse1.RESP0)
		
		impulse1.plot_waveform()
		impulse1.f_show()
		
		impulse1.append_zero_data(100) #add zero data to the front and rear
		impulse1.save_wav() # save as a sample wav file to hear
		
		# mix blast impulse with noise
		y2, factor=mix_wav_length(yn, yi, length= len0 , ratio= 2.0)   #1.0)
		plot_waveform(y2, 'mixed waveform', yi * factor, 'blast impluse portion')
		
		# save as a wav file
		save_wav(y2, 'k_noise' + str(PerlinNoise_no0)  + '_i' + str(len0)  + '_s' + str(f_start0) + '.wav')
		
		# make long version to listen
		yz10=make_zero(10)
		yz150=make_zero(150)
		y2_long= np.concatenate( (yz10, y2,  yz150 ) )
		save_wav(y2_long, 'k_noise' + str(PerlinNoise_no0)  + '_i' + str(len0)  + '_s' + str(f_start0) + '_long.wav')
		
#This file uses TAB

