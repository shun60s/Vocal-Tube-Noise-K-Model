#coding:utf-8

# making pseudo /ga/ /ka/ sound to combine waveforms

import os
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from matplotlib import pyplot as plt

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
	
def make_zero(duration, sampling_rate=48000):
	# duration unit is [msec]
	return np.zeros( int((duration / 1000.) * sampling_rate) )
	
def rfade_out(xin, duration=10, sampling_rate=48000):
	n= int((duration / 1000.) * sampling_rate)
	l0=len(xin)
	yout=np.copy(xin)
	if len(xin) > n:
		for i in range(n):
			yout[l0-1-i] = xin[l0-1-i] * ((1.0 * i ) / n)
	return yout
	
def ltrim(xin, thres=1.0):
	l0=len(xin)
	icode=0
	for i in range(l0):
		if xin[i] > (thres / (2 ** 15)):
			icode=i
			break
	yout=np.copy(xin[icode:])
	return yout
	
def plot_waveform(y1, label1, y2=None, label2=None, sampling_rate=48000):
	fig = plt.figure()
	plt.subplot(2,1,1)
	plt.xlabel('mSec')
	plt.ylabel('level')
	plt.title( label1 )
	plt.plot( (np.arange(len(y1)) * 1000.0 / sampling_rate) , y1)
	
	if y2 is not None:
		plt.subplot(2,1,2)
		plt.xlabel('mSec')
		plt.ylabel('level')
		plt.title( label2 )
		plt.plot( (np.arange(len(y2)) * 1000.0 / sampling_rate) , y2)
	
	fig.tight_layout()
	plt.show()

def mix_wav(y1, yi, ratio=0.5):
	# yiをy1のratio比率で加算して、短い方の長さで返す
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

def amp_wav(y1, ya, ratio=0.5):
	# y1をyaのratio比率まで増幅する
	y1_max0= np.max( np.abs(y1) )
	ya_max0= np.max( np.abs(ya) )
	if y1_max0 > 0.0:
		factor0= (ya_max0 * ratio) / y1_max0
	else:
		factor0=1.0
	
	yout=np.zeros( len(y1) )
	for i in range( len(y1)):
		yout[i]=y1[i] * factor0
	
	return yout

if __name__ == '__main__':
	
	import itertools
	
	# load k noise
	blast_impulse_duration=[40]
	start_frequency=[800]
	resona_no=[0] # [0, 1, 2, 3]
	Perlin_Noise_no=[0] # [0, 1]
	n_cycle_list=[2, 1] #
	
	clist=list(itertools.product(blast_impulse_duration,start_frequency,resona_no, Perlin_Noise_no, n_cycle_list ))
	print (clist)
	
	for i in range(len(clist)):
		
		y1=load_wav('k_noise' + str(clist[i][3]) + '_i' + str(clist[i][0]) + '_s' +  str(clist[i][1]) + '_resona_' + str(clist[i][2]) +  '.wav')
		if y1 is None: # if the file is not exist, skip 
			continue
		
		# load  a portion of simulated opening mouth to /a/
		ya=load_wav( 'yout_1a' + str(clist[i][4]) + '_var.wav' )
		if ya is None: # if the file is not exist, skip 
			continue
		
		#
		y2=amp_wav(y1, ya, ratio=1.0)
		
		y3fadeout=rfade_out(y2, duration=3)
		#plot_waveform(y1, 'input', yi,  'impluse')
		
		# repeat blast waveform to emphasize blast effect
		#y3fadeout_repeats=np.concatenate((y3fadeout,y3fadeout))
		
		yaltrim= ltrim(ya)
		#plot_waveform(y2, 'input', y2ltrim,  'ltrim')
		
		# start zero data 
		yz100=make_zero(100)
		
		# combining
		yout= np.concatenate( (yz100, y3fadeout, yaltrim,  yz100 ) )
		plot_waveform(yout, 'Waveform combined ' + '(1a' + str(clist[i][4]) + ')')
		
		# save as a wav file
		save_wav(yout, 'gka_1a' + str(clist[i][4]) + '_noise' + str(clist[i][3]) + '_i' + str(clist[i][0]) + '_s' +  str(clist[i][1]) + '_resona_' + str(clist[i][2]) + '.wav')
		

#This file uses TAB

