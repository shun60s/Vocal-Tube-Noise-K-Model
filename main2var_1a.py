#coding:utf-8

#
# variable two tube model, draw waveform, considering glottal voice source and mouth radiation
#                          save generated waveform as a wav file

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy.io.wavfile import write as wavwrite
from twotubevariable import *
from glottal_fade_in import *  # use fade-in glottal pulse
from HPF import *
from tube_1A import *


# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1
#  scipy 1.0.0


def plot_freq_res(twotube, label, glo, hpf):
	plt.xlabel('Hz')
	plt.ylabel('dB')
	plt.title(label)
	amp0, freq=glo.H0(freq_high=5000, Band_num=256)
	amp1, freq=twotube.H0(freq_high=5000, Band_num=256)
	amp2, freq=hpf.H0(freq_high=5000, Band_num=256)
	plt.plot(freq, (amp0+amp1+amp2))

def plot_waveform(twotube, label, glo, hpf, repeat_num=5):
	# you can get longer input source to set bigger repeat_num 
	yg_repeat=glo.make_N_repeat(repeat_num) # input source of two tube model
	y2tm=twotube.process(yg_repeat)
	yout=hpf.iir1(y2tm)
	plt.xlabel('mSec')
	plt.ylabel('level')
	plt.title( label )
	plt.plot( (np.arange(len(yout)) * 1000.0 / glo.sr) , yout)
	return yout

def erase_head(yin, glo, n_cycle=1):
	# erase initial transient head
	
	tclosed= glo.tclosed # 5.0 msec
	trise=   glo.trise   # 6.0 msec
	tfall=   glo.tfall   # 2.0 msec
	sr=      glo.sr # sampling_rate # 48000
	
	one_cycle= tclosed + trise + tfall
	one_cycle_steps= int((one_cycle / 1000.0) * sr)
	erase_steps= one_cycle_steps * n_cycle
	print ('one cycle step ', one_cycle_steps)
	print ('erase initial portion of time [msec]', (erase_steps / sr ) * 1000.0)
	
	if len(yin) > erase_steps:
		yout=np.copy( yin[ erase_steps:])
	else:
		print ('ERROR: data size is too small. def erase_head in main2var_ua.py')
		yout= np.zeros(1)
	
	return yout, erase_steps
	
def append_zero_data(yi, append_zero_data_length=100, sampling_rate=48000):
	# input append_zero_data_length is append zero time duration to RESP0, unit is [msec]
	append_zero_data_n0= int( (append_zero_data_length / 1000.0) * sampling_rate)
	if append_zero_data_n0 > 0:
		y0= np.zeros(append_zero_data_n0)
	return np.concatenate( (y0, yi, y0) )

def save_wav( yout, wav_path, sampling_rate=48000):
	wavwrite( wav_path, sampling_rate, ( yout * 2 ** 15).astype(np.int16))
	print ('save ', wav_path) 

if __name__ == '__main__':
	
	# instance
	A=Class_UA(transient_time=50, tc=0.1)  # simulated opening mouth from one tube to /a/
	A.draw_cross_section_area()
	A.f_show_all()
	
	# instance variable two tube model
	tube=Class_TwoTube_variable( A )
	
	glo=Class_Glottal(fade_in_cycle=3, tc=1.0)   # instance as glottal voice source  fade-in
	# show an example ( Later, repeat_num will be change)
	glo.make_N_repeat(repeat_num=4)
	glo.plot_waveform()
	
	hpf=Class_HPF()       # instance for mouth radiation effect
	
	# vary erase head cycle
	n_cycle_list=[2, 1]  #
	for i in range (len(n_cycle_list)):
		
		# draw
		fig = plt.figure()
		#
		plt.subplot(2,1,1)
		yout_ua=plot_waveform(tube, 'Waveform toward /A/ (using green portion: 1a' + str(n_cycle_list[i]) + ' )', glo, hpf, repeat_num=10)
		
		# remove initial transient portion per pitch period
		yout_ua2, erase_steps= erase_head(yout_ua, glo, n_cycle= n_cycle_list[i])
		plt.plot( ((np.arange(len(yout_ua2)) + + erase_steps) * 1000.0 / glo.sr) , yout_ua2,  color='g')
		save_wav(yout_ua2, 'yout_1a' + str(n_cycle_list[i]) + '_var.wav')  # save generated wa + 'veform as a wav file
		#
		yout_ua2_zero=append_zero_data(yout_ua2)
		save_wav(yout_ua2_zero, 'yout_1a' + str(n_cycle_list[i]) + '_var_long.wav')  # save generated waveform as a wav file to listen
		
		fig.tight_layout()
		plt.show()
	
#This file uses TAB

