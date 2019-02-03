#coding:utf-8

# generate pseudo blast impulse waveform which has 1/f spectrum and minimum phase by Hilbert transform
# 

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0
#  scipy 1.0.0
#  matplotlib 2.1.1

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io.wavfile import write as wavwrite


class Class_BlastImpulse(object):
    def __init__(self, length=20, f_start=600.0, sampling_rate=48000, max_value=0.2):
        # input length is compute time duration, unit is [msec]
        # f_start:  1/f starts fequecny.  Response is flat between DC and f_start[Hz]
        # initial
        self.length= length
        self.sampling_rate= sampling_rate
        if max_value >1.0:
            self.max_value=1.0
        else:
            self.max_value= max_value
        #
        n0= int((self.length / 1000.0) * self.sampling_rate)
        if n0 % 2 != 0:
            n0 +=1
        self.size_of_fft= n0  # compute length, must be an even number
        self.step= (1.0 *  self.sampling_rate / self.size_of_fft)
        print ('compute length number ', self.size_of_fft)
        print ('frequecny step [Hz]', self.step )
        
        self.start_step= int(f_start / self.step)
        # 1/f list
        self.H_list= np.ones(self.size_of_fft)  #  * (1.0 / self.start_step) # initial is flat
        #self.H_list[0]= 1.0  # H_list[0] is always set 1.0. DC[0] becomes 1/size_of_fft, due to ifft coefficient of DC[0] is 1/size_of_fft
        for i in range( self.start_step + 1 , int(self.size_of_fft/2)+1):
            self.H_list[i]= 1.0 / (i  - self.start_step + 1) # value = 1/f
            self.H_list[self.size_of_fft-i]=self.H_list[i]
        
        #for i in range( 0, self.start_step ):
        #    self.H_list[i] =self.H_list[i] / (self.start_step - i)
        
        
        # log and ifft
        lnH_list=np.log(self.H_list)
        IFFT0=np.fft.ifft(lnH_list)
        
        # h is like hilbert filter
        h= np.zeros(self.size_of_fft)
        h[0]=1.0
        h[int(self.size_of_fft/2)]=1.0        
        h[1:int(self.size_of_fft/2)]=2.0
        
        # get minimum phase response
        FFT0=np.fft.fft( IFFT0 * h )  # imag(FFT0) is  approximate minimum phase
        
        self.ComplexH_list=np.zeros(self.size_of_fft, dtype=np.complex)
        for i in range(self.size_of_fft):
            self.ComplexH_list[i]= self.H_list[i] * np.exp( 1j * np.imag(FFT0[i]))
        
        # get impulse from frequency response with minimum phase via ifft
        self.ComplexRESP0=np.fft.ifft(self.ComplexH_list) # complex impulse response
        self.RESP0=np.real(self.ComplexRESP0) # get only real part
        
        # normalize to max_value
        smax0= np.max( np.abs(self.RESP0) )
        self.RESP0 = self.RESP0 * (self.max_value / smax0)
        
    def append_zero_data(self, append_zero_data_length=0):
        # input append_zero_data_length is append zero time duration to RESP0, unit is [msec]
    	self.append_zero_data_n0= int( (append_zero_data_length / 1000.0) * self.sampling_rate)
    	if self.append_zero_data_n0 > 0:
    	    y0= np.zeros(self.append_zero_data_n0)
    	    self.RESP0= np.concatenate( (y0, self.RESP0, y0) )
        
    def plot_waveform(self,):
        # plot waveform
        fig = plt.figure()
        plt.xlabel('mSec')
        plt.ylabel('level')
        plt.title('blast impulse waveform')
        plt.plot( (np.arange(len(self.RESP0)) * 1000.0 / self.sampling_rate) , self.RESP0)
        fig.tight_layout()
        plt.show()
        
    def f_show(self,):
        # plot frequency response of impulse waveform
        BunsiZcoefs = self.RESP0
        BunboZcoefs = np.array([1.0])
        # 
        wlist, fres = signal.freqz(BunsiZcoefs, BunboZcoefs, worN=1024)
        flist = wlist / ((2.0 * np.pi) / self.sampling_rate)
        #
        fig = plt.figure()
        plt.title('frequency response')
        ax1 = fig.add_subplot(111)
        plt.semilogx(flist, 20 * np.log10(abs(fres)), 'b')  # plt.plot(flist, 20 * np.log10(abs(fres)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [Hz]')
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(fres))
        angles = angles / ((2.0 * np.pi) / 360.0)
        plt.semilogx(flist, angles, 'g')  # plt.plot(flist, angles, 'g')
        plt.ylabel('Angle(deg)', color='g')
        plt.grid()
        plt.axis('tight')
        plt.show()
        
    def save_wav(self, wav_path=None):
        if wav_path is None:
            wav_path = 'blast_impulse_' + str(self.length) + '.wav'
        wavwrite( wav_path, self.sampling_rate, ( self.RESP0 * 2 ** 15).astype(np.int16))
        print ('save ', wav_path) 

if __name__ == '__main__':
    
    # list of impulse length to create: unit is [msec]
    list0=[10] # ,15,20,25]
    
    for len0 in list0:
        # instance
        impulse1=Class_BlastImpulse(f_start=600.0, length=len0)
        #
        impulse1.plot_waveform()
        impulse1.f_show()
        # impulse1.append_zero_data(100) # option
        impulse1.save_wav()
