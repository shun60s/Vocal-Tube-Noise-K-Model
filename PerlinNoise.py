#coding:utf-8

# 1D Perlin noise python implementation
#
# This is a change from noise.py 
# of which
# Copyright (c) 2016 Brady Kieffer
# Please see LICENSE_SimplexNoise.txt.
# <https://github.com/bradykieffer/SimplexNoise>
# that was ported from C code:
# <https://github.com/stegu/perlin-noise/blob/master/src/noise1234.c>
#
#---------------------------------------------------
# Change features:
#  delete fractal function
#  add acceleration 変化の加速の設定
#  add lattice_size 格子サイズの設定
#

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1
#  scipy 1.0.0

import math
import random
import numpy as np
import matplotlib.pyplot as plt


class PerlinNoise(object):
    def __init__(self, num_octaves=8, pow_decrease_factor=0.1, freq_base=1.0, noise_amplitude=1.0, tc=2.0, sampling_rate=48000, speed_up_rate=0.0, lattice_size=256, upsampling_ratio=1): 
        self.num_octaves = num_octaves
        self.noise_amplitude = noise_amplitude
        self.freq_base = freq_base
        self.lattice_size=lattice_size
        self.octaves = [PerlinNoiseOctave(self.lattice_size) for i in range(self.num_octaves)]  # num_octaves個のPerlinNoiseOctavオブジェクトを準備する
        
        self.frequencies = [self.freq_base / pow(2, i) for i in range(self.num_octaves)]  # Octave分の周波数を準備する
        
        self.weight = [pow(pow_decrease_factor, len(self.octaves) - i) for i in range(self.num_octaves)]  # 各Octave分の加算の重みの計算
        self.sampling_rate=sampling_rate
        self.upsampling_ratio= upsampling_ratio
        print ('frequencies * sampling_rate /2, weight ')
        for l in range(1, len(self.frequencies)+1):
            print ( np.array(self.frequencies)[-l] * (self.sampling_rate/2.0) * self.upsampling_ratio, self.weight[-l]) 
        
        self.time_counter0 = 0 # use for acceleration time counter
        self.speed_up_rate = speed_up_rate    # speed up rate per[unit %]:   (1.0 / self.frequencies[-1]) period
        self.linear_a=  (self.speed_up_rate / 100.0) / (1.0 / self.frequencies[-1])
        self.tc= tc  # rise shape factor of exp acceleration
        
    def acceleration_linear(self,xin):
        # 直線加速 a
        return xin * ( 1.0 + self.linear_a * self.time_counter0)
        
    def acceleration_exp(self,xin):
        # (1-exp(-t))カーブの加速 a
        x= 1.0 * self.time_counter0 / self.length
        return  ( xin * (1.0 + self.speed_up_rate/100.0) - xin) * ( 1.0 -  np.exp( -self.tc * x)) / (1.0 - np.exp(- self.tc)) + xin
        
    def noise(self, x):
        # 入力ポイント（整数）に　Octave分の周波数を掛けている
        self.time_counter0 +=1  # 計算用に入力が1ポイントづつ呼ばれる毎に　カウンターを更新。
        noise = [
            self.octaves[i].noise(
                # xin=self.acceleration_linear ( x * self.frequencies[i]),
                xin=self.acceleration_exp ( x * self.frequencies[i]),
                noise_amplitude=self.noise_amplitude
            ) * self.weight[i] for i in range(self.num_octaves)]
        
        return sum(noise)
        
    def normalize(self,x):    
        # 1 ～ 0の間へクランプする
        res = (1.0 + x) / 2.0
        # Clamp the result, this is not ideal
        if res > 1:
            res = 1
            print ('warning: data was cliped to 1')
        if res < 0:
            res = 0
            print ('warning: data was cliped to 0')
        return res
        
    def make(self, length):
        # length ポイントのノイズを生成する
        Noise= np.zeros(length)
        self.length= length
        for i in range(length):
            Noise[i]= self.normalize( self.noise(x=i))
        
        Noise -= np.mean(Noise)  # 平均値をゼロにする
        self.Noise=Noise
        
        return Noise
        
    def plot_waveform(self, label):
        # plot waveform
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.xlabel('mSec')
        plt.ylabel('level')
        plt.title( label )
        plt.plot( (np.arange(len(self.Noise)) * 1000.0 / self.sampling_rate) , self.Noise)
        #
        fig.tight_layout()
        plt.show()
        
class PerlinNoiseOctave(object):
    # Perlin のノイズの生成関数
    def __init__(self, lattice_size, num_shuffles=100):
        # lattice_sizeまでの数字をランダムに並べ替えたハッシュ配列の準備
        self.lattice_size=lattice_size
        self.p_supply = [i for i in range(0, self.lattice_size)]
        
        random.seed(2)  # 乱数の発生を固定する（追加）
        for i in range(num_shuffles):  # シャッフルをnum_shuffles回　繰り返す
            random.shuffle(self.p_supply)
        
        self.perm = self.p_supply * 2  # 足し算で2倍になるので倍の長さを準備する
        
    def noise(self, xin, noise_amplitude):
        ix0 = int(math.floor(xin))  # 入力実数xinを切り捨てして整数にする
        fx0 = xin - ix0    # 格子点leftとの差
        fx1 = fx0 - 1.0    # 格子点rightとの差
        ix1 = (ix0 + 1) & (self.lattice_size - 1)  # ハッシュ配列ポイント用に切り詰め
        ix0 = ix0 & (self.lattice_size -1)        # ハッシュ配列ポイント用に切り詰め
        
        s = self.fade(fx0)    # fade関数で変換して滑らかにつなぐ  0<= s <=1
        
        n0 = self.grad(self.perm[ix0], fx0)  # 格子点left　勾配
        n1 = self.grad(self.perm[ix1], fx1)  # 格子点right　勾配
        
        return noise_amplitude * self.lerp(s, n0, n1)
        
    def lerp(self, t, a, b):   # 線形補間 0<= t <= 1
        return a + t * (b - a)
        
    def fade(self, t):  # fade関数　fade(0)=0, fade(1)=1, fade'(0)=fade'(1)=0 1次導関数が零の条件を満たす
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        
    def grad(self, hash, x):
        # 勾配の値を ±1.0 ～ ±8.0に制限する
        h = hash & 15   # hash and 1110
        grad = 1.0 + (h & 7)  # Gradient value from 1.0 - 8.0  下位3ビットに１を足す
        if h & 8:  # h and 1000
            grad = -grad  # Add a random sign  4ビット目が１の場合は　勾配をマイナスにする
        return grad * x

# utilty program
def sub_plot_waveform(wav, label, sampling_rate=48000):
    # plot waveform
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.xlabel('mSec')
    plt.ylabel('level')
    plt.title( label )
    plt.plot( (np.arange(len(wav)) * 1000.0 / sampling_rate) , wav)
    #
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    from scipy import signal
    from scipy.io.wavfile import write as wavwrite
    from iir1 import *
    
    # instance (1)
    speed_up_rate=[100] # [100.0, 60.0]
    
    for i in range( len(speed_up_rate) ):
        pn = PerlinNoise(num_octaves=8, speed_up_rate=speed_up_rate[i], tc=1.0)
        fu= pn.make(length = int( 150 * 0.001 / (1.0 / pn.sampling_rate)) )  # 150ms, to hear, length is longer than need
        # apply HPF(low freq cut) to narrow spectrum band
        hpf2= Class_IIR1(fc=480.0, btype='high', n_order=3)
        yh= hpf2.filtering( fu )
        print ('apply hpf to narrow band') 
        sub_plot_waveform(yh,'Noise waveform (narrow band)', pn.sampling_rate)
        wavwrite( 'k_noise' + str(i) + '.wav', pn.sampling_rate, ( yh * 2 ** 15).astype(np.int16))
        print ( 'k_noise' + str(i) + '.wav')
        
        