from cmath import pi
from ctypes import sizeof
from re import M
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st 
import scipy as sp
import sympy as sym
from scipy import signal
from scipy import integrate 
from scipy.integrate import quad
import scipy.fftpack as fourier
from math import *
import scipy.io.wavfile as waves

def serief(n,t,y,T,p):

    A = np.zeros((n))
    B = np.zeros((n))
    c = np.zeros(n)
    t5=np.zeros(n)
    fase= np.zeros(n)
    m=np.size(t)
    a0=0
    for i in range(m):
        a0=a0+(1/T)*y[i]*p

    for i in range(n):
        for j in range (m):
            A[i]=A[i] + ((2/T)*y[j]*np.cos(i*t[j]*f))*p   
            B[i]=B[i] + ((2/T)*y[j]*np.sin(i*t[j]*f))*p

        c[i] = ((A[i]**2)+(B[i]**2))**0.5
        t5[i]=i+1
        if A[i]==0:
            fase[i]= (np.pi)/2
        else: 
            fase[i] = -np.arctan(B[i]/A[i])

    t1=np.arange(-T,T,p)
    xf=0*t1-a0

    for i in range(n):
        xf= xf+A[i]*np.cos(i*f*t1)+B[i]*np.sin(i*f*t1)

    return t1,xf,c,fase,t5,a0


def tf(y,fs):
    ft=np.fft.fft(y) 
    hzft=abs(ft)
    phift=np.angle(ft)
    t1=np.fft.fftfreq(len(y))*fs
    return t1,hzft,

plt.plot(t1,hzft,'g')
plt.show()
