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

def señalc(w0,w1,w2,fs,A0,A1,A2):
    Ts=1/fs  
    t=np.arange(0,1,Ts)
    y=A0*np.sin(w0*t) + A1*np.cos(w1*t) + A2*np.sin(w2*t)
    return t,y

def señald(w0,w1,w2,fs,A0,A1,A2,n):
    t=np.arange(0,n)
    y=A0*np.sin(2*np.pi*w0*t/fs) + A1*np.cos(2*np.pi*w1*t/fs) + A2*np.sin(2*np.pi*w2*t/fs)
    return t,y

def tf(y,fs):
    ft=np.fft.fft(y)
    oft=sp.fft.fftshift(ft) 
    hzft=abs(ft)
    phift=np.angle(oft)
    t1=np.fft.fftfreq(len(y))*fs
     #t1=fs*np.arange(-0.5-(1/len(hzft)),0.5-(1/len(hzft)),(1/len(hzft)))
    t2=fs*np.arange(-0.5-(1/len(phift)),0.5-(1/len(phift)),(1/len(phift)))
    return t1,t2,hzft,phift
