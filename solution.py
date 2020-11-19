# -*- coding: utf-8 -*-

import numpy as np;
import networkx as nx;
import matplotlib.pyplot as plt;
from collections import defaultdict,Counter;

class Solution:
  def __init__(self,N,M,T,Mg=0):
    self.T = T;
    self.N = N;
    self.M = M; # x次元
    self.Mg = Mg; # 罰金項の数
    self.ts = range(0,T);
    if M > 1:
      self.xs = np.zeros((N,M,T));
    else:
      self.xs = np.zeros((N,T));
    self.Js=np.zeros(T);
    self.us = np.zeros((N,T));
    self.rs = np.zeros((N,T));
    if Mg > 1:
      self.gs = np.zeros((Mg,T)); # 罰金項
      self.ls = np.zeros((Mg,T)); # 重み(双対変数)
    elif Mg > 0:
      self.gs = np.zeros(T);
      self.ls = np.zeros(T);
    self.fig = 1;
    self.conv = False;
    self.sinks = {};

  def addsink(self,s,T,t):
    self.sinks[s] = t;
    print "------";
    for k,v in self.sinks.items():
      if t-v > T:
        del self.sinks[k];

  def add(self,t,x,J=None,u=None,g=None,l=None,r=None):
    if J is not None:
      self.Js[t] = J;
    if u is not None:
      self.us[:,t] = u;
    if r is not None:
      self.rs[:,t] = r;
    if self.M > 1:
      self.xs[:,:,t] = x;
    else:
      self.xs[:,t] = x;
    if g is not None:
      if self.Mg > 1:
        self.gs[:,t] = g;
	self.ls[:,t] = l;
      else:
        self.gs[t] = g;
	self.ls[t] = l;
  
  def plotJ(self,fname=None):
    fig = plt.figure(self.fig); self.fig+=1;
    plt.cla();
    ax = fig.gca();
    ax.plot(self.ts,self.Js);

    plt.ylabel("j");
    if fname is None:
      plt.show();
    else:
      print fname;
      plt.savefig(fname);

  def plotdJ(self):
    fig = plt.figure(self.fig); self.fig+=1;
    plt.cla();
    ax = fig.gca();
    dj = np.diff(self.Js);
    ax.plot(self.ts[:-1],np.diff(self.Js));

    plt.ylabel("dj");
    plt.show();

  def plotL1d(self):
    fig = plt.figure(self.fig); self.fig+=1;
    ax = fig.gca();

    print self.gs*np.mat(self.ls).A.flatten();
    ax.plot(self.ts,self.Js+self.gs*np.mat(self.ls).A.flatten());

    plt.ylabel("L");
    plt.show();
    
  def plotU(self):
    fig = plt.figure(self.fig); self.fig+=1;
    ax = fig.gca();
    M,T=self.us.shape;
    for j in range(0,M):
      ax.plot(self.ts,np.mat(self.us[j,:]).A.flatten());

    #plt.ylim([-1.1,1.1]);
    plt.ylabel("ui");
    plt.show();

  def plotPdga(self):
    fig = plt.figure(self.fig); self.fig+=1;
    ax = fig.gca();
    #print self.gs;
    #print self.ls;
    ax.plot(self.ts,np.mat(self.xs).A.flatten());
    #plt.ylim([0,1]);
    plt.ylabel("x"); plt.xlabel("iteration");
    plt.show();

  # action比率
  def plotProp(self,A,fname=None):
    fig = plt.figure(self.fig); self.fig+=1;
    plt.cla();
    ax = fig.gca();
    N,T = self.xs.shape;
    ax.hold(1);
    col = ['r','b','g','c'];
    for a in range(A):
      pa = [Counter(self.xs[:,t])[a] for t in range(T)];
      print "a",a,A;
      ax.plot(self.ts,pa,color=col[a]);
    plt.ylabel("p"); plt.xlabel("iteration");
    plt.ylim([0,self.N+1]);
    if fname is None:
      plt.show();
    else:
      print fname;
      plt.savefig(fname);    

  def plotX(self):
    fig = plt.figure(self.fig); self.fig+=1;
    ax = fig.gca();
    j = 0;
    i = 0;
    for i in range(0,self.N): # 全員のj番目genomeを比較
    #for j in range(0,self.M): # player iの全genomeを比較
      ax.plot(self.ts,np.mat(self.xs[i,j,:]).A.flatten()); 
      #ax.hold(True);
      #plt.ylim([0,1]);
    plt.ylabel("x"); plt.xlabel("iteration");
    plt.show();