# -*- coding: utf-8 -*-

from gridbru import GridBRU;
import sys;
import numpy as np;
import networkx as nx;
import matplotlib.pyplot as plt;
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties;
import itertools;
import pulp;
import copy;
import sympy;
import re;
import gamebase as gb;
from scipy.spatial import distance;

# Gridを動くゲーム
# 状態xは(group,x)
# groupと近傍は違う
class GroupGridGame(GridBRU):
  def __init__(self,n,space,ri,exc=True,x0=None):
    GridBRU.__init__(self,n,space,exc,x0);
    self.ri = ri; # neighbor radius
    self.x = [(0,xi) for xi in self.x];
    self.updateg();


  # 近傍(距離行列)更新
  def updateg(self):
    x = [e[1] for e in self.x];
    #self.D = distance.cdist(x,x, metric='euclidean'); # ユークリッド距離
    self.D = distance.cdist(x,x, metric='cityblock'); # マンハッタン距離

  def neighbors(self,i):
    for j in np.where(self.D[i] <= self.ri)[0]:
      if j!=i:
        yield j;

  def d2a(self,i,dai):
    gi,vi = dai;
    xi = self.x[i][1];
    x = [e[1] for e in self.x];
    ai = self.move(vi,xi,x);
    if gi < 0:
      gi = np.max([e[0] for e in self.x])+1;
    return gi,ai;

  # 状態でなく動きの選択肢
  # 新groupは-1
  def dAi(self,i):
    xi = self.x[i][1];
    x = [e[1] for e in self.x];
    gset = set([self.x[j][0] for j in self.neighbors(i)]+[self.x[i][0],-1]); # 自分のと隣のと新しいの
    for th in range(0,5):
      for g in gset:
        yield (g,th);
	
  # 状態xにおけるiの選択肢=隣接位置
  def Ai(self,i):
    xi = self.x[i][1];
    x = [e[1] for e in self.x];
    newg = np.max([e[0] for e in self.x])+1;
    gset = set([self.x[j][0] for j in self.neighbors(i)]+[self.x[i][0],newg]); # 自分のと隣のと新しいの
    for xin in set(self.move(th,xi,x) for th in range(0,5)):
      for g in gset:
        yield (g,xin);

  def transition(self,pf):
    self.x = pf; # そのまま上書き
    self.updateg();

  def randomwalk(self,T):
    for t in range(T):
      for i in self.players():
        x = [e[1] for e in self.x];      
        xi = self.x[i][1];
        self.x[i] =(self.x[i][0],self.move(np.random.randint(5),xi,x));

  def plot(self,plt,t):
    fig = plt.figure(1)
    fig.set_size_inches(self.space)
    plt.clf();
    '''
      plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False);
      plt.tick_params(bottom=False,
                left=False,
                right=False,
                top=False);      
    '''
    #plt.cla();
    #gs = GridSpec(1,1)
    #ax = plt.subplot(gs[0,0]);
    ax = fig.gca();      
      
    plt.xlim([0,self.space[0]]);
    plt.ylim([0,self.space[1]]);
    #ax.grid(which='both');
    for aax in range(self.space[0]):
      ax.axvline(aax, ls = "--");
    for ay in range(self.space[1]):
      ax.axhline(ay, ls = "--");
    #cval = np.array(self.active);
    cval = np.zeros(self.N);
    scale = 10./(self.space[0])*(self.space[1]);

    xs = np.array([self.getx()[i][1][0] for i in self.players()]);
    ys = np.array([self.getx()[i][1][1] for i in self.players()]);

    dbgi = -1;
    cval[dbgi] = 0.;

    # 円
    im = plt.scatter(xs+0.5,ys+0.5,s=12*scale*5**2,alpha=0.5,vmin=0.,vmax=1.,c=cval,cmap='bwr');
    # 時刻
    #plt.text(1,self.space[1],"t:%d"%self.t,fontsize=10,ha='center',va='center');
    plt.title("t:%d"%t);
    
    if 0: # アニメ生成がめちゃ遅くなる。gifならok
      for i in range(self.N):
        plt.text(self.S[i,0],self.S[i,1],i,fontsize=10,ha='center',va='center');
        #plt.annotate(i,(self.S[i,0],self.S[i,1]));
    #plt.colorbar(im);
    #plt.show();
    return im;

  def dumppos(self,fname=None):
    print fname;
    f = open(fname,"w");

    for i in self.players():
      x = self.x[i][1];
      print >>f, ",".join(str(e) for e in x);
    f.close();

