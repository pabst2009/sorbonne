# -*- coding: utf-8 -*-

from markovfixu import MarkovFixU;
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
class GridGame(MarkovFixU):
  def __init__(self,n,space,exc=True,obstacle=None,x0=None):
    self.space = space;
    self.obstacle=obstacle; # [(x,y)]
    X = list(self.eachx());
    if obstacle is not None:
      for ax in obstacle:
        X.remove(ax);
    self.exclusive = exc;
    self.alpha = 0.5;
    if x0 is None:
      if exc:
        x0 = gb.sample(X,n); # 初期位置。衝突あり
      else:
        x0 = gb.sampledup(X,n); # 初期位置。衝突なし。重なりok
    MarkovFixU.__init__(self,n,None,x0);

  def eachx(self):
    for x in itertools.product(*(range(dm) for dm in self.space)):
      yield x;

  # 形状tから距離rだけ拡張
  def extend(self,t,r):
    ret = {};
    for x in self.eachx():
      for xt in t:
        if self.dist(x,xt)<=r:
          ret[x]=1;
    return ret.keys();

  def circle(self,c,r):
    for x in self.neighbor(c,r):
      yield x;

  def rectangle(self,x0,dm):
    for x in itertools.product(*(range(adm) for adm in dm)):
      yield (x[0]+x0[0],x[1]+x0[1]);

  def showx(self):
    for i in self.players():
      print i,self.x[i];

  def partgraph(self,nodes):
    g = nx.Graph();
    for n in nodes:
      g.add_node(str(n));
      for n2 in nodes:
        if n==n2:
          continue;
        if self.dist(n,n2)>1:
          continue;
        g.add_edge(str(n),str(n2));
    return g;

  # マンハッタン距離でのneighborサイズ
  def nbsize(self,r):
    return 2*r*(r+1)+1;

  def dist(self,x1,x2):
    return distance.cdist([x1],[x2], metric='cityblock')[0,0]; # マンハッタン距離

  # マンハッタン距離でr内のcell(agentでない)
  def neighbor(self,xi,r=1):
    for ax in self.eachx():
      #d = distance.cdist([xi],[ax], metric='cityblock'); # マンハッタン距離
      d = self.dist(xi,ax);
      if d <= r:
        #print xi,ax,d[0][0];
	yield ax;

    '''
    # 遅い
    done = {};
    #for ar in range(1,r+1):
    #  for ai in set(self.move(th,xi,r=ar) for th in range(0,5)):
    for ai in set(self.mover(xi,r,{})):    
        if ai not in done:
          done[ai] = 1;
          yield ai;
    '''

  # マンハッタン距離でr内のagent
  def neighboragents(self,i,r=1,x=None):
    if x is None:
      x = self.x;
    x2i = {x[j]:j for j in self.players()};
    for xn in self.neighbor(x[i],r):
      if xn in x2i:
        yield x2i[xn];

  # 状態xにおけるiの選択肢=隣接位置
  def Ai(self,i):
    xi = self.x[i];
    #print "Ai",i,xi;
    return self.vacant(xi);

  # 空いてる隣接位置
  def vacant(self,ax,r=1):
    x = None;
    if self.exclusive:
      x = self.getx();
    #print " vacant",x;
    for ai in set(self.move(th,ax,x,r=r) for th in range(0,5)):
      yield ai;

  def transition(self,pf):
    self.x = pf; # そのまま上書き

  # obsolete 遅い
  def mover(self,xi,r,doneo,x=None):
    done = {k:v for k,v in doneo.items()}; # deepcp
    #print "r",xi,r;
    if r==1:
      for th in range(0,5):
        yield self.move(th,xi,x);
      return;
    else:
      r -= 1;
      for th in range(0,5):
        xj = self.move(th,xi,x);
        #print "r,th,xi,xj",r,th,xi,xj;
        #print " xi,r",xi,r;
	if xj not in done:
          for xk in self.mover(xj,r,done,x):
            yield xk;
	  done[xj] = 1;
    return;

  # xが指定されてたら衝突判定
  def move(self,ai,xi,x=None,r=1):
    if ai == 4:
      return xi;
    th = ai*np.pi/2;
    v = np.array([np.cos(th),np.sin(th)],dtype=np.int);
    ret = np.array(xi)+v*r;
    for dm in range(2):
      if ret[dm] < 0:
        ret[dm] = 0;
      if ret[dm] > self.space[dm]-1:
        ret[dm] = self.space[dm]-1;
    ret = tuple(ret);
    #print "collision?",ai,ret,x.values();
    if (x is not None) and (ret in x): # 衝突
      return xi;
    if (self.obstacle is not None) and (ret in self.obstacle): # 障害物
      return xi;
    return ret;

  # 離散ボロノイ(gao2008notes)
  # 全点の一番近いagent hash
  def voronoi(self,xh,xgoal=None):
    q2i = {}; i2v = {i:[] for i in self.players()};
    x = [xh[i] for i in self.players()];
    N = [i for i in self.players()];
    def euclideanc(u,v):
      e = 0.0000001;
      ret = np.sqrt(((u-v)**2).sum());
      for k in range(len(u)):
        if v[k] > u[k]:
          ret += e; # 原点に近い方優先
      return ret;

    itr = self.eachx();
    if xgoal is not None:
      itr = xgoal;
    d = distance.cdist(list(itr),x, metric=euclideanc);
    for j,q in enumerate(itr): # 各点について
      #print q,d[j],np.argmin(d[j]);
      i = N[np.argmin(d[j])];
      q2i[q] = i;
      i2v[i].append(q);
    return q2i,i2v;

  # 離散被覆制御(gao2008notes)
  def covering(self,T,stopcycle=False):
    def mass(i,h):
      return sum(h[i]);
    def center(i,h,x):
      if len(h[i])==0:
        return np.array(x[i]);
      # m = mass(i,h); # 一律重みなら不要
      return np.mean(np.array(h[i]),axis=0);

    pf = self.initpf();
    dbgi = self.dbgi;
    wupdated = True;
    cycle = False;
    hpf = {};
    t = 0;
    yield t,0,pf; t+=1;
    while wupdated: # tループ:だれかupdateしてたら
      if cycle:
        break;
      if 0 and (t >= T):
        break;
      wupdated = False;
      q2i,i2v = self.voronoi(pf,xgoal=self.xgoal);
      for i in self.players():
        xc = center(i,i2v,pf);
        dx = xc-np.array(pf[i]);
        if np.linalg.norm(dx) >= 1: # 1マス以上なら一番近い選択肢に動く
          #print i,pf[i],xc,dx;
	  Ai = [ai for ai in self.Ai(i)]; # 現在のxでの選択肢
          best = np.argmin(distance.cdist([xc],Ai));
          print " ",i,pf[i],"->",Ai[best],"xc",xc;
          pf[i] = Ai[best];
          self.transition(pf); # 状態更新しないとぶつかる
	  if tuple(pf.values()) in hpf:
	    cycle = True;
	  hpf[tuple(pf.values())] = 1;
	  wupdated = True;
	  if cycle and stopcycle:
	    print "cycle! i",i;
	    break;

      yield t,i,pf; t+=1;

  def plotGrid(self,plt):
    fig = plt.figure(1)
    space = np.array(self.space)*1.;
    if space[1] < 2:
      space[1]=2.5;
    fig.set_size_inches(space)
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
      

  def plot(self,plt,t,cval=None,zorder=1,id=False):
    self.plotGrid(plt);
    if cval is None:
      cval = np.zeros(self.N);
    scale = 10./(self.space[0])*(self.space[1]);

    xs = np.array([self.getx()[i][0] for i in self.players()]);
    ys = np.array([self.getx()[i][1] for i in self.players()]);

    #cval[1] = 1.;

    # 円
    im = plt.scatter(xs+0.5,ys+0.5,s=12*scale*5**2,alpha=self.alpha,vmin=0.,vmax=1.,c=cval,cmap='bwr',zorder=zorder);
    
    if self.obstacle is not None:
      oxs = np.array([ax[0] for ax in self.obstacle]);
      oys = np.array([ax[1] for ax in self.obstacle]);
      ocval = np.ones(len(self.obstacle));
      plt.scatter(oxs+0.5,oys+0.5,s=12*scale*5**2,alpha=self.alpha,vmin=0.,vmax=1.,c='g',zorder=zorder);
    # 時刻
    #plt.text(1,self.space[1],"t:%d"%self.t,fontsize=10,ha='center',va='center');
    plt.title("t:%d"%t);
    
    if id: # アニメ生成がめちゃ遅くなる。gifならok
      for i in self.players():
        plt.text(self.getx()[i][0]+0.5,self.getx()[i][1]+0.5,i,fontsize=12,ha='center',va='center');
        #plt.annotate(i,(self.S[i,0],self.S[i,1]));
    #plt.colorbar(im);
    #plt.show();
    return im;

