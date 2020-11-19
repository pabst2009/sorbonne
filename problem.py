# -*- coding: utf-8 -*-

import platform;
irid = (platform.system()=='Linux');

import numpy as np;
import copy;
import networkx as nx;
from networkx.algorithms import bipartite;
import matplotlib.pyplot as plt;
import itertools;
import scipy.sparse as sp;
from scipy.optimize import minimize,fmin;
from scipy.spatial import distance;
import sys
import meanfield;
import gamebase as gb;
import scipy.spatial as spa;
from collections import Counter;
import util;

class Problem:
  # 分散なので各エージェントiごとに解候補xiを持っている。次元が共通とは限らない
  def __init__(self,n,maxe=None):
    self.N = n; # エージェント数
    self.A = None; # 選択肢
    self.t = 0; # plot用時刻
    if maxe is None:
      self.G = nx.complete_graph(n); # 完全グラフ
    else:
      self.G = self.randgraph(n,maxe); # 近接グラフ
    for i in self.G.nodes():
      self.G.add_node(i,agent=Agent(i,self));
    #A = nx.adjacency_matrix(self.G).todense();
    #self.draw(); exit();
    self.err = 0.01;

  def randgraph(self,k,maxe):
    ug = nx.Graph();
    ug.add_nodes_from(range(k));
    # hight-depthバランスをとるよう最大outdegreeを決める
    # 接続するまでランダムにエッジを足す
    while not nx.is_connected(ug):
      no,nd = np.random.randint(k,size=2);
      if no == nd:
        continue;
      if ug.degree(no) >= maxe:
        continue;
      ug.add_edge(no,nd);
    return ug;

  def agents(self):
    for i in self.G.nodes():  
      yield i,self.G.node[i]['agent'];

  def agent(self,i):
    return self.G.node[i]['agent'];

  def actions(self,i):
    for a in self.A:
      yield a;

  # 近傍PF(自分は含まない)
  def aNi(self,a,i):
    return np.array([a[j] for j,tmp in self.agent(i).neighbors()]);

  # 近傍の競合人数
  def congestion(self,a,i):
    aN = self.aNi(a,i);
    return len(aN[np.where(aN==a[i])])+1; # 自分の選択の人数

  # interface
  def ui(self,pf,i):
    print("implement ui!"); exit();

  def us(self,pf):
    return np.array([self.ui(pf,i) for i,a in self.agents()]);

  def socialWelfare(self,pf):
    ret = 0.0;
    for i,a in self.agents():
      ret += self.ui(pf,i);
    return ret;

  def show(self,pf):
    print(pf, [self.ui(pf,i) for i,a in self.agents()],self.socialWelfare(pf));

  # 利得グラフ
  def drawu(self):
    print("implement problem.drawu!"); exit();

  # グラフを描く
  def draw(self):
    npos=nx.spring_layout(self.G,k=10.0,iterations=120); # なかなか
    nx.draw(self.G,npos);
    labels = {};
    for i,a in self.agents():
      labels[i] = i;    
    nx.draw_networkx_labels(self.G,npos,labels);
    plt.show();

class Agent:
  def __init__(self,aid,env):
    self.id = aid;
    self.env = env;

  def p(self):
    print("Agent",self.id);

  # 近傍：自分は含まない
  def neighbors(self):
    for i in self.env.G.neighbors(self.id):
      yield i,self.env.G.node[i]['agent'];

class SmallDCOP(Problem):
  def __init__(self,n):
    Problem.__init__(self,n);
    Ai = ['L','R'];
    self.subg={};
    gA = gb.AtomG([Ai]*2);
    gA.u = self.makeU(Ai,[-1,2,1,-1]);
    self.subg['A'] = gA;
    gB = gb.AtomG([Ai]*2);
    gB.u = self.makeU(Ai,[-1,3,1,-1]);
    self.subg['B'] = gB;
    gC = gb.AtomG([Ai]*2);
    gC.u = self.makeU(Ai,[-1,1,3,-1]);
    self.subg['C'] = gC;
    A = [ai for ai in itertools.product(gb.powerset(self.subg.keys()),Ai)];
    g=gb.AtomG([A]*n,zerou=1);
    self.g = g;

    so = None; maxj = -np.inf;
    for pf in g.pfs():
      j = self.J(pf);
      if maxj < j:
        maxj = j;
        so = pf;
    print so,maxj;

  def makeU(self,Ai,u):
    ret = {};
    for i,pf in enumerate(itertools.product(Ai,Ai)):
      ret[pf]=np.ones(2)*u[i];
    return ret;

  def subpf(self,pf,g):
    ret = [];
    ps = [];
    for i in self.g.players():
      if g in pf[i][0]:
        ret.append(pf[i][1]);
        ps.append(i);
    return ps,ret;

  def J(self,pf):
    ret=0; o1 = 0; o2 = 0;
    for k,g in self.subg.items():
      ps,spf = self.subpf(pf,k);
      if len(spf) != 2:
        ret-=100;
      else:
        u = g.calU(spf,0);
        if k == 'A':
          o1 += u;
        else:
          o2 += u;
    ret += -(o1-2)**2-(o2-3)**2;
    return ret;
      
    


class ProblemMFG(Problem):
  def __init__(self,n,maxe=None):
    Problem.__init__(self,n,maxe);

  def prntdbg(self,a):
    1;

  def plot(self,plt,t):
    self.game.plot(plt,t);

  def showresult(self,prnt=0):
    return None,None;

# 次元mベクトルConsensus問題
class Consensus2D(Problem):
  def __init__(self,n,maxe,m,c=5):
    Problem.__init__(self,n,maxe);
    self.X = np.zeros((n,m));
    for i,a in self.agents():
      self.X[i,:] = np.random.rand(m); # m次元[0,1]

    # 最小化問題
    self.j = lambda x:(np.sum(x)-c)**2; # xはpf。分離がSO：ui=J。対称だから分離起きない
    self.dj = lambda x,i:np.ones(len(x))*2*(sum(x)-c); # xはpf。分離がSO：対称だから分離起きない
    self.ui = lambda x,i:(np.sum(x)-c)**2; # コスト。xはpfのこと
    self.dui = self.dj;
    # 改良罰金dhi/dxi
    self.hi = lambda x,i:np.array([np.abs(sum(x)-1)]); # h=|x-1|=0
    self.dhi = lambda x,i:np.ones(len(x))*np.sign(np.sum(x)-1); # dhi/dxi=sign(x-1)
    self.gi = lambda x,i:np.hstack((-np.log(x),-np.log(1-x))); # g=[-log(x),-log(1-x)] (x>=0, x<=1)
    self.dgi = lambda x,i:np.vstack((np.diag(-1./x),np.diag(1./(1.-x))));


class salaryYen(Problem):
  def __init__(self,n,maxe,m):
    Problem.__init__(self,n,maxe);
    self.X = np.zeros((n,m));
    for i,a in self.agents():
      self.X[i,:] = np.random.rand(m); # m次元[0,1]

    mu = 127.56; sig = 16.51; # EURJPYのmu,sig
    xt = 125.24; # 今月のEURJPY
    s = 3000; # salary EUR
    # 最小化問題
    self.j = lambda r:-s*r*(xt-mu);
    self.dj = lambda r:-s*(xt-mu);
    self.ui = lambda r,i:0; # ダミー
    # 罰金
    self.gi = lambda r:(s*(1-r)*sig)**2;
    self.dgi = lambda r:2*(r-1)*(s*sig)**2;

# コンプライアンス最小化
class TopoOpt(Problem):
  def __init__(self,n):
    Problem.__init__(self,n,maxe);
    self.A = [(0,0)]; # 移動可能な位置座標
    self.ui = lambda pf:(np.sum(x)-c)**2; # コスト。xはpfのこと    
    #util.plotFunc(ui);

# Tiger
class Tiger(Problem):
  def __init__(self,n,anime=0):
    Problem.__init__(self,n);
    self.anime=anime; # 0:なし,1:mp4,2:gif
    m = decpomdp.DecPOMDP(2);
    sl = m.add_state();
    sr = m.add_state();
    ol=0; Or=1; lt=2;
    m.A = range(3); # policysearch用
    trans = [(sr,sr),(sr,sl),(sl,sr),(sl,sl)];
    for o,d in trans:
      for ai in range(2):
        for aj in range(3):
          m.add_action(o,d,(ai,aj),0.5);
          m.add_action(o,d,(ai,aj),0.5);
          m.add_action(o,d,(aj,ai),0.5);
          m.add_action(o,d,(aj,ai),0.5);
      m.add_action(o,d,(lt,lt),1.);

    m.setU(sl,(Or,Or),[20]*2);
    m.setU(sr,(Or,Or),[-50]*2);
    m.setU(sl,(ol,ol),[-50]*2);
    m.setU(sr,(ol,ol),[20]*2);
    m.setU(sl,(Or,ol),[-100]*2);
    m.setU(sl,(ol,Or),[-100]*2);
    m.setU(sr,(Or,ol),[-100]*2);
    m.setU(sr,(ol,Or),[-100]*2);
    m.setU(sl,(lt,lt),[-2]*2);
    m.setU(sr,(lt,lt),[-2]*2);
    m.setU(sl,(lt,Or),[9]*2);
    m.setU(sr,(lt,Or),[-101]*2);
    m.setU(sl,(Or,lt),[9]*2);
    m.setU(sr,(Or,lt),[-101]*2);
    m.setU(sl,(lt,ol),[-101]*2);
    m.setU(sr,(lt,ol),[9]*2);    
    m.setU(sl,(ol,lt),[-101]*2);
    m.setU(sr,(ol,lt),[9]*2);

    hl = 0; hr = 1;
    m.setO(sl,(lt,lt),hl,0.85);
    m.setO(sl,(lt,lt),hr,0.15);
    m.setO(sr,(lt,lt),hl,0.15);
    m.setO(sr,(lt,lt),hr,0.85);
    for s in [sl,sr]:
      for o in [hl,hr]:
        for ai in range(2):
          for aj in range(3):
	    m.setO(s,(ai,aj),o,0.5);
	    m.setO(s,(aj,ai),o,0.5);
	  
    m.setSource(sl);
    m.fix();
    #m.show();
    m.plot();
    #print(m.opt());

    self.game = m;

# 非対称mean field game
# ui=mean(x)からの距離
class MFcircle(ProblemMFG):
  def __init__(self,n,space,wall=0,anime=0):
    ProblemMFG.__init__(self,n);
    self.anime=anime; # 0:なし,1:mp4,2:gif
    self.space = space;
    self.delay=20; # gif delay
    f = lambda x: np.mean(x,0); # 平均位置
    u = lambda i,x,f: self.u_(i,x[i],f(x));
    #u = lambda i,x,f: -np.abs(np.linalg.norm(x[i]-f(x))-((i%3)+1)*4); # 平均位置からの距離と目標距離の差

    self.game = meanfield.MeanField(n,space,f,u,self.u_,dbgi=-1);
    #x0 = [x for i,x in enumerate(self.game.eachx()) if i < self.game.N]; # 偏りy軸寄り
    #x0 = [x for i,x in enumerate(sorted(self.game.eachx(), key=lambda x:x[0]+x[1])) if i < self.game.N]; # (0,0)寄り
    x0 = [x for i,x in enumerate(sorted(self.game.eachx(), key=lambda x:(x[0]-space[0]/2)**2+(x[1]-space[1]/2)**2)) if i < self.game.N]; # grid中心寄り
    #self.game = meanfield.MeanField(n,space,f,u,self.u_,x0=x0);

  def u_(self,i,xi,fx):
    xbias = np.array([5.,0.])*0;
    r = np.linalg.norm(xi-fx); # 半径：平均からの距離
    th = gb.azimuth(xi-fx)/(2*np.pi); # 角度[0,1]
    b = 8; a = 4;
    b = 7; a = 2;
    b = 7; a = 7;
    b = 6; a = b;
    R = ((i%3)+1)*b; # 目標半径(円)
    Rl = ((i%3)+1)*a; # 目標半径
    # R = 10; Rl=R;
    R = 12; Rl=7; # 五,6芒星(W=40)
    R = 12*100/40.; Rl=7*100/40.; # 五,6芒星(N=1000,W=100)
    #R = 12; Rl=8;
    # 三角波 x:[0,1], y:[Rl,R]
    T=6
    gamma = lambda x : (R-Rl)*np.arccos(np.cos(x*np.pi*2*T))/np.pi+Rl;
    # 中心非0
    #R = 15; Rl=3; gamma = lambda x : 4*(Rl-R)*(x-0.5)**2+R;
    #R = 15*100/40.; Rl=3*100/40.; gamma = lambda x : 4*(Rl-R)*(x-0.5)**2+R;
    #if i == self.game.dbgi:
    #  print "i,x,th,ri,r(th)",i,xi,th,r,gamma(th);
    if 0:
      th = np.arange(0,1.1,0.05);
      r = gamma(th);
      th = th*2*np.pi;
      x,y = [r*np.cos(th),r*np.sin(th)];
      fig = plt.figure(1);
      #plt.plot(th,r);
      plt.plot(x,y); plt.hold(True); plt.plot(0,0,'ro');
      plt.show();
      exit();
    #return -np.abs(r-R); # 目標半径との差＝形状からの距離(円)
    return -np.abs(r-gamma(th)); # 目標半径との差＝形状からの距離(☆)
    #return -np.abs(np.linalg.norm(xi-fx-xbias)-((i%3)+1)*4); # 平均位置からの距離と目標距離の差
    #return sum(xi-fx-xbias); # 平均より右上がいい

  def ng(self,i,a):
    return self.N;

  def plot(self,plt,t):
    im = self.game.plot(plt,t);
    # 中心上書き
    xf = self.game.f(self.game.x);
    x,y = xf;
    scale = 10./(self.space[0])*(self.space[1]);
    #im = plt.scatter(x+0.5,y+0.5,s=12*scale*5**2,alpha=0.5,vmin=0.,vmax=1.,c=1.,cmap='bwr');

    # 目標
    if 1:
      R = 12; Rl=7; # 五,6芒星
      R = 12*100/40.; Rl=7*100/40.; # 五,6芒星
      T=6
      gamma = lambda x : (R-Rl)*np.arccos(np.cos(x*np.pi*2*T))/np.pi+Rl;
      #R = 15; Rl=3; gamma = lambda x : 4*(Rl-R)*(x-0.5)**2+R; # 中心非0
      #R = 15*100/40.; Rl=3*100/40.; gamma = lambda x : 4*(Rl-R)*(x-0.5)**2+R; # 中心非0
      th = np.arange(0,1.1,0.005);
      r = gamma(th);
      th = th*2*np.pi;
      x,y = [r*np.cos(th),r*np.sin(th)];
      im = plt.plot(x+xf[0]+0.5,y+xf[1]+0.5,'r-',linewidth=8.0);

    
    # 誰かを赤く
    #x,y = self.game.x[46];
    #im = plt.scatter(x+0.5,y+0.5,s=12*scale*5**2,alpha=0.5,vmin=0.,vmax=1.,c=1.,cmap='bwr');

    #for i in self.game.players():
    #  xi = self.game.x[i];
    #  if np.linalg.norm(np.array(xi)-xf) < 3:
    #    print i,self.game.x[i],self.u_(i,xi,xf);
    return im;

# 非対称mean field game
# ui=mean(x)からの距離
# 回転・シフトの2パラメタdistribution rule
class MFformation(ProblemMFG):
  def __init__(self,n,space,wall=0,anime=0):
    ProblemMFG.__init__(self,n);
    self.anime=anime; # 0:なし,1:mp4,2:gif
    self.space = space;
    self.delay=20; # gif delay
    #f = lambda x: (0,np.mean(x,0)); # 図形中心
    f = lambda x: self.distrule(x); # 図形回転・中心
    u = lambda i,x,f: self.u_(i,x[i],f(x));

    self.game = meanfield.MeanField(n,space,f,u,self.u_,dbgi=-1);
    #x0 = [x for i,x in enumerate(self.game.eachx()) if i < self.game.N]; # 偏りy軸寄り
    #x0 = [x for i,x in enumerate(sorted(self.game.eachx(), key=lambda x:x[0]+x[1])) if i < self.game.N]; # (0,0)寄り
    x0 = [x for i,x in enumerate(sorted(self.game.eachx(), key=lambda x:(x[0]-space[0]/2)**2+(x[1]-space[1]/2)**2)) if i < self.game.N]; # grid中心寄り
    #self.game = meanfield.MeanField(n,space,f,u,self.u_,x0=x0);

  # xから(th,b)を計算
  def distrule(self,x):
    b = np.mean(x,0);
    # thの分散計算。各自のui最大thの平均
    fobj = lambda th: np.abs(self.S(xi,th,b));
    #th = np.mean([minimize(fobj, 0., method='nelder-mead',options={'xtol': 1e-8, 'disp': False}).x[0] for xi in x]);
    th = np.mean([fmin(fobj, 0.,disp=False)[0] for xi in x]);
    if 0:
      for i,xi in enumerate(x):
        #print sopt.fmin(fobj,0.);
        res = minimize(fobj, 0., method='nelder-mead',options={'xtol': 1e-8, 'disp': False}); #シンプレックス
        print res.x,res.fun;
        #th = np.arange(-np.pi/2,np.pi/2,0.001);
        #plt.plot(th,fobj(th));
        #plt.show();
        #exit();
    return th,b;

  # 目標図形
  def S(self,xi,th,b):
    # R = 10; Rl=R;
    R = 12; Rl=7; # 五,6芒星
    #R = 12; Rl=8;
    # 三角波 x:[0,1], y:[Rl,R]
    T=6  
    gamma = lambda th,R,Rl,T : (R-Rl)*np.arccos(np.cos(th*T))/np.pi+Rl;
    dgamma = lambda th,R,Rl,T : (R-Rl)*T/np.pi*np.sign(np.sin(th*T));
    # 中心非0
    #R = 15; Rl=3; gamma = lambda x : 4*(Rl-R)*(x-0.5)**2+R;

    r = np.linalg.norm(xi-b); # 半径：平均からの距離
    phi = gb.azimuth(xi-b); # 角度[0,2pi]

    if 0:
      dth = 0.001*2*np.pi;
      th = np.arange(0,2*np.pi+dth,dth);
      r = gamma(th,R,Rl,T);
      dr = dgamma(th,R,Rl,T);
      #rdr = [rk*dr[k] for k,rk in enumerate(r)];
      #dr2th = [drk**2*th[k] for k,drk in enumerate(dr)];
      x,y = [r*np.cos(th),r*np.sin(th)];
      fig = plt.figure(1);
      plt.hold(True);
      plt.plot(th,r);
      plt.plot(th,dr,'r');
      #plt.plot(th,rdr);
      #plt.plot(x,y); plt.hold(True); plt.plot(0,0,'ro');
      plt.show();
      exit();

    #return r**2-gamma(phi,R,Rl,T)**2; # 2乗である必要は特にない
    return r-gamma(phi-th,R,Rl,T);
    
  def u_(self,i,xi,dr):
    th,b=dr;
    return -np.abs(self.S(xi,th,b));

  def ng(self,i,a):
    return self.N;

  def plot(self,plt,t):
    im = self.game.plot(plt,t);
    # 中心上書き
    xf = self.game.f(self.game.x);
    x,y = xf;
    scale = 10./(self.space[0])*(self.space[1]);
    #im = plt.scatter(x+0.5,y+0.5,s=12*scale*5**2,alpha=0.5,vmin=0.,vmax=1.,c=1.,cmap='bwr');

    # 目標
    if 0:
      R = 12; Rl=7; # 五,6芒星
      T=6
      gamma = lambda x : (R-Rl)*np.arccos(np.cos(x*np.pi*2*T))/np.pi+Rl;
      #R = 15; Rl=3; gamma = lambda x : 4*(Rl-R)*(x-0.5)**2+R; # 中心非0
      th = np.arange(0,1.1,0.005);
      r = gamma(th);
      th = th*2*np.pi;
      x,y = [r*np.cos(th),r*np.sin(th)];
      im = plt.plot(x+xf[0]+0.5,y+xf[1]+0.5,'r-',linewidth=8.0);

    
    # 誰かを赤く
    #x,y = self.game.x[46];
    #im = plt.scatter(x+0.5,y+0.5,s=12*scale*5**2,alpha=0.5,vmin=0.,vmax=1.,c=1.,cmap='bwr');

    #for i in self.game.players():
    #  xi = self.game.x[i];
    #  if np.linalg.norm(np.array(xi)-xf) < 3:
    #    print i,self.game.x[i],self.u_(i,xi,xf);
    return im;    


# 非対称mean field game
# ui=|f(ai=x)-fg(x)|目標分布からの距離
class MFdist(Problem):
  def __init__(self,n,space,ri,wall=0,anime=0):
    Problem.__init__(self,n);
    self.anime=anime; # 0:なし,1:mp4,2:gif
    self.space = space;
    self.delay=20; # gif delay
    f0 = np.random.random(space); # 目標分布
    f0 /= np.sum(f0);
    self.f0 = f0;
    print "f0";
    print f0*n;
    self.ri = ri;

    self.ux = self.congest;
    self.ux = self.edist;
    self.ux = self.covering;
    f = self.fdist;
    u = self.ulocal;
    u_= self.usinglecell;
    u_= self.uallcell;
    u_= self.uneighbor;

    self.game = meanfield.MeanField(n,space,f,u,u_,exc=0);
    self.game.dbgi = -1;
    self.game.alpha = 0.2;
    self.f0 = self.peaks();
    #x0 = [x for i,x in enumerate(self.game.eachx()) if i < self.game.N];
    #print x0;
    #self.game = meanfield.MeanField(n,space,f,u,self.u_,x0=x0);

    if 0:
      fig = plt.figure(1);
      self.game.plot(plt,0);
      plt.show();
      exit();

  def peaks(self):
    x1 = np.int(self.space[0]/3);
    y1 = np.int(self.space[1]/3);
    x2 = 2*x1; y2 = 2*y1;
    f = np.zeros(self.space);
    p1 = (1,1); p2 = (3,3);
    p1 = (3,3); p2 = (6,6);
    p = [(2,2),(7,7),(2,7),(7,2)];
    p = [(2,2),(12,12),(2,12),(12,2)];
    p = [(3,3)];
    p = [(2,2),(7,7)];
    p = [(2,2),(9,9)];
    V = 0.5; # 分散
    #V = 4.; # 分散smooth
    for ax in self.game.eachx():
      for ap in p:
        d = distance.cdist([ax],[ap], metric='cityblock')[0][0];
        f[ax] += np.exp(-d/V);
    f = f/np.sum(f);
    return f;

  def showresult(self,prnt=True):
    err = np.sum(np.abs(self.fdist(self.game.x)-self.f0)*self.N);
    f = self.fdist(self.game.x)*self.N;
    if prnt:
      print "f0";
      print self.f0*self.N;
      print "f";
      print f;
      print "e";
      print (self.fdist(self.game.x)-self.f0)*self.N;
      print "err",err;
      #self.game.showx();
    return f,err;

  def covering(self,f,f0):
    ps = 0.5;
    Nx = np.int(f*self.N);
    return f0*(1.-(1.-ps)**Nx);

  def congest(self,f,f0):
    if f < np.finfo(float).eps:
      return 0.;
    return f0/f;

  def edist(self,f,f0):
    return -np.abs(f-f0);

  def fdist(self,x):
    ret = np.zeros(self.space);
    for ax in x:
      ret[ax]+=1;
    #ret /= np.sum(ret);
    ret /= self.N;
    return ret;

  # u_も変えること
  def ulocal(self,i,x,f):
    lmd=1.;
    #return self.usinglecell(i,x[i],f(x));
    #return self.usinglecell(i,x[i],f(x))+self.game.ext(x,i); # WLUi
    #return self.usinglecell(i,x[i],f(x))+lmd*self.N*self.uallcell(i,x[i],f(x)); # ほぼSW。チート
    return self.uneighbor(i,x[i],f(x)); # IJCAI2020WS
    #return self.uneighbor(i,x[i],f(x))+lmd*self.N*self.uallcell(i,x[i],f(x)); # neighbor+チート
    #return self.uneighbor(i,x[i],f(x))+self.game.ext(x,i); # neighborのWLUi   
    #return self.uallcell(i,x[i],f(x)); # opt
    #return np.sum([self.usinglecell(j,x[j],f(x)) for j in self.game.players()]); # WLUiではない

  def uallcell(self,i,xi,fx):
    ret = 0.;
    if i==self.game.dbgi:
      print "xi",xi;
    for ax in self.game.eachx():
      if i==self.game.dbgi:
        print " ax,fx,fx0,ux",ax,fx[ax]*self.N,self.f0[ax]*self.N,self.ux(fx[ax],self.f0[ax]);
      ret += self.ux(fx[ax],self.f0[ax]); # 目標分布との差

    if i==self.game.dbgi:
      print;
    return ret;

  def uneighbor(self,i,xi,fx):
    ret = 0.;
    cnt = 0;
    for xn in self.game.neighbor(xi,self.ri): # d=0だとsingle cellと同じ
      cnt += 1;
      fn = fx[xn];
      f0n = self.f0[xn];
      #if i==36:
      #  print xi,xn,self.ux(fn,f0n)*80;
      ret += self.ux(fn,f0n); # 目標分布との差
    #print cnt; exit();
    return ret;

  def usinglecell(self,i,xi,fx):
    fi = fx[xi];
    f0i = self.f0[xi];
    return self.ux(fi,f0i); # 目標分布との差

  def ng(self,i,a):
    return self.N;

  def prntdbg(self,a):
    1;

  def plot(self,plt,t):
    return self.game.plot(plt,t);


# self organization game
# 指定バッグに収まるようにできるだけ大きい構造を作る
class MForganize(Problem):
  def __init__(self,n,space,ri,rbag,rmax,E,wall=0,anime=0):
    Problem.__init__(self,n);
    self.dbgi = -1;
    self.anime=anime; # 0:なし,1:mp4,2:gif
    self.space = space;
    self.delay=5; # gif delay
    self.ngroup = 1; # 最初は1group
    self.rbag = rbag; # バッグ半径
    self.rmax = rmax; # 全部同じグループのときの半径
    f = self.fgroup;
    u = self.ugorg;
    u_ = self.uorg_;
    if E<0:
      self.game = meanfield.ModularMFG(n,space,ri,f,u,u_,dbgi=self.dbgi);
    else:
      x0 = self.readpos("input/sopos.org.txt");
      self.game = meanfield.ModularMFG(n,space,ri,f,u,u_,x0=x0,dbgi=self.dbgi);
      self.game.randomwalk(E);
    #self.game.du = self.du; # デバッグ用

  def showresult(self,prnt=True):
    return 0,0;

  def readpos(self,fname):
    ret = [];
    with open(fname) as f:
      for l in f:
	col = l.rstrip().split(",");
        ret.append(tuple(np.float(s) for s in col));
    #ret = np.array(ret);
    return ret;

  # aは(group,x)
  # 平均位置ハッシュxavg[g]を返す
  def fgroup(self,a):
    g =  self.groups(a);
    ret = {};
    gs = sorted(set(g));
    for ag in gs: # 各グループ
      xg = np.array([a[k][1] for k in range(len(a)) if (g[k]==ag) and (a[k][1][0] is not None)]);
      ret[ag] = (np.mean(xg,0),np.mean(np.std(xg,0)));
    return ret;

  def groups(self,a):
    return [e[0] for e in a];

  # グループサイズを返す
  def ng(self,i,a):
    gid = a[i][0];
    g = [e[0] for e in a if e[0]==gid];
    return len(g);

  # デバッグ用
  def du(self,i,a,b):
    fsa = self.fgroup(a);
    fsb = self.fgroup(b);
    ai = a[i]; bi = b[i];
    j1a,j2a = self.jorg_(ai,a,fsa);
    j1b,j2b = self.jorg_(bi,b,fsb);
    ula = self.uorg_(i,ai,a,fsa);
    ulb = self.uorg_(i,bi,b,fsb);
    print "i",i,"dul",ulb-ula,"dj1",j1b-j1a,"dj2",j2b-j2a;


  def ugorg(self,i,a,f):
    fs = f(a);
    ai = a[i];
    lmdc =30.; lmdj = 10.;
    j1,j2 = self.jorg_(ai,a,fs);
    #return self.uorg_(i,ai,a,fs)+lmdc*j1+lmdj*j2; # ui=uli+lmd*J
    lmdc /= self.N; lmdj /= self.N;
    ret = self.uorg_(i,ai,a,fs)+lmdc*self.N*j1+lmdj*self.N*j2; # WLUi=uli+lmd*N*J
    if self.dbgi == i:
      print "yes",ai,ret;
    return ret;
  
  def uorg_(self,i,ai,a,fs):
    dbgi = self.dbgi;
    #a[0] = (1,a[0][1]); a[1] = (2,a[1][1]);
    
    gid = ai[0];
    fg = fs[gid];
    xi = ai[1];
    g = np.array(self.groups(a));

    # グループ半径
    ng = len(g[np.where(g==gid)]);
    rg = np.float(ng)/self.N*self.rmax;
    if rg < 3.: # 最小半径
      rg = 3.;
    rg = 4; # 一定半径

    #ul = -np.abs(np.linalg.norm(xi-fg[0])-rg)+10*(fg[1]-100); # グループlocal項
    #ul = -np.abs(np.linalg.norm(xi-fg[0])-rg)+10*(ng-100); # グループlocal項
    ul = -np.abs(np.linalg.norm(xi-fg[0])-rg); # グループlocal項

    #if i==dbgi:
    #  print "i,g,ng",i,gid,ng;

    return ul;

  def jorg_(self,ai,a,fs):
    uc = 0.; # 構造制約項
    gset = set(self.groups(a)); l = len(gset);
    xgmean = np.mean([x for x,s in fs.values()],0);
    g = np.array(self.groups(a));
    for ag in gset:
      #ng = len(g[np.where(g==ag)]);
      #uc += -np.abs(np.linalg.norm(fs[ag][0]-xgmean)-7)*ng/self.N; # グループでも円。重心
      uc += -np.abs(np.linalg.norm(fs[ag][0]-xgmean)-12)/l; # グループでも円。中心
    
    #for ag in gset:
    '''
    # 全グループ組み合わせ
    for g1,g2 in itertools.combinations(gset,2):
      ng1 = len(g[np.where(g==g1)]);
      rg1 = np.float(ng1)/self.N*self.rmax;    
      ng2 = len(g[np.where(g==g2)]);
      rg2 = np.float(ng2)/self.N*self.rmax;
      rG = (rg1+rg2)/2;
      uc += -np.abs(np.linalg.norm(fs[g1][0]-fs[g2][0])-ropt);
      l += 1;
      #print g1,g2,(rg1,rg2),rG;
    if l == 0:
      l = 1;
    '''


    '''
    j = 0.; # global objective
    # 全体サイズ＝全グループ中心からの最大距離＋そのグループの半径
    xgmean = np.mean([x for x,s in fs.values()],0);
    gd = [(ag,np.linalg.norm(xg[0]-xgmean)) for ag,xg in sorted(fs.items(),key=lambda x:-np.linalg.norm(x[1][0]-xgmean))];
    gmax,xdgmax = gd[0];
    ng = len(g[np.where(g==gmax)]);
    rgmax = np.float(ng)/self.N*self.rmax;
    penalty = -100.;
    j = rgmax+xdgmax;
    j = -np.abs(self.rbag-j);
    '''

    j = 0.;
    gset = set(self.groups(a));
    for ag in gset:
      ng = len(g[np.where(g==ag)]);
      rg = np.float(ng)/self.N*self.rmax;
      j += -np.abs(rg-4);
    j /= len(gset);
    

    #j = -np.abs(len(fs)-3);
    j=0;

    return uc,j;

    '''
    lmdc =10; lmdj = 100;
    lmdc =1; lmdj = 5; # uc無い時
    lmdc =5; lmdj = 5; 
    lmdc =15; lmdj = 10; 
    lmdc =30; lmdj = 10; # so
    ui = ul+lmdc*uc+lmdj*j;
    if i==dbgi:
      print "ai",a[i];
      print " ul,uc,j,ui",ul,uc,j,ui;
    return ui;
    '''

  def prntdbg(self,a):
    fs = self.fgroup(a);
    g = np.array(self.groups(a));
    gset = set(self.groups(a)); l = len(gset);
    xgmean = np.mean([x for x,s in fs.values()],0);

    rg = self.rmax/len(gset);

    rgs = []; rls = [];
    for ag in gset:
      rgs.append(np.linalg.norm(fs[ag][0]-xgmean));
      fg = fs[ag];
      rls.append(np.mean([np.linalg.norm(xi-fg[0]) for gi,xi in a if gi==ag]));
    print "### global r:15",rgs;
    print "### local r:",rg,rls;
    print "### sw",self.game.sw(a);
      
  def so(self):
    # soグループ半径
    rgso = 4.;
    # soグループサイズ
    ngso = rgso*self.N/self.rmax;
    # soグループ数
    Gso = np.int(self.N/ngso);
    # soグループ分け
    gso = [i%Gso for i in self.game.players()];
    self.game.gso = gso;

  def plot(self,plt,t):
    return self.game.plot(plt,t);

# self organization game
# ui=中心が近くに来たらグループに等分される複数のえさ。nに比例する半径。別れないと届かない。
class MFforage(Problem):
  def __init__(self,n,space,xfood,ri,E,wall=0,anime=0):
    Problem.__init__(self,n);
    self.dbgi = -1;
    self.anime=anime; # 0:なし,1:mp4,2:gif
    self.space = space;
    self.xfood = xfood;
    self.delay=5; # gif delay
    self.ngroup = 1; # 最初は1group
    f = self.fgroup;
    u = self.ugforage;
    u_ = self.uforage_;

    if E<0:
      self.game = meanfield.ModularMFG(n,space,ri,f,u,u_,dbgi=self.dbgi);
    else:
      x0 = self.readpos("input/sopos.forage.txt");
      self.game = meanfield.ModularMFG(n,space,ri,f,u,u_,x0=x0,dbgi=self.dbgi);
      self.game.randomwalk(E);

  def showresult(self,prnt=True):
    return 0,0;


  def readpos(self,fname):
    ret = [];
    with open(fname) as f:
      for l in f:
	col = l.rstrip().split(",");
        ret.append(tuple(np.float(s) for s in col));
    #ret = np.array(ret);
    return ret;

  def prntdbg(self,a):
    print "### sw",self.game.sw(a);  

  def so(self):
    Gso = len(self.xfood);
    iup = [i for i in sorted(self.game.players(),key=lambda i:np.linalg.norm(np.array(self.game.x[i][1])-np.array([20,30])))];
    il = [i for i in sorted(self.game.players(),key=lambda i:np.linalg.norm(np.array(self.game.x[i][1])-np.array([10,10])))];
    ir = [i for i in sorted(self.game.players(),key=lambda i:np.linalg.norm(np.array(self.game.x[i][1])-np.array([30,10])))];
    # soグループ分け
    done = {};
    for j in range(0,self.N):
      g = j%Gso;
      if g==0:
        tmpi = iup;
      elif g==1:
        tmpi = il;
      else:
        tmpi = ir;
      for i in tmpi:
	if i not in done:
	  done[i] = g;
	  break;
    gso = [g for i,g in sorted(done.items(),key=lambda x:x[0])];
    #gso = [i%Gso for i in self.game.players()];
    self.game.gso = gso;

  # aは(group,x)
  def fgroup(self,a):
    g =  self.group(a);
    ret = {};
    gs = sorted(set(g));
    for ag in gs:
      xg = np.array([a[k][1] for k in range(len(a)) if (g[k]==ag) and (a[k][1][0] is not None)]);
      ret[ag] = np.mean(xg,0);
    return ret;

  def group(self,a):
    return [e[0] for e in a];

  def ng(self,i,a):
    gid = a[i][0];
    g = [e[0] for e in a if e[0]==gid];
    return len(g);

  def ugforage(self,i,a,f):  
    fs = f(a);
    ai = a[i];
    lmd = 100;
    #return self.uforage_(i,ai,a,fs)+lmd*self.jforage_(ai,a,fs); # ui=uli+lmd*J
    return self.uforage_(i,ai,a,fs)+lmd*self.N*self.jforage_(ai,a,fs); # WLUi=uli+lmd*N*J
  
  def uforage_(self,i,ai,a,fs):  
    dbgi = -1;
    gid = ai[0];
    fg = fs[gid];
    xi = ai[1];
    g = np.array([e[0] for e in a]);

    # 餌
    xf = self.xfood;
    nf = len(self.xfood);
    dfood = np.linalg.norm(xf[0]-xf[1]);
    rmax = (dfood-4)/2;
    ng = len(g[np.where(g==gid)]);
    rg = np.float(ng)/self.N*rmax;
    #rg = 0.4*rmax;

    ul = -np.abs(np.linalg.norm(xi-fg)-rg); # グループ項

    if i==dbgi:
      print "i,g,ng",i,gid,ng;

    '''
    nofood = -1000; # foodにありつけないペナルティ
    if ng < np.float(self.N)/nf: # food発掘に必要な最低人数
      return ul+nofood;
    '''

    return ul;

  def jforage_(self,ai,a,fs):
    ug = 0.;

    gid = ai[0];
    g = np.array([e[0] for e in a]);
    ng = len(g[np.where(g==gid)]);
    fg = fs[gid];
    nf = len(self.xfood);
    pg = np.float(ng)/self.N;
    opt = self.N/nf;
    #opt = 30;
    for axfood in self.xfood:
      d = np.linalg.norm(fg-axfood);
      if d < 1:
        d = 1.;
      #ug += (2000-((ng-opt)**2))/(d**2); # +じゃないとdが逆に作用する
      ug += (1-((pg-1./nf)**2))/(d**2); # +じゃないとdが逆に作用する
      #print (pg-1./nf)**2,pg;
      #ug += (opt**2-((ng-opt)**2))/(d**2);
    return ug;

  def plot(self,plt,t):
    self.game.plot(plt,t);
    # 上書き
    xs = np.array([x[0] for x in self.xfood]);
    ys = np.array([x[1] for x in self.xfood]);
    scale = 10./(self.space[0])*(self.space[1]);
    cval = np.ones(len(self.xfood));
    im = plt.scatter(xs+0.5,ys+0.5,s=12*scale*5**2,alpha=0.5,vmin=0.,vmax=1.,c=cval,cmap='bwr');
    return im;


# neighbor game
# ui=mean(xNi)からの距離+mean(xNi)同士の距離
class MFmodular(Problem):
  def __init__(self,n,space,wall=0,anime=0):
    Problem.__init__(self,n);
    self.anime=anime; # 0:なし,1:mp4,2:gif
    self.space = space;
    self.delay=20; # gif delay
    f = self.fgroup;
    #u = self.wlu;
    u = self.utricircle;
    self.game = meanfield.MeanField(n,space,f,u);

  def fgroup(self,x):
    n = len(x);
    ret = [];
    for j in range(3):
      g = j%3;
      xg = np.array([x[k] for k in range(self.N) if (k%3==g) and (x[k][0] is not None)]);
      ret.append(np.mean(xg,0));
    return ret;

  # WLU:間違い。neighbor以外もキャンセルできない
  def wlu(self,i,x,f):
    x_i = list(x);
    x_i[i] = [None,None];
    ui = self.utricircle(i,x,f);
    g = i%3;
    ext =0.;
    for j in range(self.N):
      if (j%3==g) and (j!=i):
        ext+= self.utricircle(j,x,f)-self.utricircle(j,x_i,f);
    return ui+ext;

  def utricircle(self,i,x,f):
    fs = f(x);
    g = i%3;
    fg = fs[g];
    rni = 8; rg = 8;
    uni = -np.abs(np.linalg.norm(x[i]-fg)-rni);
    ug = 0.;
    dg = [];
    for g1 in range(3):
      for g2 in range(3):
        if g1<g2:
	  ug += -np.abs(np.linalg.norm(fs[g1]-fs[g2])-rg);
	  dg.append(np.linalg.norm(fs[g1]-fs[g2]));
    #print i,dg,uni,ug;
    return uni+5*ug; # 重みつけないとug効かない

# 画像
# (0,0)から(m,n)
class MyImage():
  def __init__(self,path):
    if irid:
      self.w = [];
      self.path = 'input/'+path+'.txt';
      with open(self.path) as f:
        for l in f:
	  col = l.rstrip().split(",");
          self.w.append([np.float(e) for e in col]);
      self.w = np.array(self.w);
      return;

    self.path = path+'.png';
    #print path;
    #print os.popen('pwd').read();
    self.img = plt.imread(self.path)[:,:,1]; # iridではエラー
    m,n = self.img.shape;
    for y in range(0,n):
      ltmp = list(self.img[:,y]);
      ltmp.reverse();
      self.img[:,y] = np.array(ltmp);

    self.w = np.array(np.round(self.img),dtype=np.int);
    self.w = 1-self.w;
    '''
    self.w = np.zeros((m,n));
    for x in range(0,m):
      for y in range(0,m):
        #d = np.sqrt((x/100.0-0.8)**2+(y/100.0-0.2)**2);
        #self.w[x,y] = np.exp(-20*d);
        #self.w[x,y] = np.exp(-50*self.img[x,y]);
    '''
    self.w = self.w.T;
    #self.w[np.where(self.w == 0.)] = 1e-4;
    #self.img[20:60,0:20] = 1;

  def shape(self):
    if irid:
      return self.w.shape;
    return self.img.shape;

  def show(self):
    plt.imshow(self.img,cmap='gray'); #plt.show();

# 非対称mean field game。画像入力
# mean(x)を図の中心としたときのdrawing
class MFdrawing(Problem):
  def __init__(self,n,space,file,wall=0,anime=0):
    Problem.__init__(self,n);
    self.anime=anime; # 0:なし,1:mp4,2:gif
    self.space = space;
    self.delay=20; # gif delay
    self.xg = self.readimg("input/star50",0.9); # rateは縮小率
    self.xg = self.readimg("input/starnoface50",0.9); # rateは縮小率
    #self.xg = self.read(file); # 中心0。平均位置からの目標変位
    f = lambda x: np.mean(x,0); # 平均位置
    #u = MFcovering.ucover;
    u = self.udraw;
    u_ = self.udraw_;
    #print np.mean(x0,0); exit();
    self.game = meanfield.MeanField(n,space,f,u,u_);
    # 目標チェック
    if 0:
      x0 = self.xg+np.array(self.space)/2;
      self.game = meanfield.MeanField(n,space,f,u,u_,x0=x0);
      im = self.game.plot(plt,0);
      plt.savefig("output/tmp.png");exit();

  def readimg(self,file,rate):
    img = MyImage(file);
    w = img.w.shape;
    ret = np.array(np.where(img.w),dtype=np.float).T;
    if len(ret) > self.N:
      ret = gb.sample(ret,self.N);
    else:
      ret = gb.sampledup(ret,self.N);
    ret = np.array(ret);
    ret[:,0] *= 1.*rate*self.space[0]/w[0];
    ret[:,1] *= 1.*rate*self.space[1]/w[1]; # スケールをspaceの1/rateになるよう縮小
    center = np.array(self.space)/2.;
    ret -= np.mean(ret,0); # 中心を0にする
    return ret;    
    
  def read(self,file):
    path = file+'.pos';
    ret = [];
    with open(path) as f:
      for l in f:
	col = l.rstrip().split(",");
        ret.append([np.float(s) for s in col]);
    ret = gb.sample(ret,self.N);
    ret = np.array(ret);
    rate = 2.;
    ret[:,0] *= self.space[0]/400./rate;
    ret[:,1] *= self.space[1]/400./rate; # スケールをspaceの半分になるよう縮小
    center = np.array(self.space)/2.;
    ret -= np.mean(ret,0); # 中心を0にする
    #ret += center;
    #print ret;
    #print np.mean(ret,0),np.max(ret,0),np.min(ret,0); exit();
    return ret;

  # mean(x)を図の中心としたときの目的座標距離
  def udraw(self,i,x,f):
    fs = f(x);
    xi = x[i];
    return self.udraw_(i,xi,fs);

  def udraw_(self,i,xi,fs):
    xr = xi-fs;
    xerr = xr-self.xg[i];
    ui = -np.linalg.norm(xerr);
    #if (x[i][1] < 7) and (x[i][0] < 30):
    #  print "low i",i,x[i];
    #  exit();
    #if i==1:
    #  print i,"x",x[i],"xr",xr,"xg",self.xg[i],xerr,ui;
    #print i,x[i],f(x),xr,self.xg[i],xerr,ui;
    return ui;

  def ng(self,i,a):
    return self.N;

  def prntdbg(self,a):
    1;

  def plot(self,plt,t):
    return self.game.plot(plt,t);


# Meeting on a grid
class MeetingOnaGrid(Problem):
  def __init__(self,n,space,wall=0,usewlu=1,anime=0):
    Problem.__init__(self,n);
    self.coordgame = True;
    self.space = space;
    self.usewlu=usewlu; # WLUを使うかどうか
    self.anime=anime; # 0:なし,1:mp4,2:gif
    self.delay=20; # gif delay
    m = decmdp.DecMDP(n);
    #m = decpomdp.DecPOMDP(n);
    #m.setPO(2); # instant policy
    X = list(self.eachx());
    X = gb.sample(X,len(X)-wall);
    self.wall = set(self.eachx())-set(X);
    print "wall",self.wall;
    # stateは盤面＝位置profile。agentの区別必要？とりあえず区別しとく
    for xpf in itertools.product(*(X for i in range(n))):    
      id = m.add_state(mdp.State(m,xpf));
      #print id,m.g.node[id]['state'].code();

    m.A = range(5); # 4:stay, 0..3:i*pi/2
    for pf in itertools.product(*(m.A for i in range(n))):
      for o in m.g.nodes():
        oxpf = m.g.node[o]['state'].code();
	for spf in itertools.product(*(range(2) for i in range(n))): # noisy transition 成功か失敗か
	  dxpf = [None]*2; p = 1.;
	  for i in range(n):
	    ai = pf[i];
	    if spf[i] > 0:
	      p*=0.6;
	    else:
	      p*=0.4;
	      if ai<4: # stayは失敗しない
	        ai = self.opposite(ai); 
	    xi = oxpf[i];
            dxpf[i] = self.move(ai,xi);
	  dxpf = tuple(dxpf);
	  d = m.gdict[dxpf];
	  #print spf,pf,oxpf,dxpf,o,d,p;
	  #if (o == 1) and (pf==(2,4)):
	  #  print spf,o,d,pf,p;
	  m.add_action(o,d,pf,p);

    # rewardはstateだけできまる。pfは関係ない
    for pf in itertools.product(*(m.A for i in range(n))):
      for s in m.g.nodes():
        xpf = m.g.node[s]['state'].code();
	#cu = self.competeU(xpf); # 競合するケース
	cu = np.array([0]*2); # 共通利得のcoordination game
	if xpf[0]==xpf[1]:
          #print s,xpf,pf;
	  if self.coordgame:
            m.setU(s,pf,np.array([1.]*2)+cu);
	  else:
            m.setU(s,pf,np.array([1.,0.])+cu); # 競合ケース
	else:
	  if self.coordgame:
            m.setU(s,pf,np.array([0.]*2)+cu);
	  else:
            m.setU(s,pf,np.array([0.,1.])+cu); # 競合ケース

        '''
        # perfect observation
        for o in m.g.nodes():
	  if s==o:
	    m.setO(s,pf,o,1.);
	  else:
	    m.setO(s,pf,o,0.);
        '''


    #m.setSource(0); # 両方(0,0)
    src = gb.sample(list(m.states()),1)[0];
    m.setSource(src);
    m.fix();
    #m.show();
    #m.plot(w=3,h=3,edges=0);
    #print(m.opt());
    #for s in m.states():
    #  print s,m.g.node[s]['state'].code();


    self.game = m;

  # 各agentがlocal MDPを持つ
  def localSetting(self,view=1):
    #if view < 1:
    #  print "view invalid",view; exit();
    n = self.game.N;
    self.view = view;

    self.localgame = {};
    # 実際の位置profile
    for i in self.game.players():
      m = decmdp.DecMDP(n); # local MDP
      self.localgame[i] = m;
      for s in self.game.states():
        x = self.game.g.node[s]['state'].code();
	lx = self.localxpf(x,i,view); # local view
        id = m.add_state(mdp.State(m,lx)); 
	#print s,x,lx,id;

      m.A = self.game.A;
      denom = {o:{} for o in m.g.nodes()}; # 複数の実状態sが観測oに対応するので確率和が1を超えてしまう
      # 実際のゲームの遷移
      for o in self.game.g.nodes():
        oxpf = self.game.g.node[o]['state'].code();
	ooxpf = self.localxpf(oxpf,i,view); # 観測
	oo = m.gdict[ooxpf];
	#print i,o,oxpf,ooxpf,oo;
        for pf in itertools.product(*(m.A for i in range(n))):
	  #print i,o,oxpf,pf;
	  for spf in itertools.product(*(range(2) for i in range(n))): # noisy transition 成功か失敗か
	    dxpf = [None]*2; p = 1.;
	    for j in range(n):
	      aj = pf[j];
	      if spf[j] > 0:
	        p*=0.6;
	      else:
	        p*=0.4;
	        if aj<4: # stayは失敗しない
	          aj = self.opposite(aj); 
	      xj = oxpf[j];
              dxpf[j] = self.move(aj,xj);
	    dxpf = tuple(dxpf); # 実際の結果
	    d = self.game.gdict[dxpf];
	    odxpf = self.localxpf(dxpf,i,view); # 観測
	    od = m.gdict[odxpf];	    
 	    #print i,o,oxpf,pf,spf,dxpf;
	    #print i,d,dxpf,odxpf,od;
	    if pf not in denom[oo]:
	      denom[oo][pf] = 0.;
	    denom[oo][pf] += p;
	    #if (oo==4) and (pf==(0,0)):
	    #  print i,ooxpf,pf,odxpf,p,denom[oo][pf];
	    m.add_action(oo,od,pf,p);

      # 実初期状態に対応する初期状態
      osrc = self.obs(self.game.src,i);
      m.setSource(osrc);
      m.fix();

      # 確率正規化
      for o in m.g.nodes():
        for pf,T in m.g.node[o]['As'].items():
          for d,p in T.items():
	    T[d] /= denom[o][pf];

      # rewardはstateだけできまる。pfは関係ない
      for pf in itertools.product(*(m.A for i in range(n))):
        for s in m.g.nodes():
          xpf = m.g.node[s]['state'].code();
	  #cu = self.competeU(xpf); # 競合するケース
	  cu = np.array([0]*2); # 共通利得のcoordination game
	  if xpf[0]==xpf[1]:
            #print s,xpf,pf;
	    if self.coordgame:
              m.setU(s,pf,np.array([1.]*2)+cu);
	    else:
              m.setU(s,pf,np.array([1.,0.])+cu); # 競合ケース
	  else:
	    if self.coordgame:
              m.setU(s,pf,np.array([0.]*2)+cu);
	    else:
              m.setU(s,pf,np.array([0.,1.])+cu); # 競合ケース


  # 実sに対応する観測oを返す
  def obs(self,s,i):
    xpf = self.game.g.node[s]['state'].code();
    oxpf = self.localxpf(xpf,i,self.view); # 観測
    return self.localgame[i].gdict[oxpf];

  def localxpf(self,xpf,i,view):
    xi = np.array(xpf[i],dtype=np.int);
    d0 = [self.dist(x,(0,0)) if self.dist(x,xi) <= view else np.inf for x in xpf];
    i0 = np.where(d0 == np.min(d0))[0][0];
    x0 = np.array(xpf[i0],dtype=np.int); # 壁見えないときは、(0,0)に一番近いagentが0点
    if self.space[0]-1-xi[0] <= view: # 右上壁が見える
      x0[0] = self.space[0]-1-2*view-1; # view左端-壁までの距離=壁-2*viewよりもう一個小さいのが0点
    if self.space[1]-1-xi[1] <= view: # 右上壁が見える
      x0[1] = self.space[1]-1-2*view-1; # view左端-壁までの距離=壁-2*viewよりもう一個小さいのが0点
    if xi[0]-0 <= view: # 左下壁が見える
      x0[0] = 1; # 壁が0点
    if xi[1]-0 <= view:
      x0[1] = 1; # 壁が0点
    x0-= 1;
    ret = [];
    for ix,x in enumerate(xpf):
      if d0[ix] < np.inf:
        lx = tuple(np.array(x,dtype=np.int)-x0);
      else:
        lx = None;
      ret.append(lx);
    return tuple(ret);

  def dist(self,x,y):
    return np.linalg.norm([x[0]-y[0],x[1]-y[1]]);

  def competeU(self,xpf):
    # p0は(0,0), p1はspace-1に近いほうが利得高い
    xm = np.array([w-1 for w in self.space]);
    return np.array([sum(xm-np.array(xpf[0])),sum(np.array(xpf[1])-np.array([0,0]))])*0.3;

  def eachx(self):
    for x in itertools.product(*(range(dm) for dm in self.space)):
      yield x;

  def move(self,ai,xi):
    if ai == 4:
      return xi;
    th = ai*np.pi/2;
    v = np.array([np.cos(th),np.sin(th)],dtype=np.int);
    ret = np.array(xi)+v;
    for dm in range(2):
      if ret[dm] < 0:
        ret[dm] = 0;
      if ret[dm] > self.space[dm]-1:
        ret[dm] = self.space[dm]-1;
    ret = tuple(ret);
    if ret in self.wall:
      return xi;
    return ret;

  # 逆方向
  def opposite(self,ai):
    if ai==4:
      return ai;
    return (ai+2)%4;

  def plot(self,plt,t,s):
    if self.anime==2:
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

    x = self.game.g.node[s]['state'].code();

    xs = np.array([x[i][0] for i in self.game.players()]);
    ys = np.array([x[i][1] for i in self.game.players()]);

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

# Grid
class SwarmGrid(Problem):
  def __init__(self,n,space=(1,1),task=0,anime=0):
    Problem.__init__(self,n);
    self.t = 0;
    self.space = space;
    self.task=task;
    self.delay=20; # gif delay
    self.anime=anime; # 0:なし,1:mp4,2:gif
    self.th0 = np.round(np.random.rand(n)*3); # 初期方位
    self.S = np.random.rand(n,4); # 状態。位置、方位、速さ
    self.S[:,0]*=(space[0]-1);
    self.S[:,1]*=(space[1]-1);
    while not self.rearrange(): # 最初にぶつかってるのを直す
      1;    
    self.S[:,2]*=self.th0; # 初期方位:*np.pi/2でx軸からの角度(反時計回り)
    self.S=np.round(self.S);
    self.S[:,3]=1; # 0か1
    self.newS = np.zeros((self.N,2));    
    self.A = list(range(4)); # action:初期方位と比べて、*np.pi/2の角度(反時計回り)
    self.X = gb.sampledup(self.A,self.N);
    #self.X = np.ones(self.N,dtype=np.int)*self.A[0]; # Action profile

    self.ui = self.Ni;

  def show(self):
    for i in range(self.N):
      print i,self.S[i,:];

  def rearrange(self):
    # 距離行列
    x = self.S[:,0:2];
    D = distance.cdist(x,x, metric='euclidean');
    for i in range(self.N):
      D[i,i] = np.inf;

    col = np.where(D < 1);
    ncol = len(col[0]);
    if ncol == 0:
      return True;
    print(len(col[0]),"col. rearrange"); sys.stdout.flush();
    for ind,tmp in enumerate(col[0]):
      i,j = col[0][ind],col[1][ind];
      self.S[i,0:2] = np.random.rand(1,2);
      self.S[i,0]*=np.round(self.space[0]-1);
      self.S[i,1]*=np.round(self.space[1]-1);

    return False;

  def state(self):
    ret = tuple(tuple(np.int(e) for e in np.round(self.S[i,0:2])) for i in range(self.N));
    if 1 and any([e > 1 for e in Counter(ret).values()]):
      print ret,False;
      exit();
    return ret;

  def overlap(self):
    stat = self.state();
    return any([e > 1 for e in Counter(stat).values()]);

  def torus(self,x):
    e = self.err;
    ret = np.zeros(2);
    ret[0] = x[0]; ret[1] = x[1];
    if x[0] < -e:
      ret[0] += self.space[0];
    elif x[0] > self.space[0]-1+e:
      ret[0] -= self.space[0];
    if x[1] < -e:
      ret[1] += self.space[1];
    elif x[1] > self.space[1]-1+e:
      ret[1] -= self.space[1];
    return ret;

  def getv(self,i):
    th = self.S[i,2]*np.pi/2;
    return np.array([np.cos(th),np.sin(th)])*self.S[i,3];

  # torus前後の最小距離
  def torusDist(self,x,y,prnt=False):
    N,M = x.shape;
    ret = np.zeros((N,N));
    for i in range(N):
      for j in range(N):
        xnew = self.torus(x[i,:]);
	ynew = self.torus(y[j,:]);
        dxy = util.norm(x[i,:]-y[j,:]);
        dxnewy = util.norm(xnew-y[j,:]);
        dxynew = util.norm(x[i,:]-ynew);
        dxnewynew = util.norm(xnew-ynew);
	if 0 and prnt:
	  print i,j,x[i,:],y[j,:];
	  print " ",xnew,ynew;
	  print " ",dxy,dxnewy,dxynew,dxnewynew;
        ret[i,j] = np.min([dxy,dxnewy,dxynew,dxnewynew]);
    return ret;

  # dtだけ時間を進める
  def game(self,dt,pf,prnt=False):
    dbg = 0;
    self.t += dt;  
    if prnt:
      print "gamepf",pf;
    # 方位update
    # action:初期方位と比べて、*np.pi/2の角度(反時計回り)
    self.S[:,2] = self.th0+pf;

    # 衝突判定移動
    for i in range(self.N):
      self.newS[i,:] = self.S[i,0:2]+self.getv(i)*0.6;
      if dbg and prnt:
        print "colchk",i,self.S[i,:],self.newS[i,:];

    # 距離行列:(トーラス考慮)
    x = self.newS;
    D = self.torusDist(x,x,prnt=prnt);
    #D = distance.cdist(x,x, metric='cityblock'); # マンハッタン距離
    #D = distance.cdist(x,x, metric='euclidean');
    for i in range(self.N):
      D[i,i] = np.inf;

    # ロボ間衝突検出/速度調整
    col = np.where(D < 1-self.err);
    col = [list(acol) for acol in col];
    if len(col[0]) > 0:
      for ind,tmp in enumerate(col[0]):
        i,j = col[0][ind],col[1][ind];
        self.S[i,3] = 0.; # 臨時。衝突したら止まる

    if dbg and prnt:
      print "chkD";
      print D;
      print col;
    self.col = col;

    # 壁に衝突したらトーラス
    for i in range(self.N):
      self.newS[i,:] = self.torus(self.newS[i,:]);
      if dbg and prnt:
        print "torus colchk",i,self.newS[i,:];
    
    '''
    # aggregate計算
    self.aggregate = np.array([Counter(col[0])[i] for i in range(self.N)]); # 衝突人数
    '''

    # 玉突き衝突チェック
    prestat=None;
    while True:
      for i in range(self.N):
        self.newS[i,:] = self.S[i,0:2]+self.getv(i)*dt;
        if dbg and prnt:
          print "chain",i,dt,self.S[i,:],self.newS[i,:];
      stat = tuple(tuple(np.int(e) for e in np.round(self.newS[i,:])) for i in range(self.N));
      if dbg and prnt:
        print "chain check",stat;
      # 距離行列:(トーラス考慮)
      x = self.newS;
      D = self.torusDist(x,x,prnt=prnt);
      for i in range(self.N):
        D[i,i] = np.inf;
      # ロボ間衝突検出/速度調整
      col = np.where(D < 1-self.err);
      col = [list(acol) for acol in col];    
      if len(col[0]) > 0:
        for ind,tmp in enumerate(col[0]):
          i,j = col[0][ind],col[1][ind];
          self.S[i,3] = 0.; # 臨時。衝突したら止まる
	
      if dbg and prnt:
        print "chainD";
        print D;
        print col;

      self.col[0] += col[0];
      self.col[1] += col[1];
      if stat == prestat:
        break;
      prestat = stat;

    # 移動
    for i in range(self.N):
      self.newS[i,:] = self.S[i,0:2]+self.getv(i)*dt;
      if dbg and prnt:
        print "move",i,self.S[i,:],self.newS[i,:];
    for i in range(self.N):
      self.newS[i,:] = self.torus(self.newS[i,:]);
      if dbg and prnt:
        print "torus",i,self.newS[i,:];
    for i in range(self.N):
      self.S[i,0:2] = self.newS[i,:];

    # 速度回復
    self.S[:,3] = 1;

    return True;

  def neighbors(self,i):
    col = np.where(np.array(self.col[0]) == i)[0];
    for jind in col:
      yield self.col[1][jind];

  def Ni(self,i):
    ret = list(self.neighbors(i));
    return len(ret)-self.N; # 負に正規化

  def intriangle(self,plt):
    r = 0.5;
    #th0 = [np.pi/2,0,-np.pi/2,np.pi];
    for i in range(self.N):
      x,y= [],[]
      # 方位:*np.pi/2でx軸からの角度
      for j in np.linspace(0, 2 * np.pi, 4):
        #th = j+th0[np.int(self.S[i,2])];
	th = j+(1-self.S[i,2])*np.pi/2;
	#print [j,(self.S[i,2]+1)];
	#th = j;
        x.append(np.sin(th)*r+self.S[i,0]+r);
        y.append(np.cos(th)*r+self.S[i,1]+r);
      plt.plot(x,y,color='r');

  def plot(self,plt):
    if self.anime==2:
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
    # 内接三角形
    self.intriangle(plt);
    
    # 円
    im = plt.scatter(self.S[:,0]+0.5,self.S[:,1]+0.5,s=12*scale*5**2,alpha=0.5,vmin=0.,vmax=1.,c=cval,cmap='bwr');
    # 時刻
    #plt.text(1,self.space[1],"t:%d"%self.t,fontsize=10,ha='center',va='center');
    plt.title("t:%d"%self.t);
    
    if 0: # アニメ生成がめちゃ遅くなる。gifならok
      for i in range(self.N):
        plt.text(self.S[i,0],self.S[i,1],i,fontsize=10,ha='center',va='center');
        #plt.annotate(i,(self.S[i,0],self.S[i,1]));
    #plt.colorbar(im);
    #plt.show();
    return im;
    
# Douchan2019
class Swarm(Problem):
  def __init__(self,n,space=(1,1),r=5.,epi=0., maxe=None,Alen=2,ufunc=0,task=0,gif=0):
    Problem.__init__(self,n,maxe);
    self.space = space;
    self.r = r; # 半径
    self.v = 1.; # 速度一律
    self.A0 = 5; # 回避時間
    self.epi =epi; # 移動ノイズ
    self.task=task;
    self.gif=gif;
    self.delay=5; # gif delay    
    self.S = np.random.rand(n,4); # 状態。位置と方位と速さの4次元分布
    self.S[:,0]*=(space[0]-2*self.r);
    self.S[:,0]+=self.r;
    self.S[:,1]*=(space[1]-2*self.r);
    self.S[:,1]+=self.r;
    self.S[:,2]*=2*np.pi; # [0,2pi]。th=0がx軸方向
    while not self.rearrange(): # 最初にぶつかってるのを直す
      1;
    #self.debug3();
    self.S[:,3]=self.v;
    self.newS = np.zeros((self.N,2));
    self.l = 0.; # 衝突サイクル
    self.lp = np.zeros(self.N); # passive time;
    self.na = Counter(); # 近傍数
    self.prena = Counter(); # 前回衝突時の近傍数
    self.active = np.zeros(self.N); # 回避中
    self.averting = np.zeros(self.N); # 残り回避時間
    self.mileage = np.zeros(self.N); # 総移動時間
    self.aggregate = np.zeros(self.N); # 近接人数
    self.namecard = [{} for i in range(self.N)]; # 名刺
    self.A = [0,1,2]; # Action set
    #self.A = [0,1]; # Action set
    self.A = list(range(Alen));
    #self.X = gb.sampledup(self.A,self.N);
    self.X = np.ones(self.N,dtype=np.int)*self.A[0]; # Action profile
    if ufunc==0: # reward function
      self.ui = self.wlu;
    elif ufunc==1:
      self.ui = self.tg;
    elif ufunc==2:
      self.ui = self.su;

    if task == 2:
      self.ui = self.Ni;

  # 初期配置でぶつかってるのを直す
  def rearrange(self):
    # 距離行列
    x = self.S[:,0:2];
    D = distance.cdist(x,x, metric='euclidean');
    for i in range(self.N):
      D[i,i] = np.inf;

    col = np.where(D < self.r);
    ncol = len(col[0]);
    if ncol == 0:
      return True;
    print(len(col[0]),"col. rearrange"); sys.stdout.flush();
    for ind,tmp in enumerate(col[0]):
      i,j = col[0][ind],col[1][ind];
      self.S[i,0:2] = np.random.rand(1,2);
      self.S[i,0]*=(self.space[0]-2*self.r);
      self.S[i,0]+=self.r;
      self.S[i,1]*=(self.space[1]-2*self.r);
      self.S[i,1]+=self.r;

    return False;

  def debug(self):
    self.N = 2;
    self.epi =0;
    self.r = 5.;
    self.S = np.random.rand(self.N,4);
    self.S[:,1] = 10;

    self.S[0,0] = 50;
    self.S[1,0] = 53;
    self.S[0,2] = np.pi/2;
    self.S[1,2] = np.pi/2;

    self.S[0,0] = 5;
    self.S[1,0] = 45;
    self.S[0,2] = np.pi/4;
    #self.S[0,2] = np.pi/6;
    self.S[1,2] = 3*np.pi/4;

  def debug3(self):
    self.N = 3;
    self.epi =0;
    self.r = 5.;
    self.S = np.random.rand(self.N,4);
    self.S[:,1] = 10;
    self.S[2,1] = 10+10*np.sqrt(3);

    self.S[0,0] = 10;
    self.S[1,0] = 30;
    self.S[2,0] = 20;

    self.S[0,2] = np.pi/4;
    #self.S[0,2] = np.pi/6;
    self.S[1,2] = 3*np.pi/4;
    self.S[2,2] = 3*np.pi/2;

  def debug4(self):
    self.N = 4;
    self.epi =0;
    self.r = 5.;
    self.S = np.random.rand(self.N,4);
    self.S[0:2,1] = 10;
    self.S[2:4,1] = 50;

    self.S[[0,2],0] = 10;
    self.S[[1,3],0] = 50;
    self.S[0,2] = np.pi/4;
    self.S[1,2] = 3*np.pi/4;
    self.S[2,2] = 7*np.pi/4;
    self.S[3,2] = 5*np.pi/4;

  def debug5(self):
    self.N = 5;
    self.epi =0;
    self.r = 5.;
    self.S = np.random.rand(self.N,4);
    rad = np.pi/180;
    a = 54*rad;
    b = np.pi-2*a;
    r = b/2;
    e = np.pi/2-b;
    self.S[[0,1],1] = 10;
    self.S[[2,4],1] = 10+40*np.sin(b);
    self.S[3,1] = 10+20*np.sin(a)/np.sin(r)+20/np.sin(r);
    
    self.S[0,0] = 40*np.sin(e);
    self.S[1,0] = 40+40*np.sin(e);
    self.S[2,0] = 40+80*np.sin(e);
    self.S[3,0] = 20+40*np.sin(e);
    self.S[4,0] = 0;
    self.S[0,2] = a;
    self.S[1,2] = np.pi-a;
    self.S[2,2] = 3*np.pi/2-e-a;
    self.S[3,2] = 3*np.pi/2;
    self.S[4,2] = 3*np.pi/2+e+a;

  def reset(self):
    self.l = 0.;
    self.lp = np.zeros(self.N); # passive time;
    self.prena = self.na;
    self.na = Counter(); # 近傍数

  # debug用congestion game
  def gamedebug(self,dt,pf):
    self.na = np.zeros(self.N,dtype=np.int); # 状態なし
    self.l = self.N;
    self.active = np.ones(self.N);
    self.mileage = np.zeros(self.N);
    for i in range(self.N):
      x = self.congestion(pf,i); # 混雑
      self.lp[i] = self.N-x;
      self.mileage[i] = self.ui(i);
    return True;

  # dtだけ時間を進める
  def game(self,dt,pf):
    # 利得変数カウント
    self.l += dt;
    self.t += dt;
    self.lp[np.where(self.active==0)]+=dt;
    self.mileage[np.where(self.active==0)]+=dt; # 速度1
  
    # 移動
    for i in range(self.N):
      #self.S[:,0:2]+= self.S[:,3]*np.array([np.cos(self.S[:,2]),np.sin(self.S[:,2])]).T;
      th = self.S[i,2];
      if self.epi > 0:
        th += (np.random.rand()-0.5)*self.epi;
      self.newS[i,:] = self.S[i,0:2]+self.S[i,3]*np.array([np.cos(th),np.sin(th)])*dt; # 速度1
      #print i,self.S[i,:],self.newS[i,:];

    # 壁に衝突したら鏡面反射/トーラス
    for i in range(self.N):
      #self.mirror(i);
      self.torus(i);

    # 距離行列:newSで計算してるのでトーラス対策になってる
    x = self.newS;
    D = distance.cdist(x,x, metric='euclidean');
    for i in range(self.N):
      D[i,i] = np.inf;

    # 回避動作終わってたら取り敢えず回避状態終了
    self.active[np.where(self.averting == 0)] = 0;

    # ロボ間衝突検出
    col = np.where(D < self.r);
    iscol=False;
    if len(col[0]) > 0:
      iscol=True;
      for ind,tmp in enumerate(col[0]):
        i,j = col[0][ind],col[1][ind];
        self.namecard[i][j] = 1;
        self.namecard[j][i] = 1;
        self.na[i] += 1; # 近傍
        # 衝突起きたら取り敢えず止まって回避状態に入る
        #print "crush",i,j,D[i,j],self.newS[i,0:2],self.newS[j,0:2];
        self.active[i] = 1;
        self.S[i,3] = 0.;
        if self.averting[i] > 0: # 回避動作中
	  continue;
	self.averting[i] = self.A0; # 回避動作時間

    self.col = col;
    # aggregate計算
    self.aggregate = np.array([Counter(col[0])[i] for i in range(self.N)]); # 衝突人数

    for i in range(self.N):
      #if (self.active[i] > 0) and (self.na[i] == 0):
      #  print i,self.active[i],self.averting[i];
      #  print D; exit();
      if np.abs(self.S[i,3]) > 0: # 衝突しなければ動く
        #self.S[i,0:2] = copy.deepcopy(self.newS[i,:]);
        self.S[i,0:2] = self.newS[i,:];
      if self.averting[i] > 0:
        # 止まってるやつは回避動作を行う(入力のpfを使う)
        self.avert(i,pf[i],dt,D);
      else:
        # 回避状態にない
        self.S[i,3] = self.v;
	if 0 and self.task ==2: # aggregate
	  self.gather(i,D);

    return iscol;

  # avertingだけいじる。activeはいじっちゃダメ
  # self.active[i] # 回避中
  # self.averting # 残り回避時間
  # self.S # 状態[x,y,th,v]。thは[0,2pi]
  def avert(self,i,a,dt,D):
    if a == 0:
      # そのまま動く
      self.averting[i] = 0;
      self.S[i,3] = self.v;
    elif a == 1:
      # 下がる
      if self.S[i,3] >=0:
        self.S[i,3] = -self.v;
      self.averting[i] -= dt;
      if self.averting[i]<=0.:
        # 終わった
	self.averting[i] = 0;
    else:
      #self.stop(i); # 止まる
      #self.bestevade(i,D);
      self.gather(i,D);
      #self.randdir(i);
      self.averting[i] = 0;

  def randdir(self,i):
    self.S[i,2] = np.random.rand()*2*np.pi;
    self.S[i,3] = self.v;
    
  def stop(self,i):
    self.S[i,3] = 0;

  def bestevade(self,i,D):
    # 向きを変えてすぐ動く
    self.S[i,3] = self.v;
    visrate = 0.1; # spaceの対角線の何倍見えるか
    spacediag = np.sqrt(self.space[0]**2+self.space[1]**2);
    self.S[i,2] = self.bestEvadeDir(i,spacediag*visrate,D); # 向きをget

  def gather(self,i,D):
    visrate = 0.3; # spaceの対角線の何倍見えるか
    spacediag = np.sqrt(self.space[0]**2+self.space[1]**2);	
    dir = self.gatherDir(i,spacediag*visrate,D);
    if dir is not None:
      self.S[i,2] = dir;
    self.S[i,3] = self.v;


  # Alphabetsoup:getBestEvadeDirection
  def bestEvadeDir(self,i,visd,D):
    # 方角index
    Th = 30; # 分解能
    dth = 2*np.pi/Th;
    i2th = np.arange(0,Th)*dth;
    th2i = {}; thdist = {};
    for thi,th in enumerate(i2th):
      th2i[(th,th+dth)] = thi;
      thdist[(th,th+dth)] = [];
      
    # 方角サーチ
    for j in range(self.N):
      if j==i:
        continue;
      if D[i,j] > visd:
        continue;
      th = util.azimuth(self.newS[i,0:2],self.newS[j,0:2]); # トーラス対策のため、newSで計算
      th0 = i2th[np.where((i2th<=th) & (i2th>th-dth))][0];
      thdist[(th0,th0+dth)].append(D[i,j]);
      #print j,self.S[j,0:2],self.S[j,0:2]-self.S[i,0:2],D[i,j],th;

    # gap検出
    hgap = [];
    for thi,(th,ds) in enumerate(sorted(thdist.items(), key=lambda x:x[0][0])):
      if len(ds) > 0:
        hgap.append((th,np.mean(ds)));
        #print thi,th,np.mean(ds);
    for thi,(th,ds) in enumerate(sorted(thdist.items(), key=lambda x:x[0][0])):
      if len(ds) > 0:
        hgap.append((th,np.mean(ds)));
	break;

    maxgapi = None; maxgapd = 0; retdir = None;
    for igap in range(len(hgap)-1):
      meand = (hgap[igap][1]+hgap[igap+1][1])/2;
      gapsize = hgap[igap+1][0][0]-hgap[igap][0][1];
      if gapsize < 0:
        gapsize += 2*np.pi;
      #print i,igap,hgap[igap][0],hgap[igap+1][0],meand,gapsize;
      if meand*gapsize >maxgapd:
        maxgapi = igap;
	maxgapd = meand*gapsize;
	retdir = hgap[igap][0][0]+gapsize/2;
	if retdir > 2*np.pi:
	  retdir -= 2*np.pi;
    #print "max",i,maxgapi,hgap[maxgapi],retdir;
    if retdir is None:
      return 0.;
    return retdir;

  def gatherDir(self,i,visd,D):
    # 方角index
    Th = 30; # 分解能
    dth = 2*np.pi/Th;
    i2th = np.arange(0,Th)*dth;
    th2i = {}; thcnt = {};
    for thi,th in enumerate(i2th):
      th2i[(th,th+dth)] = thi;
      thcnt[(th,th+dth)] = 0;
      
    # 方角サーチ
    for j in range(self.N):
      if j==i:
        continue;
      if D[i,j] > visd:
        continue;
      th = util.azimuth(self.newS[i,0:2],self.newS[j,0:2]); # jの方角。トーラス対策のため、newSで計算
      th0 = i2th[np.where((i2th<=th) & (i2th>th-dth))][0]; # 離散化した方角
      thcnt[(th0,th0+dth)]+=1;
      #print j,self.S[j,0:2],self.S[j,0:2]-self.S[i,0:2],D[i,j],th,th0;

    # 一番多い方角
    for thi,(th,cnt) in enumerate(sorted(thcnt.items(), key=lambda x:-x[1])):
      if cnt > 0:
        return np.mean(th);
      break;

    return None;


  def tg(self,i):
    ret = 0.;
    for i in range(self.N):
      ret += self.lp[i]/self.l-1.; # 負にする
    return ret/self.N;

  def su(self,i):
    return self.lp[i]/self.l-1; # 負にする

  def wlu(self,i):
    la = self.l-self.lp[i];
    ret = -(la+self.na[i]*self.A0)/self.l;
    #print i,self.active[i],la,self.na[i]*self.A0,ret;
    return ret;

  def neighbors(self,i):
    col = np.where(self.col[0] == i)[0];
    for jind in col:
      yield self.col[1][jind];

  def Ni(self,i):
    ret = list(self.neighbors(i));
    return len(ret)-self.N; # 負に正規化

  def torus(self,i):
    if self.newS[i,0] < 0.:
      self.newS[i,0] += self.space[0];
    elif self.newS[i,0] > self.space[0]:
      self.newS[i,0] -= self.space[0];
    if self.newS[i,1] < 0.:
      self.newS[i,1] += self.space[1];
    elif self.newS[i,1] > self.space[1]:
      self.newS[i,1] -= self.space[1];

  def mirror(self,i):
    if (self.newS[i,0]-self.r < 0.) or (self.newS[i,0]+self.r > self.space[0]):
      self.newS[i,2] = self.mirror_(self.newS[i,2],1);
    if (self.newS[i,1]-self.r < 0.) or (self.newS[i,1]+self.r > self.space[1]):
      self.newS[i,2] = self.mirror_(self.newS[i,2],0);
	
  def mirror_(self,th,ax):
    if ax == 0: # x軸対称
      th = -th;
    if ax == 1: # y軸対称:pi/2回転して鏡面して戻す
      th += np.pi/2;
      th = -th;
      th -= np.pi/2;

    if th < 0.:
      th += 2*np.pi;
    return th;

  def plot(self,plt):
    if self.gif:
      fig = plt.figure(1)
      fig.set_size_inches( (3.0, 3.0) )
      plt.clf();
      plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False);
      plt.tick_params(bottom=False,
                left=False,
                right=False,
                top=False);      
      #plt.cla();
      #gs = GridSpec(1,1)
      #ax = plt.subplot(gs[0,0]);
      
    plt.xlim([0,self.space[0]]);
    plt.ylim([0,self.space[1]]);
    cval = np.array(self.active);
    scale = 100./np.sqrt(self.space[0]*self.space[1]);
    im = plt.scatter(self.S[:,0],self.S[:,1],s=15*scale*self.r**2,alpha=0.5,vmin=0.,vmax=1.,c=cval,cmap='bwr');
    if 0: # アニメ生成がめちゃ遅くなる。gifならok
      plt.text(5,45,"t:%d"%self.t,fontsize=10,ha='center',va='center');
      for i in range(self.N):
        plt.text(self.S[i,0],self.S[i,1],i,fontsize=10,ha='center',va='center');
        #plt.annotate(i,(self.S[i,0],self.S[i,1]));
    #plt.colorbar(im);
    #plt.show();
    return im;

class MixedZerosum(Problem):
  def __init__(self,N,M):
    Problem.__init__(self,N);
    self.M = M;
    Ai = range(0,M);
    self.A = np.array([Ai]*self.N);
    
    self.X = np.random.rand(M,N); # ランダムmixed
    for i,a in self.agents():
      self.X[:,i] /= np.sum(self.X[:,i]);
    #self.X = np.ones((M,N))/M; # 一様分布

  # mixedを実行し、pfをサンプリング
  def run(self):
    ret = np.zeros(self.N,dtype=np.int);

    for i,a in self.agents():
      freq = np.random.multinomial(1,self.X[:,i]); # 一回転がす
      #print i,self.X[:,i],freq,np.where(freq > 0)[0][0];
      ret[i] = np.where(freq > 0)[0][0];
    return ret;

class Penny(MixedZerosum):
  def __init__(self):
    MixedZerosum.__init__(self,2,2);

  def ui(self,a,i):
    x = self.congestion(a,i)-1; # 同じだと1違うと0
    if i == 0:
      return 2*x-1;
    return -2*x+1;

# graphical potential game
class GPG(Problem):
  def __init__(self,n,c,maxe=None):
    Problem.__init__(self,n,maxe);
    self.C = c;
    self.M = np.int(self.N/self.C);
    Ai = range(0,self.M);
    self.A = np.array([Ai]*self.N); # 全エージェント共通
    
    #self.X = np.array([self.actions(i).next() for i,a in self.agents()]); # 全員自分の第一選択肢
    self.X = np.ones(self.N)*0.5; # 一様分布(binary choiceの場合)

    self.fx = lambda x:1.-self.fxshare(x);
    self.fx = lambda x:self.fxpeaky(x);

    g = nx.Graph();
    g.add_edge(0,2);
    g.add_edge(1,2);
    g.add_edge(3,2);
    g.add_edge(3,1);

    self.game = gpg.GPG(self.A,g,fx=self.fx);
    #self.game = gpg.GPG(self.A,fx=self.fx);
    

  # mixedを実行し、pfをサンプリング(binarychoiceの場合)
  def run(self,x):
    return np.array([1 if np.random.rand() < p else 0 for p in x]);

  def ui(self,a,i):
    return self.game.calU(a,i);

  def drawu(self):
    #x = np.arange(1,self.N);
    x = np.arange(1,12);
    plt.plot(x,map(lambda x: self.fx(x),x));
    plt.ylim([0,4]);
    plt.show();


  def fxshare(self,x):
    C = np.float(self.C);
    if x <= C:
      return 1./x;
    else:
      p = C/x; # 乗れる確率
      return p/C+(1-p);
      
  def fxpeaky(self,x):
    if x == self.C:
      return 4.;
    if x == self.N:
      return 3.;
    if x == 1:
      return 2.;
    return 1.;

# local resource game
# Mリソースキャパ付き
# xi=pf={0,..,M-1}^N
# 最大化問題
class LRG(GPG):
  def __init__(self,n,m,c,r=None,maxe=None):
    Problem.__init__(self,n,maxe);
    self.C = c;
    '''
    self.A = range(0,m);
    #self.rsc2g(r,n,m); # player-resource2部グラフ
    #self.X = np.random.randint(len(self.A),size=self.N);
    #self.X = np.ones(self.N)*list(bipartite.sets(self.R)[1])[0];
    # 全員自分の第一選択肢
    self.X = np.array([self.actions(i).next() for i,a in self.agents()]);
    #self.X[5] = 2; # 呼び水
    #self.ui = self.uimono;
    self.ui = self.ushare;
    self.ui = self.uimulti;
    '''

    self.fx = lambda x:self.fxpeaky(x);
    self.fx = lambda x:1.-self.fxshare(x);

    self.game = lrg.LRG(n,m,r,fx=self.fx);

  def drawu(self):
    #x = np.arange(1,self.N);
    x = np.arange(1,12);
    #plt.plot(x,map(lambda x: 1-self.sharecost(x,1),x));
    plt.plot(x,map(lambda x: self.upeaky(x),x));
    plt.show();

  '''
  def drawr(self):
    G = self.R;
    X, Y = bipartite.sets(G);
    col = [0 if i in X else 1 for i in G.nodes()];
    pos = dict();
    pos.update((n, (0, i*10)) for i, n in enumerate(X));
    pos.update((n, (0.5, i*10)) for i, n in enumerate(Y));
    labels = {i:i for i in G.nodes()};
    nx.draw(G, pos=pos,node_color=col,cmap=plt.get_cmap('bwr'));
    nx.draw_networkx_labels(G,pos,labels);
    plt.show();  

  # エージェント-リソース2部グラフ
  # エージェントはリンクのあるリソースしか選択肢にならない
  # これでエージェント間グラフも決まる
  def rsc2g(self,r,n,m):
    # プレイヤ-リソースグラフ  
    A = sp.lil_matrix((n,m)); 
    if r is None:
      A = np.ones((n,m));
      A = sp.csr_matrix(A);      
    else:
      for i,choice in enumerate(r):
        for a in choice:
          A[i,a] = 1;
    #print A.todense();
    self.R = bipartite.from_biadjacency_matrix(A, create_using=None, edge_attribute="weight");

    if r is None:
      return; # Gは完全グラフ

    # emptyグラフ
    g = nx.create_empty_copy(self.G);
    X, Y = bipartite.sets(self.R);
    tmp = bipartite.projected_graph(self.R,X); # playerだけのグラフ
    for o,d in tmp.edges():
      g.add_edge(o,d);

    self.G = g;

  # Rの隣接リソースしか使えない
  def actions(self,i):
    for r in self.R.neighbors(i):
      yield r;

  # uc:一人コスト
  def sharecost(self,x,uc):
    C = np.float(self.C);
    if x <= C: # キャパ以下
      return uc/x;
    else:
      p = C/x; # 乗れる確率
      return p*uc/C+(1-p)*uc;

  def ushare(self,a,i):
    x = len(a[np.where(a==a[i])]); # 自分の選択の人数
    return 1-self.sharecost(x,1.);

  # 単峰
  def uimono(self,a,i):
    na = len(a[np.where(a==a[i])]); # 自分の選択の人数
    return -(na-self.C);

  def uquadratic(self,x):
    return -((x-5)**4-10*(x-2)**2-10*(x-8)**2+x);

  def ugauss(self,x):
    s = 1.;
    return 10*np.exp(-(x-self.C)**2/(2*s**2))+9*np.exp(-(x-8)**2/(2*s**2));

  def upeaky(self,x):
    if x == self.C:
      return 3.;
    if x == 8:
      return 2.;
    if x == 1:
      return 1.;
    return 0.;      
    
  # 複峰
  def uimulti(self,a,i):
    x = self.congestion(a,i);
    #x = len(a[np.where(a==a[i])]); # 自分の選択の人数
    return self.upeaky(x);
'''

# ２リソースキャパ付き
# xi=pf={0,1}^N
# 最大化問題
class Binary(Problem):
  def __init__(self,n,maxe,c):
    Problem.__init__(self,n,maxe);
    self.A = [0,1]; # バイナリ選択肢
    self.X = np.random.randint(len(self.A),size=self.N);
    self.C = c;
    #self.ui = self.uimulti;
    #self.ui = self.uilin;
    self.ui = self.uimono;
    #util.plotFunc(ui);

  # 線形
  def uilin(self,a,i):
    ai = a[i];
    if ai < 1: # 0なら
      a = 1-a; # 反転
    na = sum(a); # 自分の選択の人数
    return -(na-self.C);
	
  # 単峰
  def uimono(self,a,i):
    ai = a[i];
    if ai < 1: # 0なら
      a = 1-a; # 反転
    na = sum(a); # 自分の選択の人数
    #print " na",na,-(na-self.C)**2;
    return -(na-self.C)**2;

  # 複峰
  def uimulti(self,a,i):
    ai = a[i];
    if ai < 1: # 0なら
      a = 1-a; # 反転
    x = sum(a); # 自分の選択の人数
    return -((x-5)**4-10*(x-2)**2-10*(x-8)**2+x);

# ２リソースキャパ付き
# xi=pf=[0,1]^N: mixed戦略
# 最大化問題
class MixedBinary(Problem):
  def __init__(self,n,c):
    Problem.__init__(self,n);
    self.A = [0,1]; # バイナリ選択肢
    self.X = np.random.rand(self.N); # barに行く確率
    self.C = c;
    #util.plotFunc(ui);
    #self.ui = self.elfarol;
    self.ui = self.uimono;

  # mixedを実行し、pfをサンプリング
  def run(self,x):
    return np.array([1 if np.random.rand() < p else 0 for p in x]);

  # 単峰
  def uimono(self,a,i):
    ai = a[i];
    if ai < 1: # 0なら
      a = 1-a; # 反転
    na = sum(a); # 自分の選択の人数
    #print " na",na,-(na-self.C)**2;
    return -(na-self.C)**2;

  # El Farol Bar問題
  def elfarol(self,a,i):
    G = 2.;
    S = 1.;
    B = 0.;
    ai = a[i];
    nbar = sum(a);
    
    if ai < 1: # stay
      return S;
    if nbar < self.C:
      return G; # 空いてる
    return B;
