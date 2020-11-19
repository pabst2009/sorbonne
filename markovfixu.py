# -*- coding: utf-8 -*-

from markovgame import MarkovGame;
import gamebase as gb;
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
import platform;

# 状態xによって利得関数は変わらないが選択肢集合Aが変わる
class MarkovFixU(MarkovGame):
  def __init__(self,n,Ai,x0):
    MarkovGame.__init__(self,n,Ai,x0);
    #self.rl = self.erev;
    #self.rl = self.qlearn;
    self.rl = self.qlearnl;

  def brupreprocess(self,pf):
    1;

  def brualldev(self,i,abefore,aafter):
    1;

  # ポリシーではなくPFのBRU
  def bru(self,T,stopcycle=False):
    pf = self.initpf();
    self.brupreprocess(pf);    
    if self.gso is not None:
      for i in self.players():
        pf[i] = (self.gso[i],pf[i][1]);
    dbgi = self.dbgi;
    wupdated = True;
    cycle = False;
    hpf = {};
    t = 0;
    yield t,0,pf; t+=1;
    while wupdated: # だれかupdateしたら
      if cycle and stopcycle:
	print "cycle!";
        break;
      if 0 and (t >= T):
        break;
      wupdated = False;
      ss = [];
      self.brupreprocess(pf); # 全員devし終わったら?毎回devごとにすべし?
      for i in self.players():
        tmppf = {k:v for k,v in pf.items()}; # deepcp
        bestui = self.u(i,tmppf); # 各iごとにBRU
	if i in dbgi:
	  print "#### before update i,ai,ui",i,tmppf[i],bestui;
	for ai in self.Ai(i): # 現在のxでの選択肢
          if self.gso is not None:
	    ai = (self.gso[i],ai[1]);
	  tmppf[i] = ai;
	  tmpui = self.u(i,tmppf);

	  if i in dbgi:
	    print "####### i",i,pf[i],"->",tmppf[i],"u",bestui,"->",tmpui;	  
	  dui = tmpui-bestui;
	  if dui>0.:
	    #self.du(i,pf,tmppf); # デバッグ用
	    #df = self.df(tmppf,i);

	    #print "i,dui",i,dui,"dext",dext,"df,dext/df,Ndf,avg(dext/df)",np.array([df,dext/df,self.N*df,dext/df/(self.N-1)]);
            #print "update",i,pf[i],"->",tmppf[i],"u",bestui,"->",tmpui;
	    if 0 and (i==14):
	      print "#######",pf[i],"->",tmppf[i],"u",bestui,"->",tmpui;
	      print self.f(tmppf)*self.N;
	    print "i",i,"dui",dui;
	    
	    bestui = tmpui;
            self.brualldev(i,pf,tmppf);
            pf = {k:v for k,v in tmppf.items()}; # deepcp            
	    self.transition(pf); # devしたら毎回状態遷移
            #self.brupreprocess(pf); # 毎回devごとにすべし?(怪しいけどとりあえずそこまでしなくても動く)
	    if i in dbgi:
	      print "#### best",i,ai,tmpui;
	    #print t,i,pf,bestui;
	    #yield t,i,pf; t+=1;
	    #t += 1;
	    if tuple(pf) in hpf:
	      cycle = True;
	    hpf[tuple(pf)] = 1;
	    wupdated = True;
	    if cycle and stopcycle:
	      print "cycle! i",i;
	      break;
      yield t,i,pf; t+=1;

  # mixedを実行し、pfをサンプリング
  def sampleAi(self,i,xi):
    Ai = xi.keys();
    pi = [xi[ai] for ai in Ai];
    freq = np.random.multinomial(1,pi); # 一回転がす
    #print Ai;
    #print i,freq,np.where(freq > 0)[0][0],Ai[np.where(freq > 0)[0][0]]; exit();
    ret = Ai[np.where(freq > 0)[0][0]];
    
    return ret;

  # 線形近似Qlearning。Szepesvari2017reinforce.p60
  # GroupGridGame用
  def qlearnl(self,T=1000,forget=False):
    # 動径基底関数(RBF)
    # hはスケールパラメタ
    def rbf(xi,xd,h=1.):
      d = np.linalg.norm(np.array(xi)-np.array(xd));
      return np.exp(-d**2/h);

    # 特徴量基底関数:(xi,ai)空間をd次元ベクトルに圧縮する
    # カーネル平滑化
    def basekernel(xi,dai,x):
      yi = self.move(dai[1],xi,x); # xi,ai合わせて移動先yi。これを圧縮する
      w1,w2 = self.space;
      xd = [(0,0),(w1,0),(0,w2),(w1,w2)]; # アリーナ四隅
      gs = np.array([rbf(yi,axd,w1**2) for axd in xd]);
      gs = gs/sum(gs);
      return gs;

    # 位置がそのまま特徴量
    def basepos(xi,dai,x):
      yi = self.move(dai[1],xi,x); # xi,ai合わせて移動先yi。これを圧縮する
      w1,w2 = self.space;
      return 1.*np.array(yi)/np.array(self.space); # スケーリング
      #return np.array(yi);

    # 全エージェントの位置分布が特徴量
    # 各特徴点を最寄りとするagentの数
    # すごい遅い
    def basedist(xi,dai,x):
      w1,w2 = self.space;
      d = 4;
      dw1,dw2 = 1.*w1/d,1.*w2/d;
      xd = [(dw1*(ax+0.5),dw2*(ay+0.5)) for ax,ay in itertools.product(range(d),range(d))]; # 特徴点
      ret = np.zeros(d**2);
      for ax in x:
        ds = [rbf(ax,adx,w1**2) for adx in xd];
	ret[np.where(ds==np.max(ds))[0][0]]+=1;
      return ret/sum(ret);

    #base = basepos; d = 2;
    #base = basekernel; d = 4; # 特徴空間次元
    base = basedist; d=16;

    # 線形パラメタ初期化
    th = {i:np.zeros(d) for i in self.players()};

    # Q(xi,ai)=th*base(xi,ai)
    # (xi,ai) in X*Aは高次元だけど、base(xi,ai) in R^dはd次元
    def Qi(thi,xi,i,x):
      return {dai:np.dot(thi,base(xi,dai,x)) for dai in self.dAi(i)}; # Qの線形近似

    # mixed strategy(boltzmann exploration)
    def mixa(thi,xi,i,x):
      Qix = Qi(thi,xi,i,x); # 正規化したほうがいいかも
      b = 1.;
      den = sum([np.exp(b*aQ) for a,aQ in Qix.items()]);
      if den == 0:
        print "den 0 in boltzmann exploration in qlearnl"; exit();
      ret = {a:np.exp(b*aQ)/den for a,aQ in Qix.items()};
      return ret;

    gamma = 0.9; # 割引率(p3)。小さいほど未来を無視する
    alpha = 0.005; # ステップサイズ(p16)

    dbgi = [];

    for t in range(0,T):
      # mixed pfからpure pfをサンプリング
      x = [e[1] for e in self.x];
      pf = [self.sampleAi(i,mixa(th[i],self.x[i][1],i,x)) for i in self.players()];      
      pfa = [self.d2a(i,pf[i]) for i in self.players()];

      for i in self.players():
        r = self.u(i,pfa); # 各iごとにlearning。報酬rはマイナス
	ai = pfa[i]; dai = pf[i]; xi = self.x[i][1];
	yi = self.move(dai[1],self.x[i][1],x);
	pxi = base(xi,dai,x);
	thi = th[i];
	maxQiy = max([np.dot(thi,base(yi,dbi,x)) for dbi in self.dAi(i)]);
	# 学習
	Qixa = np.dot(thi,pxi);
	delta = r+gamma*maxQiy-Qixa;
	th[i] = thi+alpha*delta*pxi;

	if i in dbgi:
	  print "i,ai,dai,xi",i,ai,dai,xi;
	  print "thi",th[i];
	  #print "Qix",Qi(th[i],xi,i,x);
	  #print "mixa",mixa(th[i],xi,i,x);
	
      self.transition(pfa);

      yield t,i,pfa;

  # Qlearning。Szepesvari2017reinforce.p58
  # GroupGridGame用
  # 状態数の次元の呪いをモロ食らう
  def qlearn(self,T=1000,forget=False):
    # Qtable初期化
    # Q[i][xi][ai]
    Q = {i:{x:{a:0. for a in self.dAi(i)} for x in self.eachx()} for i in self.players()};

    # mixed strategy(boltzmann exploration)
    def mixa(Qix):
      b = 1.;
      den = sum([np.exp(b*aQ) for a,aQ in Qix.items()]);
      ret = {a:np.exp(b*aQ)/den for a,aQ in Qix.items()};
      return ret;

    gamma = 0.9; # 割引率(p3)。小さいほど未来を無視する
    alpha = 0.1; # ステップサイズ(p16)

    dbgi = [];

    for t in range(0,T):
      # mixed pfからpure pfをサンプリング
      pf = [self.sampleAi(i,mixa(Q[i][self.x[i][1]])) for i in self.players()];      
      pfa = [self.d2a(i,pf[i]) for i in self.players()];
      x = [e[1] for e in self.x];

      for i in self.players():
        r = self.u(i,pfa); # 各iごとにlearning。報酬rはマイナス
	ai = pfa[i]; dai = pf[i]; xi = self.x[i][1];
	yi = self.move(dai[1],self.x[i][1],x);
	Qix = Q[i][xi];
	Qiy = Q[i][yi];
	maxQiy = max([aQ for a,aQ in Qiy.items()]);
	dai2 = (ai[0],dai[1]);
	if dai2 not in Qix:
	  Qix[dai2] = 0.; # 初期化
	# 学習
	Qixa = Qix[dai2];
	delta = r+gamma*maxQiy-Qixa;
	Qix[dai2] = Qixa+alpha*delta;
	if dai[0] == -1: # 新規グループ発生も学習
	  Qixa = Qix[dai];
	  delta = r+gamma*maxQiy-Qixa;
	  Qix[dai] = Qixa+alpha*delta;

	if i in dbgi:
	  print "i,ai,dai,dai2,x",i,ai,dai,dai2,xi;
	  print "Qix",Qix;
	  
	
      self.transition(pfa);

      yield t,i,pfa;

  # reinforcement learning@duncan2008
  #  erev1998predicting
  #  状態のないじゃんけんゲーム用
  #  mixed strategyを更新する
  # forget onするとpNEに収束
  def erev(self,T=1000,forget=False):
    e = 0.01;
    e = 0.2; # mutation rate。全然収束しない。
    phi = 0.1; # 忘却率。収束しない
    phi = 0.01;

    # 初期はuniform mixed strategy(最初の一回しか使わない)
    # supportは、self.dAi(i)
    # グループのsupportは増えていくけどどうするか？→q=0と同じ
    x = {i:{ai:1. for ai in self.dAi(i)} for i in self.players()};
    if self.gso is not None:
      x = {i:{(self.gso[i],ai[1]):1. for ai in self.dAi(i)} for i in self.players()};
    for i in self.players():
      l = len(x[i]);
      for ai in x[i].keys():
        x[i][ai] /= l;

    # 累積利得は0
    q = {i:{ai:0. for ai in self.dAi(i)} for i in self.players()};    
    if self.gso is not None:
      q = {i:{(self.gso[i],ai[1]):0. for ai in self.dAi(i)} for i in self.players()};    

    dbgi = [];
    for t in range(0,T):
      # mixed pfからpure pfをサンプリング
      pf = [self.sampleAi(i,x[i]) for i in self.players()];      
      pfa = [self.d2a(i,pf[i]) for i in self.players()];
      for i in self.players():
        #ui = self.u(i,pfa); # 各iごとにlearning。uiはマイナス
        ui = self.u(i,pfa)+400; # 各iごとにlearning。uiはプラス
        if i in dbgi:
          print "qi",q[i];
	  print "xi",x[i];
          print "ai",pf[i],"ui",ui;
        if forget:
	  ai = pf[i]; # 試したaction
	  if ai not in q[i]:
	    q[i][ai] = 0.; # 初期化
          q[i][ai] = (1.-phi)*q[i][ai] + (1.-e)*ui; # (1')
	  for bi,qbi in q[i].items(): 
            if bi != ai: # その他のaction
              q[i][bi] = (1.-phi)*qbi + ui*e/len(q[i]); # (1')
	else:
          q[i][pf[i]] += ui; # reward累積
	if 0 and any(q[i][ai]>0 for ai in q[i].keys()):
	  print "positive utility",q[i],ui;
	  exit();
      for i in self.players():
        Ai = [ai for ai in q[i].keys()];
        qi = np.array([q[i][ai] for ai in Ai]);
        # qは常にプラス
	sumq = sum(qi);
        for ai in Ai:
	  x[i][ai] = q[i][ai]/sumq; # mixed strategy
	
        '''
        # qは常にマイナス
        sumq = -sum(qi);
	pui = sumq+qi;
	den = sum(pui);

        for ai in Ai:
	  if den > 0:
            x[i][ai] = (sumq+q[i][ai])/den; # mixed strategy更新
	  else:
	    x[i][ai] = -q[i][ai]/sumq; # mixed strategy更新(ここには来ない？)
        '''
        #print i,q[i],sumq,den;
	#print x[i],sum([x[i][ai] for ai in x[i].keys()]);
      self.transition(pfa);


      yield t,i,pfa;



