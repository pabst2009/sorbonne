# -*- coding: utf-8 -*-

import platform;
irid = (platform.system()=='Linux');

#import psutil;
#print psutil.cpu_count(logical=False);
import numpy as np;
import networkx as nx;
import os;
from scipy.spatial import distance;

if not irid:
  import matplotlib;
  matplotlib.use('TkAgg') # mac用
  import matplotlib.pyplot as plt;
  import matplotlib.animation as animation
  from matplotlib.gridspec import GridSpec
  from matplotlib.font_manager import FontProperties;
from collections import defaultdict,Counter;
import copy;
import util;
import itertools;
import solution;
import sys;
import gamebase as gb;
import time;
#print sys.version_info;
'''
import matplotlib as mpl;
print(mpl.matplotlib_fname()); # matplotlibrcの場所
mpl.rcParams['animation.convert_path'] = 'C:\home\bin\imagemagic\ImageMagick-7.0.8-Q16\magick.exe';
from wand.image import Image
'''
#print matplotlib.rcParams['animation.writer']; # ffmpegならok


# simでなく理論
# todo
# Qlearn:Belmanに基づいた動的計画法
#  achievable set@macdermed
# sharpley value
# BRU:シェアゲーはポテゲー？
#  hofbauerの微分ポテゲー
#    ボルツマン＋コンセンサスのPD gradient等価LRDでoptNEに収束？そんなことしなくてもBRUで収束？
# マトロイドBRU:southwell2012
# rims201807のM,L convex minアルゴ
#  M#凸最小化のscalingアルゴ:離散ゲー本
# 完全ユニモジュラ等価性とdouble well/level setを使ったblack white離散解を得る合意勾配法
# WLU：実装できる？何が課題？
#  ライブラリ:open AI Gym
#  monte carlo https://qiita.com/MENDY/items/4a896b9800775ad987a7
#  policy gradient/REINFORCE: e greedy mutateいれたらlocal min保証されない？
#                    どこに収束する？optNEになる？
#                 例 http://blog.brainpad.co.jp/entry/2017/09/08/140000
#                    https://www.slideshare.net/naotoyoshida9887/pydatatokyo
#                    https://qiita.com/icoxfog417/items/242439ecd1a477ece312
#       ライブラリ   https://www.slideshare.net/pfi/nlp2018-introduction-of-deep-reinforcement-learning
#       policy grad=自然勾配 https://www.slideshare.net/mobile/nishio/3-71708970
#   Qlearnと比べた利点がよくわからん(sutton p322,323)
#  Q gradient: guestrin2002 (1)
#  fictitious play
#  regret min
#  coordination graph:非凸タスクの凸分解可能？
#  Dist Q：rideshare pop gameには効かないのでは？
#  CMS-EAはoptNE選択できるのか？
#  OAL:どうやってoptimal収束証明した？
#  NashQ:内部でNE計算してるならもういらんのでは？Q(a)学習するのにp(a)学習しないってどういうこと？
#  NNつかったEE
#  DQN:収束しない？
#  POMDP上の学習
#  ハンガリアンアルゴ(センター)

class Solver:
  def __init__(self):
    1;
  
  # (1+1)EA[Doerr2011]
  def central11EA(self,prb,T=1000):
    n = prb.N;
    sol = solution.Solution(n,1,T);
    pf = prb.X;
    ui = prb.ui;
  
    pmute = 0.01;
    before = prb.socialWelfare(pf);
    for t in range(0,T):
      bpf = copy.deepcopy(pf);
      #prb.show(pf);
      sol.add(t,pf,J=before,u=prb.us(pf));
    
      # 全bitをランダムmutate
      for i in range(0,n):
        if np.random.rand() < pmute:
          bpf[i] = 1-pf[i];
	  #print "yes",bpf,prb.socialWelfare(bpf),before;
      after = prb.socialWelfare(bpf);
      if before < after:
        # ポインタ入れ替え
        tmp = pf;
        pf = bpf;
        bpf = tmp;
        before = after;
    return sol;

  def dist11EA(self,prb,T=1000):
    n = prb.N;
    sol = solution.Solution(n,1,T);
    pf = prb.X;
    ui = prb.ui;
  
    pmute = 0.01;
    for t in range(0,T):
      bpf = copy.deepcopy(pf);
      prb.show(pf);
      sol.add(t,pf,J=prb.socialWelfare(pf),u=prb.us(pf));
    
      # 各プレイヤごとにuiをみて変異
      for i in range(0,n):
        before = prb.ui(pf,i);
        if np.random.rand() < pmute:
          bpf[i] = 1-pf[i];
        after = prb.ui(bpf,i);
        if before < after:
          tmp = pf;
          pf = bpf;
          bpf = tmp;
  
    return sol;

  # まだ実装してない
  # 複雑。Q(xNit,pit)はgiven。NNパラメタpitを学習する。xNitは近傍全員の位置、センサ、action、記憶を含む
  def matingOnlyEE(self,prb,T=1000):
    n = prb.N;
    sol = solution.Solution(n,1,T);
    pf = prb.X;
    ui = prb.ui;

    pmute = 0.01;
    for t in range(0,T):
      bpf = copy.deepcopy(pf);
      prb.show(pf);
      sol.add(t,pf,J=prb.socialWelfare(pf),u=prb.us(pf));  
    
      # 各プレイヤごとにuiをみて変異
      for i in range(0,n):
        before = prb.ui(pf,i);
        if np.random.rand() < pmute:
          bpf[i] = 1-pf[i];
        after = prb.ui(bpf,i);
        if before < after:
          tmp = pf;
          pf = bpf;
          bpf = tmp;
  
    return sol;

  def mixBRLP(self,prb,i,freq):
    den = np.sum(freq.values());
    M = len(prb.A[i]);
    c = np.zeros(M);
    def getpf(opf,j,aj):
      ret = list(opf);
      ret[j] = aj;
      return ret;

    for ai in prb.A[i]:
      c[ai] = np.sum([np.float(cnt)/den*prb.ui(getpf(opf,i,ai),i) for opf,cnt in freq.items()]);
    B = np.ones(M);
    d = np.ones(1);
    A = np.vstack((-np.eye(M),np.eye(M)));
    b = np.hstack((np.zeros(M),np.ones(M)));
    return c,np.mat(A),np.mat(b),np.mat(B),np.mat(d);

  # fictitious play
  def fictitious(self,prb,T=10,dt=0.01):
    sol = solution.Solution(prb.N,prb.M,np.int(T/dt));
    # joint pf-iを学習
    mem = {i:Counter() for i,a in prb.agents()};
    r = np.zeros(prb.N);
    for ti,t in enumerate(np.arange(0,T,dt)):
      pf = tuple(prb.run()); # サンプリング
      for i,a in prb.agents():
        r[i] = prb.ui(pf,i); # reward
      for i,a in prb.agents():
        opf = list(pf); opf[i] = 0; opf = tuple(opf);
        mem[i][opf] += 1; # 学習
	c,A,b,B,d = self.mixBRLP(prb,i,mem[i]); # best response LP生成
        plp,xi,s = gb.pulpLP(-c,A,b,B=B,d=d); # maximizeなら-c
        res = gb.solvePulp(plp,msg=0);
        #gb.printSol(res,x,s);
        xi = [v.value() for k,v in xi.items()];
	prb.X[:,i] = xi;
      sol.add(ti,prb.X,u=r);

    return sol;

  # replicator dynamics
  def replicator(self,prb,T=100,dt=0.01):
    sol = solution.Solution(prb.N,prb.M,np.int(T/dt));

    y = np.zeros((prb.N,prb.M)); # 累積reward
    r = np.zeros(prb.N);
    for ti,t in enumerate(np.arange(0,T,dt)):
      pf = prb.run(); # サンプリング
      for i,a in prb.agents():
        r[i] = prb.ui(pf,i); # reward
	ai = pf[i];
	y[i,ai] += r[i]*dt;
      #print y,util.norm(y); exit();
      for i,a in prb.agents():
	Eyi = np.sum([prb.X[bi,i]*y[i,bi] for bi in prb.A[i]]);
	dxa = np.array([prb.X[bi,i]*(y[i,bi]-Eyi) for bi in prb.A[i]]);
	dxa -= np.sum(dxa)/len(dxa);  # 数値誤差を補正
	prb.X[:,i] += dxa*dt; # 更新
	
	#prb.X[np.where(prb.X[:,i]<0),i] = 0.;
	#prb.X[:,i] /= np.sum(prb.X[:,i]); # 数値誤差を補正

        if i==0:
	  print ti,i,y[i,:]-Eyi,prb.X[:,i],r[i];
      sol.add(ti,prb.X,u=r);

    return sol;

  # Douchan2019
  def continuousQ(self,prb,T=100,dt=1,expl=0,anime=False,task=0):
    n = prb.N;
    sol = solution.Solution(n,1,T);
    pf = prb.X;
    Q = [defaultdict(lambda:0) for i in range(prb.N)];
    gamma = 0.8; # 割引率
    tau = 10;
    #expl = 0;
    #expl = 0.02; # 探索率

    if not irid:
      fig = plt.figure()
    ims = [];
    lastcolt = 0;

    for ti,t in enumerate(np.arange(0,T,dt)):
      #iscol = prb.gamedebug(dt,pf);
      iscol = prb.game(dt,pf); # dtだけゲームを進める
      #if iscol:
      #  print "crush";
      #print "meanpf",np.mean(pf);
      unamecard=[len(h.keys()) for h in prb.namecard];
      print "t",t; sys.stdout.flush();
      print "pf",[(k,v) for k,v in sorted(Counter(pf).items(), key=lambda x:x[0])],"meannc",np.mean(unamecard);

      if 0:
        print "pf",pf;
        for i in range(prb.N):
          if prb.active[i] > 0:
	    print "i,pf,avert,na",i,pf[i],prb.averting[i],prb.na[i];
            self.printQi(Q,i,prb.na[i]);

      #self.printQ(Q,prb.N);
      #self.printQi(Q,0,0);
      r = [prb.ui(i) for i in range(prb.N)]; # reward
      # 衝突時に利得計算
      if iscol:
        l = t-lastcolt;
	alpha = 1.-np.exp(-l/tau);
        for i in range(prb.N):
          u = prb.ui(i); # reward
	  prena = prb.prena[i];
	  na = prb.na[i];
	  #na = 1; # 状態なし
	  # 学習は前回のprenaで
	  maxQ = np.max([Q[i][(prena,a)] for a in prb.A]);
	  Q[i][(prena,pf[i])] = (1.-alpha)*Q[i][(prena,pf[i])]+alpha*(u+gamma*maxQ); # Qlearning
	  # 選択は今回のnaで
	  pf[i] = np.argmax([Q[i][(na,a)] for a in prb.A]); # Q値によるBest response
	  if np.random.rand() < expl:
	    otherA = list(set(prb.A)-set([pf[i]]));
	    pf[i] = gb.sample(otherA,1)[0]; # mutation
	  #pf[i] = 2; # debug。決め打ち

	prb.reset(); # 利得変数リセット
	lastcolt = t;
      if task==0:
        sol.add(ti,pf,J=sum(prb.mileage),u=prb.mileage,r=r); # mileageタスク
      elif task==1:
        sol.add(ti,pf,J=sum(unamecard),u=prb.mileage,r=r); # namecardタスク
      if not irid:
        im = prb.plot(plt,pf);
        ims.append([im]);
	if prb.gif:
	  fname = "output/anime/%05d.png" % ti;
	  plt.savefig(fname);


    '''
    print "Qtable";
    for i in range(prb.N):
      na = np.max([na for (na,a),v in Q[i].items()]);
      besta = [np.argmax([Q[i][(ana,a)] for a in prb.A]) for ana in range(na+1)];
      print i,besta;
      for k,v in Q[i].items():
        print " ",k,v;
    '''

    if anime and (not irid) and (not prb.gif):
      # http://cflat-inc.hatenablog.com/entry/2014/03/17/214719
      print "making anime..."; sys.stdout.flush();
      ani = animation.ArtistAnimation(fig, ims, interval=30);
      afile = "output/swarm.mp4";
      ani.save(afile);

    if prb.gif and (not irid):
      print "converting gif anime..."; sys.stdout.flush();
      print os.popen('convert -delay 5 output/anime/*.png output/swarm.gif').read();

    return sol;

  # Douchan2019 動的ゲーのBRU
  # forI:予見iter回数
  # バグ。betterRUじゃなくてmaxUpdateになってる
  def BRURLcol(self,prb,T=100,dt=1,forI=10,expl=0,anime=False,task=0):
    n = prb.N;
    sol = solution.Solution(n,1,T);
    pf = prb.X;

    if not irid:
      fig = plt.figure()
    ims = [];
    lastcolt = 0;

    updated = [];
    # 各tに一人だけBRU
    for ti,t in enumerate(np.arange(0,T,dt)):
      #iscol = prb.gamedebug(dt,pf);
      iscol = prb.game(dt,pf); # dtだけゲームを進める
      #if iscol:
      #  print "crush";
      #print "meanpf",np.mean(pf);
      print "t",t; sys.stdout.flush();
      print "pf cnt",[(k,v) for k,v in sorted(Counter(pf).items(), key=lambda x:x[0])];
      print "pf",pf;

      r = [prb.ui(i) for i in range(prb.N)]; # reward
      # 衝突時に利得計算
      if iscol:
        l = t-lastcolt;
	
        for i in range(prb.N):
	  if i not in updated:
	    break;
	updated.append(i);
	if len(updated) >= prb.N:
	  updated = [];

        '''
	maxu = -np.inf; maxa = None;
        for a in prb.A: # best responseをシミュレート
	  aprb = gb.deepcp(prb); # ゲームコピー
	  aprb.X[i] = a;
	  aprb.reset(); # 利得変数リセット
          for ta in np.arange(0,dt*forI,dt): # 3iterゲームを試す
            iscola = aprb.game(dt,aprb.X);
            u = aprb.ui(i); # reward
	    if maxu < u:
	      maxu = u;
	      maxa = a;
	  if 0:
            print i,a,u,maxu,(prb.t,aprb.t),aprb.S[i,:],aprb.newS[i,:],aprb.active[i];
            x = aprb.newS;
            D = distance.cdist(x,x, metric='euclidean');
	    col = np.where(D < aprb.r)[0];
	    print col;
	  #if u > maxu:
	  #  maxu = u;
	  #  maxa = a;
	#print " maxa",maxa;
	
	pf[i] = maxa;
        '''
	prb.reset(); # 利得変数リセット
	lastcolt = t;
      if task==1:
        sol.add(ti,pf,J=sum(prb.mileage),u=prb.mileage,r=r); # mileageタスク
      elif task==1:
        sol.add(ti,pf,J=sum(unamecard),u=prb.mileage,r=r); # namecardタスク
      elif task==2:
        sol.add(ti,pf,J=sum(prb.aggregate),u=prb.aggregate,r=r); # aggregateタスク
      if not irid:
        im = prb.plot(plt);
        ims.append([im]);
	if prb.anime==2:
	  fname = "output/anime/%05d.png" % ti;
	  plt.savefig(fname);

    if (prb.anime==1) and (not irid):
      # http://cflat-inc.hatenablog.com/entry/2014/03/17/214719
      print "making anime..."; sys.stdout.flush();
      ani = animation.ArtistAnimation(fig, ims, interval=30);
      afile = "output/swarm.mp4";
      print afile;
      ani.save(afile);

    if prb.anime==2:
      print "converting gif anime..."; sys.stdout.flush();
      print os.popen("convert -delay %d output/anime/*.png output/swarm.gif" % prb.delay).read();

    return sol;

  # gameのBRUを実行するだけ
  def runBRU(self,prb,T,dir="output/anime"):
    n = prb.N;
    sol = solution.Solution(n,1,T);

    if not irid:
      fig = plt.figure()
    ims = [];

    pf0 = prb.game.x; # 位置分布がpf。いらんのでは？bru内でgetできる
    for t,i,pf in prb.game.bru(T):
      #print "t",t,"i",i,"ai",pf[i],"ng",prb.ng(i,pf),"sw",prb.game.sw(pf),"ui",prb.game.u(i,pf);
      print "t",t,"sw",prb.game.sw(pf);
      #print "f",prb.game.f(pf);
      sys.stdout.flush();
      prb.prntdbg(pf);
      if 0 and (prb.anime>0) and (not irid):
        im = prb.plot(plt,t,bg=i);
        ims.append([im]);
        fname = dir+"/%05d.png" % t;
        #im.set_facecolor('pink');
        #print im.get_facecolor()[0]; exit();
        plt.savefig(fname,facecolor=im.get_facecolor()[0]);

    if prb.game.gso is not None:
      prb.game.dumppos("input/sopos.txt");
    if (prb.anime>1) and (not irid):
      #print "converting gif anime..."; sys.stdout.flush();
      command = "convert -delay %d %s/*.png output/swarm.gif" % (prb.delay,dir);
      print command; sys.stdout.flush();
      print os.popen(command).read();


  # gameのRLを実行するだけ  
  def runRL(self,prb,T=1000,forget=False):
    n = prb.N;
    sol = solution.Solution(n,1,T);

    if not irid:
      fig = plt.figure()
    ims = [];

    for t,i,pf in prb.game.rl(T,forget=forget):
      print "t",t,"i",i,"ai",pf[i],"ng",prb.ng(i,pf),"ui",prb.game.u(i,pf),"f",prb.game.f(pf); sys.stdout.flush();
      prb.prntdbg(pf);
      if (prb.anime==2) and (not irid) and (t%20==0):
        im = prb.plot(plt,t);
        ims.append([im]);
        fname = "output/anime/%05d.png" % t;
        plt.savefig(fname);

    if prb.game.gso is not None:
      prb.game.dumppos("input/sopos.txt");
    if 0 and (prb.anime==2) and (not irid) and (prb.dbgi<0):
      #print "converting gif anime..."; sys.stdout.flush();
      command = "convert -delay %d output/anime/*.png output/swarm.gif" % prb.delay;
      print command; sys.stdout.flush();
      print os.popen(command).read();



  # local State-time wise WLUに基づくstatewise BRU
  def localStWLUBRU(self,prb,T=2):
    wlu = prb.usewlu;
    n = prb.N;
    sol = solution.Solution(n,1,T);

    if not irid:
      fig = plt.figure()
    ims = [];

    print "s0",prb.game.src;
    print "npolicy",prb.game.npolicy();
    print "mstate",prb.game.nstate();
    ppf = prb.game.randppf(); # 初期ポリシー

    '''
    # center同期型。相手のpolicyを自分のMDPに翻訳しなくてはならない
    # 自分のpolはいいけど、相手のpolは観測縮約時に競合する！
    # POMDPのように、真のsを推定するアプローチが自然。
    for i in prb.game.players():
      m = prb.localgame[i];
      o0 = m.src; # 初期観測
      print "i,o0",o0,m.g.node[o0]['state'].code();
      # ppfのlocal翻訳
      prb.transPolicy(ppf,i);
    '''


  # State-time wise WLUに基づくstatewise BRU
  def StWLUBRU(self,prb,T=2):
    wlu = prb.usewlu;
    n = prb.N;
    sol = solution.Solution(n,1,T);

    if not irid:
      fig = plt.figure()
    ims = [];

    m = prb.game;
    print "npolicy",m.npolicy();
    print "mstate",m.nstate();
    ppf = m.randppf(); # 初期ポリシー
    s0 = m.src; # 初期state
    print "s0",s0,m.g.node[s0]['state'].code();

    # complexity
    '''
    for T in range(1,10):
      start = time.time();
      m.ev(ppf,s0,T,0);
      print T,"elps",time.time()-start;
      sys.stdout.flush();
    '''

    # 非DP
    ppfs = gb.indexHash();
    cnt = 0;
    wupdated = True;
    cycle = False;
    bestu = -np.inf;
    bestv = -np.inf;
    while wupdated:
      if cycle:
        break;
      wupdated = False;
      for i in m.players():
	for s in m.states(): # 各(s,t)ごとにBRを探る
            #for tmp,s in m.reachable(s0,t,ppf):
	    #print t,s,prb.game.g.node[s]['state'].code(); sys.stdout.flush();
	    bestai = ppf[i][s];
	    if wlu:
	      # t=0にs0から始めてTまでppfをplayした場合に、sにてa=ppf[s]を実行したときの貢献度
	      tmpu = m.StWLU(ppf,s0,s,T,i); # WLU
	    else:
	      tmpu = m.ev(ppf,s0,T,0,i); # SW
	    #if tmpu < bestu:
	    #  print "i,tmpu < bestu",i,tmpu,bestu;
	    #  exit();
	    bestu = tmpu;
	    bppf = gb.deepcp(ppf);
	    for ai in m.A: # sに関わらずAi一定と仮定
	      bppf[i][s] = ai;
	      if wlu:
                u = m.StWLU(bppf,s0,s,T,i); # WLU
	      else:
	        u = m.ev(bppf,s0,T,0,i); # SW
	      if u - bestu > m.isepsilon:
	        print "update,i,s,dev",i,s,(ai,u),">",(bestai,bestu),m.ev(ppf,s0,T,0);
		sys.stdout.flush();
	        bestu = u;
	        bestai = ai;
		#print "before";
		#m.StWLU(ppf,s0,s,T,i,prnt=1);
		#print "after";
		#m.StWLU(bppf,s0,s,T,i,prnt=1);
	        ppf[i][s] = bestai;
		tplppf = m.tupleReachable(ppf,s0,T);
		if tplppf in ppfs:
		  print "cycle!";
		  cycle = True;
		ppfs.append(tplppf);
		print ppfs[tplppf];
		m.showReachable(ppf,s0,T); sys.stdout.flush();
		if m.ev(ppf,s0,T,0) < bestv:
		  print "v down!";
		  exit();
		bestv = m.ev(ppf,s0,T,0);
		wupdated = True;
		if cycle:
		  break;
	    if cycle:
	      break;
      cnt += 1;

    print "s0,T",s0,T;
    if 0:
      m.stateReachable(ppf,s0,2); exit();
      i = 0; s = 8; t1 = 1; s1=8;
      t1 = 0; s1 = 1;
      T = 4;
      print "ppf ev,wlu", m.ev(ppf,s0,T,i,prnt=0),m.StWLU(ppf,s0,s,T,i,prnt=0),m.StWLU(ppf,s1,s,T-t1,i,prnt=0);
      tmp =m.ev(ppf,s1,T,t1,i,prnt=0);
      print tmp;
      ppf[i][s] = 0;
      print "bppf ev,wlu", m.ev(ppf,s0,T,i,prnt=0),m.StWLU(ppf,s0,s,T,i,prnt=0),m.StWLU(ppf,s1,s,T-t1,i,prnt=0);
      tmp = m.ev(ppf,s1,T,t1,i,prnt=0);
      print tmp;
      exit();


    '''
    # DP
    cnt = 0;
    wupdated = True;
    while wupdated:
      if cnt > 15:
        break;
      wupdated = False;
      for i in m.players():
        for t in reversed(range(0,T)):
          for tmp,s1 in m.reachable(s0,t,ppf):
	    for s in m.states():
            #for tmp,s in m.reachable(s0,t,ppf): # (s,t)でのBRを探る
	      #print t,s,prb.game.g.node[s]['state'].code(); sys.stdout.flush();
	      bestai = ppf[i][s];
	      if wlu:
	        bestu = m.StWLU(ppf,s1,s,T-t,i); # WLU
	      else:
	        bestu = m.ev(ppf,s0,T,0,i); # SW
	      bppf = gb.deepcp(ppf);
	      for ai in m.A: # sに関わらずAi一定と仮定
	        bppf[i][s] = ai;
	        if wlu:
                  u = m.StWLU(bppf,s1,s,T-t,i); # WLU
	        else:
	          u = m.ev(bppf,s0,T,0,i); # SW
	        if u - bestu > m.isepsilon:
	          print "update,i,(t,s),dev",i,(t,s),(ai,u),">",(bestai,bestu),m.ev(ppf,s0,T,0);
		  sys.stdout.flush();
	          bestu = u;
	          bestai = ai;
	          ppf[i][s] = bestai;
		  wupdated = True;
      cnt += 1;
    '''

    m.showReachable(ppf,s0,T);
    m.stateReachable(ppf,s0,T,wlu=1);
    #ppf[0][0] = 4;
    #m.showReachable(ppf,s0,T);
    print "cnt",cnt;
    sys.stdout.flush();

    # サブゲーム完全性チェック。t=Tから逆順方向にBRUしてdevするか？
    cnt = 0;
    wupdated = False;
    while cnt<15:
      for i in m.players():
        for t1 in reversed(range(0,T)): # サブゲーム始点
          for tmp,s1 in m.reachable(s0,t1,ppf): # サブゲーム始点。reachableのみ
	      for s in m.states(): # dev state
	        #print t,s,prb.game.g.node[s]['state'].code(); sys.stdout.flush();
	        bestai = ppf[i][s];
	        if wlu:
	          # t=0にs1から初めてT-t1までppfをplayした場合に、sにてa=ppf[s]を実行したときの利得
	          bestu = m.StWLU(ppf,s1,s,T-t1,i); # WLU
	        else:
	          bestu = m.ev(ppf,s1,T,t1,i); # サブゲーSW。t1でs1から始めてTまでppfをplayしたときの累積期待利得
	        bppf = gb.deepcp(ppf);
	        for ai in m.A: # sに関わらずAi一定と仮定
	          bppf[i][s] = ai;
		  if wlu:
                    u = m.StWLU(bppf,s1,s,T-t1,i); # WLU
		  else:
	            u = m.ev(bppf,s1,T,t1,i); # SW
	          if u - bestu > m.isepsilon:
	            print "non markov perfect!,update,i",i,"(t1,s1)",(t1,s1),"s",s,"dev",(ai,u),">",(bestai,bestu);
		    #print "s,x",s,m.g.node[s]['state'].code();
		    #print "bppf,t1,t,T",t1,t,T;
		    #m.StWLU(bppf,s1,s,T-t1,i,prnt=1);
		    #print "ppf";
		    #m.StWLU(ppf,s1,s,T-t1,i,prnt=1);
		    sys.stdout.flush();
		    wupdated = True;
		    break;
	        if wupdated:
	          break;
      if wupdated:
        break;
      cnt += 1;
    print "markov perfect",not wupdated;

    if (prb.anime==2) and (not irid):
      for t,s in enumerate(prb.game.run(ppf,T)):
        #x = prb.game.g.node[s]['state'].code();
        im = prb.plot(plt,t,s);
        ims.append([im]);
        fname = "output/anime/%05d.png" % t;
        plt.savefig(fname);

      print "converting gif anime..."; sys.stdout.flush();
      print os.popen("convert -delay %d output/anime/*.png output/swarm.gif" % prb.delay).read();


  # 動的ゲーのBRU
  # forI:予見iter回数
  def BRUext(self,prb,T=100,dt=1,forI=10):
    n = prb.N;
    sol = solution.Solution(n,1,T);
    pf = prb.X;

    if not irid:
      fig = plt.figure()
    ims = [];

    updated = []; # 各tに一人だけBRU
    prestat = None;
    for ti,t in enumerate(np.arange(0,T,dt)):
      print "#### t",t+dt; sys.stdout.flush();
      prb.game(dt,pf,prnt=1); # dtだけゲームを進める
      print "game end";
      prb.show();
      stat = prb.state();
      if 0 and prb.overlap():
        print "#",stat,False;
	#exit();
      sol.conv=(stat == prestat);
      sol.addsink(stat,prb.space[0],ti);
      print "pf cnt",[(k,v) for k,v in sorted(Counter(pf).items(), key=lambda x:x[0])];
      if not irid:
        #print "plotdir",prb.S[:,2];
        im = prb.plot(plt);
        ims.append([im]);
	if prb.anime==2:
	  fname = "output/anime/%05d.png" % ti;
	  plt.savefig(fname);

      #print "meanpf",np.mean(pf);

      r = [prb.ui(i) for i in range(prb.N)]; # reward
      print "pf,r",pf,r;

      # すでにupdate済みのiは飛ばす
      for i in util.sample(range(prb.N),prb.N):
	if i not in updated:
	  break;
      updated.append(i);
      if len(updated) >= prb.N:
	updated = [];

      betu = -np.inf; beta = pf[i]; betstat = None;
      for a in prb.A: # best responseをシミュレート
	  aprb = gb.deepcp(prb); # ゲームコピー
	  aprb.X[i] = a;
          for ta in np.arange(0,dt*forI,dt): # forI iterゲームを試す
            aprb.game(dt,aprb.X);
            u = aprb.ui(i); # reward
	    #if betu < u:
	    #if u > r[i]: # better RU
	    if u > betu:
	      betu = u;
	      beta = a;
	      betstat = aprb.state();
	  if 0:
            print i,a,u,betu,(prb.t,aprb.t),aprb.S[i,:],aprb.newS[i,:],aprb.active[i];
            x = aprb.newS;
            D = distance.cdist(x,x, metric='euclidean');
	    col = np.where(D < aprb.r)[0];
	    print col;

      print "update",i,":beta,betu",beta,betu;
      print " betstat",betstat;
      if betu > r[i]:
        pf[i] = beta;

      sol.add(ti,pf,J=sum(r),r=r); # aggregateタスク
      prestat = stat;

    print sol.conv;
    print len(sol.sinks),"sinks";

    if (prb.anime==1) and (not irid):
      # http://cflat-inc.hatenablog.com/entry/2014/03/17/214719
      print "making anime..."; sys.stdout.flush();
      ani = animation.ArtistAnimation(fig, ims, interval=30);
      afile = "output/swarm.mp4";
      print afile;
      ani.save(afile);

    if (prb.anime==2) and (not irid):
      print "converting gif anime..."; sys.stdout.flush();
      print os.popen("convert -delay %d output/anime/*.png output/swarm.gif" % prb.delay).read();

    return sol;


  def printQi(self,Q,i,na):
    for k,v in Q[i].items():
      if k[0] == na:
        print i,k,v;        

  def printQ(self,Q,N):
    for i in range(N):
      for k,v in Q[i].items():
        print i,k,v;    

  # 離散BRU
  # 実行前Ai探索可能
  # Aはaction set
  def bestResponseUpdates(self,prb,T=1000,prntcg=False):
    n = prb.N;
    sol = solution.Solution(n,1,T);
    pf = prb.X;
    ui = prb.ui;

    for t in range(0,T):
      # ランダム順
      sw = prb.socialWelfare(pf);
      sol.add(t,pf,J=sw,u=prb.us(pf));
      if prntcg:
        print t,[prb.congestion(pf,i) for i,a in prb.agents()],sw;
      else:
        print t,pf,sw;
      for i in util.sample(range(0,n),n):
        pf = self.bestResponseUpdate(prb,i,pf);
  
    return sol;

  def bestResponseUpdate(self,prb,i,pf):
    best = prb.A[0];
    bestui = prb.ui(pf,i);
    updated = False;
    pfb = copy.deepcopy(pf);
    for ai in prb.actions(i):
      pfb[i] = ai;
      #print i,pf[i],ai,prb.ui(pf,i),prb.ui(pfb,i);
      if prb.ui(pfb,i) > bestui:
        updated = True;
        #print i,pfb,prb.ui(pfb,i),">",bestui;
        best = ai;
        bestui = prb.ui(pfb,i);
    if updated:
      pf[i] = best;
    return pf;

  # 長さMのベクトルの合意
  # H:隣接行列
  # X:N*Mのベクトル
  def consensus2D(self,prb,dt=0.02,T=1000):
    X = prb.X;
    print "init x",X;
    n,m=X.shape;
    #A = nx.adjacency_matrix(prb.G).todense();
    L = nx.laplacian_matrix(prb.G).todense();
    w,v = np.linalg.eig(L);
    l2 = np.min(w[np.where(w>0)]); # worst case consensus speed:最小正固有値
    #print l2; exit();
    sol = solution.Solution(prb.N,m,T);
    P=np.eye(n)-dt*L; # ペロン行列
    for t in range(0,T):
      sol.add(t,X);
      #dx = -L*x;
      #x = x+dx*dt; # =(E-dt*L)x
      X = P*X;
    print "consensus x",X;
    return sol;

  
  # 劣勾配合意制御。T2@MAS6
  # 長さMのベクトルの劣勾配合意
  # H:隣接行列
  # X:N*Mのベクトル
  # J:Mベクトル入力の目的関数
  def subGradientConsensus2D(self,prb,dt=0.02,ds=1.,T=1000):
    X = prb.X;
    L = nx.laplacian_matrix(prb.G).todense();
    n,m=X.shape;
    sol = solution.Solution(prb.N,m,T);
    print "init x",X;
    P=np.eye(n)-dt*L; # ペロン行列
    for t in range(0,T):
      si = ds/(t+1);
      pf = np.diag(X);
      sol.add(t,X,J=prb.socialWelfare(pf),u=prb.us(pf));

      #dx = -L*x;
      #x = x+dx*dt; # =(E-dt*L)x
      D = X*0; # 勾配項
      # まず全agent一斉に現在のXから勾配を計算
      # ui
      for j in range(0,n):
        D[j] = prb.dj(np.mat(X[j,:]).A.ravel(),j);
      X = P*X-si*D; # 全員一度に更新
    print "consensus x",X;
    print "avg X",np.mean(X.T,1).T,np.mean(X.T,1).shape;
    print "sum X",np.sum(X,1).T,np.sum(X,1).shape;
    return sol;

  # 主双対勾配法1D
  # min L=j+λg
  def pdga1d(self,prb,dt=0.02,T=1000):
    X = prb.X;
    print "init x",X;
    n,m=X.shape;
    sol = solution.Solution(prb.N,m,T,Mg=3);
    l = 1;
  
    for t in range(0,T):
      lt = l;
  
      # まず現在のXから双対をupdateする
      # 各agentごとのlocal双対変数をupdate。勾配を足す
      j = prb.j(X[0,0]);
      print j;
      dLi = prb.gi(X[0,0]); # dL/dli
      g = dLi;
      l += dLi*dt;
  
      sol.add(t,X[0,0],J=j,g=g,l=lt);

      # 勾配項
      dX = prb.dj(X[0,0]); # [dj/dx]
      dX += l*prb.dgi(X[0,0]); # [dgi/dx] = dimg*m
      # 主Xをupdate。勾配を引く
      X -= dX*dt;
      
    print "x",X;
    return sol;

  # 主双対勾配法
  # min sum(ui)
  # s.t. gi(xi)<=0
  #      hi(xi)=0
  #      xi=xj if (i,j) in E <=> Lx=0
  # L=sum(ui)+sum(λi*gi)+sum(μi*hi)+ε*0.5*xLx
  def pdga(self,prb,dt=0.02,T=1000):
    X = prb.X;
    L = nx.laplacian_matrix(prb.G).todense(); # n*n
    print "init x",X;
    n,m=X.shape;
    sol = solution.Solution(prb.N,m,T,Mg=3);
    P=np.eye(n)-dt*L; # ペロン行列
    #for i in range(0,n):
    #  print prb.gi(X[i,:],i);
    dimg = len(prb.gi(X[0,:],0));
    dimh = len(prb.hi(X[0,:],0));
    l = np.ones((n,dimg))/n;
    u = np.ones((n,dimh))/n;
    e = np.ones((m,m))/m/m; # global双対変数
  
    for t in range(0,T):
      lt = [np.mean(sum(l)),np.mean(sum(u)),sum(sum(e))];
      #print "l,u,e",np.mean(sum(l)),np.mean(sum(u)),sum(sum(e));
  
      j = 0.; g = 0.; h = 0.; q = 0.;
      # まず現在のXから双対をupdateする
      # 各agentごとのlocal双対変数をupdate。勾配を足す
      for i,a in prb.agents():
        j += prb.ui(X[i,:],i);
        #print "t,i,x,sumx,ui",t,i,X[i,:],sum(X[i,:]),prb.ui(X[i,:],i);
        dLi = prb.gi(X[i,:],i); # dL/dli
        g += sum(dLi);
        #print " gi",dLi;
        for k,agi in enumerate(dLi):
          if (agi < 0.) and (l[i,k] < 1e-10):
  	    dLi[k] = 0.;
        l[i] += dLi*dt;
        dUi = prb.hi(X[i,:],i); # dL/dui
        u[i] += dUi*dt;
        #print " hi",dUi;
        h += sum(dUi);
      # consensus双対変数をupdateする。勾配を足す
      dE = X.T*L*X;
      q += sum(sum(dE.A));
      e += dE*dt;
  
      gt =[g,h,q];
      pf = np.diag(X);
      sol.add(t,X,J=j,u=prb.us(pf),g=gt,l=lt);

      dX = X*0; # 勾配項
      # まず全agent一斉に現在のXから勾配dX=dL/dxを計算
      for i,a in prb.agents():
        dX[i] = prb.dui(np.mat(X[i,:]).A.ravel(),i); # [dui/dxi] = 1*m
        dX[i] += (l[i]*np.mat(prb.dgi(X[i,:],i))).A.ravel(); # [dgi/dxi] = dimg*m
        dX[i] += (u[i]*np.mat(prb.dhi(X[i,:],i))).A.ravel(); # [dhi/dxi] = dimh*m
        dX[i] += (sum(sum(e))*L[i,:]*X).A.ravel(); # [dQ/dxi] = [Li*x] = 1*m
      # 主Xをupdate。勾配を引く
      X -= dX*dt;
      
    print "consensus x",X;
    print "avg X",np.mean(X.T,1).T,np.mean(X.T,1).shape;
    print "sum X",np.sum(X,1).T,np.sum(X,1).shape;
    print "X>0.5";
    print [list(X[i,np.where(np.abs(X[i,:])>0.5)[0]]) for i in range(0,n)];
    return sol;

class Duncan2008(Solver):
  def __init__(self,prb):
    Solver.__init__(self);
    self.q = np.ones((prb.N,len(prb.A))); # 累積reward
    self.prb = prb;

  def mu(self,t):
    return np.array([t/sum(self.q[i,:]) for i,a in self.prb.agents()]);

  def eu(self,t):
    return 1./self.mu(t);
    
  # reinforcement learning@duncan2008
  #  erev1998predicting
  # forget onするとpNEに収束
  def erev1998(self,T=1000,forget=False):
    prb = self.prb;
    n = prb.N;
    sol = solution.Solution(n,1,T);
    x = prb.X; # mixed pf
    ui = prb.ui;
    e = 0.2; # mutation rate。全然収束しない。
    e = 0.01;
    phi = 0.1; # 忘却率。収束しない
    phi = 0.01;

    for t in range(0,T):
      pf = prb.run(x); # サンプリング
      # ランダム順
      sw = prb.socialWelfare(pf);
      us = np.array(prb.us(pf));
      sol.add(t,x,J=sw,u=us);
      print t,x,sum(x),sw;
      #print t,self.eu(t); # 忘却の分だけ平均rewardは減っていく
      for i,a in prb.agents():
        if forget:
	  ai = pf[i];
	  bi = 1-ai;
          self.q[i,ai] = (1.-phi)*self.q[i,ai] + (1.-e)*us[i]; # (1')
          self.q[i,bi] = (1.-phi)*self.q[i,bi] + us[i]*e/len(prb.A); # (1')
	else:
          self.q[i,pf[i]] += us[i]; # reward累積
      for i,a in prb.agents():
        x[i] = self.q[i,1]/sum(self.q[i,:]); # mixed strategy更新
  
    return sol;
