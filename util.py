# -*- coding: utf-8 -*-

import numpy as np;
import copy;
import networkx as nx;
import matplotlib.pyplot as plt;
import pickle;
import sys;

# Double Bracket flow
# 直交行列Hを回転操作によって対称行列Nに近づける
# 結果Hと回転Pを返す
def dbf(H,N,prnt=False):
  H0 = copy.deepcopy(H);
  n,n=H.shape;
  P = np.mat(np.eye(n));

  if prnt:
    print "initial H: rank",np.linalg.matrix_rank(H);
    print H;
    print;
    print "N: rank",np.linalg.matrix_rank(N);
    print N;
    print;

  # double bracket flow
  dt = 0.002; loop = 3000; # step size, iteration
  #dt = 0.01; loop = 300;
  for i in range(0,loop):
    dH = db(H,db(H,N));
    #print H*N; print N*H; print db(H,N); print dH; exit();
    if np.abs(dH).sum() < 1e-100:
      print "no rotate: dH"; print np.abs(dH); exit();
    dP = H0*P*N-P*N*P.T*H0*P;
    H = H+dH*dt;
    P = P+dP*dt;

  H = np.round(H,1);

  if prnt:
    print "final H: rank",np.linalg.matrix_rank(H);
    print H;

  '''
  print "P";
  print P;
  print "tP*H0*P";
  print P.T*H0*P;
  '''

  return H,P;

# ノードi,jを入れ替える置換行列
# P.T*A*Pで置換
def permmat(n,i,j):
  P = np.eye(n);
  P[i,i]=P[j,j]=0;
  P[i,j]=P[j,i]=1;
  return P;
    
def db(A,B):
  return A*B-B*A;

# vからk個サンプリング
def sample(v,k):
  p = np.random.permutation(len(v));
  return [v[i] for i in p[0:k]];

def socialWelfare(ui,pf,n):
  ret = 0.0;
  for i in range(0,n):
    ret += ui(pf,i);
  return ret;

# (1+1)EA[Doerr2011]
def central11EA(pf,ui,loop=1000):
  n=len(pf);
  ts = range(0,loop);
  xs = np.zeros((n,1,loop));
  Js=np.zeros((n,loop));
  bpf = copy.deepcopy(pf);

  pmute = 0.01;
  before = socialWelfare(ui,pf,n);
  for t in range(0,loop):
    Js[:,t] = before;
    xs[:,0,t] = pf;
  
    # 全bitをランダムmutate
    for i in range(0,n):
      if np.random.rand() < pmute:
        bpf[i] = 1-pf[i];
    after = socialWelfare(ui,bpf,n);
    if before < after:
      tmp = pf;
      pf = bpf;
      bpf = tmp;
      before = after;

  return ts,xs,Js;

def dist11EA(pf,ui,loop=1000):
  n=len(pf);
  ts = range(0,loop);
  xs = np.zeros((n,1,loop));
  Js=np.zeros((n,loop));
  bpf = copy.deepcopy(pf);

  pmute = 0.01;
  for t in range(0,loop):
    Js[:,t] = socialWelfare(ui,pf,n);
    xs[:,0,t] = pf;
  
    # 各プレイヤごとにuiをみて変異
    for i in range(0,n):
      before = ui(pf,i);
      if np.random.rand() < pmute:
        bpf[i] = 1-pf[i];
      after = ui(bpf,i);
      if before < after:
        tmp = pf;
        pf = bpf;
        bpf = tmp;

  return ts,xs,Js;

def matingOnlyEE(pf,ui,loop=1000):
  n=len(pf);
  ts = range(0,loop);
  xs = np.zeros((n,1,loop));
  Js=np.zeros((n,loop));
  bpf = copy.deepcopy(pf);

  pmute = 0.01;
  for t in range(0,loop):
    Js[:,t] = socialWelfare(ui,pf,n);
    xs[:,0,t] = pf;
  
    # 各プレイヤごとにuiをみて変異
    for i in range(0,n):
      before = ui(pf,i);
      if np.random.rand() < pmute:
        bpf[i] = 1-pf[i];
      after = ui(bpf,i);
      if before < after:
        tmp = pf;
        pf = bpf;
        bpf = tmp;

  return ts,xs,Js;


# 実行前Ai探索可能
# Aはaction set
def bestResponseUpdates(A,pf,ui,loop=1000):
  n=len(pf);
  ts = range(0,loop);
  xs = np.zeros((n,1,loop));
  Js=np.zeros((n,loop));

  for t in range(0,loop):
    # ランダム順
    sw = socialWelfare(ui,pf,n);
    Js[:,t] = sw;
    xs[:,0,t] = pf;
    print t,pf,sw;
    for i in sample(range(0,n),n):
      pf = bestResponseUpdate(A,i,pf,ui);

  return ts,xs,Js;

def bestResponseUpdate(A,i,pf,ui):
  best = A[0]; bestui = -np.inf;
  for ai in A:
    pf[i] = ai;
    if ui(pf,i) > bestui:
      best = ai;
      bestui = ui(pf,i);
  pf[i] = best;
  return pf;

def dump(obj,ofile='output/dbg.dump'):
    with open(ofile,mode='wb') as f:
      pickle.dump(obj,f,-1);
    print ofile; sys.stdout.flush();

def load(ofile):
    with open(ofile,mode='rb') as f:
      obj = pickle.load(f);

    return obj;


class GA:
  def __init__(self, X,ui):
    self.X = X;
    n,m=X.shape;
    self.N = n; # エージェント数
    self.M = m;
    self.nelite = 1;
    self.agents = [];
    self.children = [];
    self.pmute = 0.01; # 突然変異確率

    for i in range(0,n):
      self.agents.append(Individual(ui,X[i,:])); # 現行世代。fitnessも初期化
      self.children.append(Individual(ui,X[i,:])); # 次世代。fitnessも初期化

    self.quicksort(0,self.N-1,1); # 大きい順

  # 小さいもの順
  def quicksort(self,lb,ub,rev = False):
    cf = 1;
    if rev:
      cf = -1;
    if lb < ub:
      k = (lb+ub)/2;
      pivot = cf*self.agents[k].fitness;
      i = lb;
      j = ub;

      while i <= j:
        while cf*self.agents[i].fitness < pivot:
	  i += 1;
	while cf*self.agents[j].fitness > pivot:
	  j -= 1;
	if i <= j:
	  tmp = self.agents[i];
	  self.agents[i] = self.agents[j];
	  self.agents[j] = tmp;
	  i += 1;
	  j -= 1;
      self.quicksort(lb,j,rev);
      self.quicksort(i,ub,rev);
  

  def evolve(self,loop=1000):
    n = self.N; m = self.M;
    ts = range(0,loop);
    xs = np.zeros((n,m,loop));
    Js=np.zeros((n,loop));

    for t in range(0,loop):
      self.update();
      self.show();
      
      Js[:,t] = self.sw();
      xs[:,:,t] = self.X;
    return ts,xs,Js;

  def update(self):
    # neliteだけ保存
    for i in range(0,self.nelite):
      self.children[i].x = self.agents[i].x;

    # 他のは交配で作る
    for i in range(self.nelite,self.N):
      p1 = self.select();
      p2 = self.select();
      self.mating1(i,p1,p2);

    # さらにbest以外を突然変異
    for i in range(1,self.N):
      self.mutate(i);

    # 入れ替え
    tmp = self.agents;
    self.agents = self.children;
    self.children = self.agents;

    # fitness計算
    for i in range(0,self.N):
      self.agents[i].eval();

    # 並べ替え
    self.quicksort(0,self.N-1,1); # 大きい順
      
  # 突然変異
  def mutate(self,c):
    for i in range(0,self.M):
      if np.random.rand() < self.pmute:
        self.children[c].x[i] = 1-self.children[c].x[i];

  # 一点交叉
  def mating1(self,c,p1,p2):
    xc = self.children[c].x;
    x1 = self.agents[p1].x;
    x2 = self.agents[p2].x;
    ic = np.random.randint(0,self.M);
    xc[0:ic] = x1[0:ic];
    xc[ic:] = x2[ic:];

  # 実装はいろいろある
  def select(self):
    return np.random.randint(0,self.N); # 完全ランダム

    denom = self.N*(self.N+1)/2;
    r = (np.random.randint(0,100) % denom) +1;
    print denom,r; exit();
    for i in reversed(range(0,self.N)):
      if r <= i:
        break;
      r -= i;

    return self.N-i;

  def sw(self):
    ret = 0.0;
    for i in range(0,self.N):
      ret += self.agents[i].fitness;
    return ret;

  def show(self):
    for i in range(0,self.N):
      self.agents[i].show();
    print "sw",self.sw();

class Individual:
  def __init__(self,u,x):
    self.x = x;
    self.u = u;
    self.eval();

  def eval(self):
    self.fitness = self.u(self.x,None);

  def show(self):
    print self.x,self.fitness;

# スカラーの合意
# H:隣接行列
# x:長さNのベクトル
def consensus1D(H,x):
  g = nx.from_numpy_matrix(H);
  L = nx.laplacian_matrix(g).todense();
  n,n=H.shape;
  print "H";
  print H;
  print "init x",x.T;
  dt = 0.02; loop = 1000;
  xs = np.mat(np.zeros((n,loop)));
  P=np.eye(n)-dt*L; # ペロン行列
  for i in range(0,loop):
    xs[:,i] = x;
    #dx = -L*x;
    #x = x+dx*dt; # =(E-dt*L)x
    x = P*x;
  print "consensus x",x.T;
  for i in range(0,n):
    plt.plot(range(0,loop),xs[i,:].A.flatten());
    plt.hold(True);
  plt.show();

# 長さMのベクトルの合意
# H:隣接行列
# X:N*Mのベクトル
def consensus2D(H,X):
  g = nx.from_numpy_matrix(H);
  L = nx.laplacian_matrix(g).todense();
  print "init x",X;
  n,m=X.shape;
  dt = 0.02; loop = 1000;
  xs = np.zeros((n,m,loop));
  P=np.eye(n)-dt*L; # ペロン行列
  for i in range(0,loop):
    xs[:,:,i] = X;
    #dx = -L*x;
    #x = x+dx*dt; # =(E-dt*L)x
    X = P*X;
  print "consensus x",X;
  return range(0,loop),xs;

# 劣勾配合意制御。T2@MAS6
# 長さMのベクトルの劣勾配合意
# H:隣接行列
# X:N*Mのベクトル
# J:Mベクトル入力の目的関数
def subGradientConsensus2D(H,X,J,dJ,dt=0.02,ds=1.,loop=1000):
  g = nx.from_numpy_matrix(H);
  L = nx.laplacian_matrix(g).todense();
  print "init x",X;
  n,m=X.shape;
  xs = np.zeros((n,m,loop));
  P=np.eye(n)-dt*L; # ペロン行列
  Js=np.zeros((n,loop));
  for i in range(0,loop):
    si = ds/(i+1);
    xs[:,:,i] = X;
    #dx = -L*x;
    #x = x+dx*dt; # =(E-dt*L)x
    D = X*0; # 勾配項
    # まず全agent一斉に現在のXから勾配を計算
    # ui
    for j in range(0,n):
      D[j] = dJ(np.mat(X[j,:]).A.ravel(),j);
      Js[j,i]=J(X[j,:]);
    X = P*X-si*D; # 全員一度に更新
  print "consensus x",X;
  print "avg X",np.mean(X.T,1).T,np.mean(X.T,1).shape;
  print "sum X",np.sum(X,1).T,np.sum(X,1).shape;
  return range(0,loop),xs,Js;

# 主双対勾配法
# min sum(ui)
# s.t. gi(xi)<=0
#      hi(xi)=0
#      xi=xj if (i,j) in E <=> Lx=0
# L=sum(ui)+sum(λi*gi)+sum(μi*hi)+ε*0.5*xLx
def pdga(H,X,ui,dui,gi,dgi,hi,dhi,dt=0.02,loop=1000,prnti=-1):
  g = nx.from_numpy_matrix(H);
  L = nx.laplacian_matrix(g).todense(); # n*n
  print "init x",X;
  n,m=X.shape;
  xs = np.zeros((n,m,loop));
  ls = np.zeros((3,loop));
  js = np.zeros((4,loop));
  P=np.eye(n)-dt*L; # ペロン行列
  #for i in range(0,n):
  #  print gi(X[i,:],i);
  dimg = len(gi(X[0,:],0));
  dimh = len(hi(X[0,:],0));
  l = np.ones((n,dimg))/n;
  u = np.ones((n,dimh))/n;
  e = np.ones((m,m))/m/m; # global双対変数

  for t in range(0,loop):
    xs[:,:,t] = X;
    #print "l,u,e",np.mean(sum(l)),np.mean(sum(u)),sum(sum(e));
    ls[:,t] = [np.mean(sum(l)),np.mean(sum(u)),sum(sum(e))];

    j = 0.; g = 0.; h = 0.; q = 0.;
    # まず現在のXから双対をupdateする
    # 各agentごとのlocal双対変数をupdate。勾配を足す
    for i in range(0,n):
      j += ui(X[i,:],i);
      if i == prnti:
        print "t,i,x,sumx,ui",t,i,X[i,:],sum(X[i,:]),ui(X[i,:],i);
      dLi = gi(X[i,:],i); # dL/dli
      g += sum(dLi);
      #print " gi",dLi;
      for k,agi in enumerate(dLi):
        if (agi < 0.) and (l[i,k] < 1e-10):
	  dLi[k] = 0.;
      l[i] += dLi*dt;
      dUi = hi(X[i,:],i); # dL/dui
      u[i] += dUi*dt;
      if i == prnti:
        print " hi",dUi;
      h += sum(dUi);
    # consensus双対変数をupdateする。勾配を足す
    dE = X.T*L*X;
    q += sum(sum(dE.A));
    e += dE*dt;

    js[:,t] =[j,g,h,q];
    
    dX = X*0; # 勾配項
    # まず全agent一斉に現在のXから勾配dX=dL/dxを計算
    for i in range(0,n):
      dX[i] = dui(np.mat(X[i,:]).A.ravel(),i); # [dui/dxi] = 1*m
      dX[i] += (l[i]*np.mat(dgi(X[i,:],i))).A.ravel(); # [dgi/dxi] = dimg*m
      dX[i] += (u[i]*np.mat(dhi(X[i,:],i))).A.ravel(); # [dhi/dxi] = dimh*m
      dX[i] += (sum(sum(e))*L[i,:]*X).A.ravel(); # [dQ/dxi] = [Li*x] = 1*m
    # 主Xをupdate。勾配を引く
    X -= dX*dt;
    
  print "consensus x",X;
  print "avg X",np.mean(X.T,1).T,np.mean(X.T,1).shape;
  print "sum X",np.sum(X,1).T,np.sum(X,1).shape;
  print "X>0.5";
  print [list(X[i,np.where(np.abs(X[i,:])>0.5)[0]]) for i in range(0,n)];
  return range(0,loop),xs,ls,js;

def plotFunc(f):
  fig = plt.figure();
  t = np.arange(0,10,0.1);
  plt.plot(t,f(t));
  plt.show();


def plotConsensus(t,xs,ls=None,js=None,m=None,oneplot=True,xlim=None,ylim=None):
  fig = plt.figure(1);
  if oneplot:
    ax = fig.add_subplot(221);
  else:
    ax = fig.gca();
  N,M,T=xs.shape;
  if m is None:
    m = M;
  j = 0;
  i = 0;
  #for i in range(0,N): # 全員のj番目genomeを比較
  for j in range(0,m): # player iの全genomeを比較
    ax.plot(t,np.mat(xs[i,j,:]).A.flatten()); 
    #ax.hold(True);
    #plt.ylim([0,1]);
  #plt.ylabel("x"); plt.xlabel("iteration");
  #plt.savefig("output/cesc.png");
  if not oneplot:
    fig = plt.figure(2);
    ax = fig.gca();
  else:
    ax = fig.add_subplot(222);
  if ls is not None:
    #plt.clf();
    M,T=ls.shape;
    for j in range(0,M):
      ax.plot(t,np.mat(ls[j,:]).A.flatten());
    print "l,u,e",ls[:,T-1];
  plt.ylabel("l");
  if not oneplot:
    fig = plt.figure(3);
    ax = fig.gca();
  else:
    ax = fig.add_subplot(223);
  if js is not None:
    M,T=js.shape;
    for j in range(0,M):
      ax.plot(t,np.mat(js[j,:]).A.flatten());
    print "j,g,h,q",js[:,T-1];
    if xlim is not None:
      plt.xlim(xlim);
    if ylim is not None:
      plt.ylim(ylim);
  plt.ylabel("j");
  plt.show();

##############

# 幾何学

# p2-p1のx軸となす角
def azimuth(p1,p2):
  dx = p2[0]-p1[0];
  dy = p2[1]-p1[1];
  th = np.arctan2(dy,dx);
  if th < 0:
    th += 2*np.pi;
  return th;

def norm(v):
  ret = 0.;
  for a in v:
    ret += a**2;
  return np.sqrt(ret);