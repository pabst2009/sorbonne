# -*- coding: utf-8 -*-

import platform;
irid = (platform.system()=='Linux');

import os;
import sys;
import numpy as np;
#import autograd.numpy as np;
import pylab;
import matplotlib.pyplot as plt;
from matplotlib.font_manager import FontProperties;
from matplotlib.gridspec import GridSpec
import copy;
from math import *;
from collections import Counter;
import itertools;
import scipy;
import random;
import networkx as nx;
import scipy.sparse as sp;
import scipy.sparse.linalg as spl;
import sympy;
import dill;
import pickle;
import time;
import timeout_decorator;
#from docplex.mp.model import Model;
import cProfile
import pstats

class Game:
  def __init__(self,A):
    self.A = A;
    self.N = len(A);
    self.dbg = False;
    self.epsilon = 1e-7; # 解計算用(pulp用)epsilon:1e-7
    self.isepsilon = 1e-5; # 解判定用epsilon:1e-5

  def here(self,ext=False):
    print("now in", self.__class__.__name__);
    print;
    if ext:
      exit();

  def size(self):
    return [len(a) for a in self.A];

  def pf2str(self,pf):
    return '_'.join([str(art) for art in pf]);

  def printU(self):
    players = range(0,self.N);
    for pf in self.pfs():
      print(pf,[self.calU(pf,i) for i in players]);
    print;

  def ph(self,h,sortkey=0):
    for k, v in sorted(h.items(), key=lambda x:x[sortkey]):
      print(k,v);
    print;

  def pl(self,l):
    for a in l:
      print(a);
    print;

  #@staticmethod
  def dump(self,obj,ofile='output/dbg.dump'):
    with open(ofile,mode='wb') as f:
      #pickle.dump(obj,f,-1); # クロスプラットフォームじゃない
      pickle.dump(obj,f,0); # クロスプラットフォーム。winSCPではバイナリモードじゃないとだめ
    print(ofile); sys.stdout.flush();

  #@staticmethod
  def load(self,ofile):
    #sys.path.append('../soton')
    #import sharegameSoton;
    with open(ofile,mode='rb') as f:
      obj = pickle.load(f);

    return obj;

  def players(self):
    if not hasattr(self,"setN"):
      for i in range(0,self.N):
        yield i;
    else:
      for i in self.setN:
        yield i;

  def otherp(self,I):
    ret = range(0,self.N);
    for i in I:
      ret.remove(i);
    return ret;

  def pfs(self):
    for pf in itertools.product(*(self.A)):
      yield pf;

  def calUs(self,pf):
    return np.array([self.calU(pf,i) for i in self.players()]);


  def devpair(self):
    for p in self.players():
      for i in self.A[p]:
        for j in self.A[p]:
          yield (p,i,j);

  # best responseチェック
  def isBR(game,pf,i,prnt=False):
    for bpf in game.deviate(pf,i):
      #print i,pf,bpf,game.calU(pf,i),game.calU(bpf,i);
      if game.calU(pf,i) < game.calU(bpf,i):
        if prnt:
          print(pf,"->",bpf);
          print("dev",i,game.calU(pf,i), "->", game.calU(bpf,i));
        return False;
    return True;
  
  def deviate(game,pf,i):
    bpf = deepcp(pf);
    for e in game.A[i]:
      bpf[i] = e;
      yield bpf;

  def catpf(self,ai,opf,i,others=None):
    if others is None:
      others = self.others(i);
    ret = [None]*(len(others)+1);
    ret[i] = ai;
    for i,j in enumerate(others):
      ret[j] = opf[i];
    return tuple(ret);

  '''
  def catpf(self,ai,opf,i,others):
    ret = np.zeros(len(opf)+1,dtype=np.int);
    ret[i] = ai;
    ret[others] = opf;
    return tuple(ret);
  '''

  def isnash(self,pf,prnt=False):
    A = self.A;
    N = self.N;
    players = range(0,N);
    for i in players:
      if not self.isBR(list(pf),i,prnt):
        return False;
    return True;

  # 2player gameのみ
  def matU(self):
    Aind = {};
    for i in self.players():
      if not (i in Aind.keys()):
        Aind[i] = {};
      for ai,a in enumerate(self.A[i]):
        Aind[i][a] = ai;

    # 2playerの場合
    m = len(self.A[0]); n = len(self.A[1]);
    u1 = np.zeros((m,n));
    u2 = np.zeros((m,n));
    for pf in itertools.product(*(self.A)):
      i = Aind[0][pf[0]];
      j = Aind[1][pf[1]];
      #print pf,(i,j),self.u[pf];
      u1[i][j] = self.u[pf][0];
      u2[i][j] = self.u[pf][1];
    return u1,u2;

  # SOを計算する
  def calSO(self,prnt=False):
    sws = []; so = -np.inf; sl = np.inf;
    for ne,us in self.pureNE():
      sws.append(sum(us));
      if sum(us) > so:
        so = sum(us);
      if sum(us) < sl:
        sl = sum(us);
      if prnt:
        print(ne,us,sum(us));
    esw = np.mean(sws);
    return so,sl,esw;

  # pureナッシュ均衡を表示する
  def nash(self):
    for ne in self.pureNE():
      print(ne);
    print;

	    
  def others(self,i):
    ret = list(self.players());
    ret.remove(i);
    return ret;

  # ２人ゲーの場合
  def opponent(self,i):
    if self.N > 2:
      raise NameError('opponent is only for 2 player games');
    if i == 0:
      return 1;
    else:
      return 0;
      

  def sw(self,pf):
    return np.sum(self.calUs(pf));
  
##### utility ######

def deepcp(o):
  return pickle.loads(pickle.dumps(o));

def some(l,n=None):
  for i,e in enumerate(l):
    if (n is not None) and (i > (n-1)):
      break;
    yield e;

def pp(x,s):
  if not dbg:
    return;
  print(s);
  print(x);
  print;

def randpf(game):
  N = game.N;
  A = game.A;
  players = range(0,N);

  pf = [];
  for i in players:
    pf.append(np.random.choice(A[i],1,replace=False)[0]);

  return pf;

# 単調増加か？
def ismonotone(x,y,descend=True,f=None):
  #print x,y;
  if f is not None:
    x = np.array(x); y = np.array(y);
    x = x[np.where(f)];
    y = y[np.where(f)];
    
  coef=1;
  if descend:
    coef=-1;
  ind = np.argsort(x);
  for k in range(len(x)-1):
    i = ind[k]; j = ind[k+1];
    if x[i] < x[j]:
      if y[i]*coef > y[j]*coef: # ascendなら符号は同じでないといけない
        return False;
    #print (x[i],x[j]),(y[i],y[j]);
  return True;

# 逆順にする
def rev(v):
  return list(reversed(v));

# 一次元にする
def flat(a):
  return np.reshape(np.asarray(a),np.prod(a.shape)).flatten();

# 要素が全て整数かどうか
def isint(v):
  return all(np.array([np.abs(av-np.int(av))<1e-7 for av in v]));

# v中の最後のeの次の要素
# なければNone
def nextlaste(l,e):
  v = np.array(l);
  i = np.where(v==e)[0];
  if (len(i) > 0) and (len(v[i[-1]:]) >1):
    return v[i[-1]+1];
  else:
    return None;

# vからk個重複なしランダムサンプリング
# k <= len(v)
def sample(v,k):
  p = np.random.permutation(len(v));
  return [v[i] for i in p[0:k]];

# vからk個重複ありランダムサンプリング
def sampledup(v,k):
  ind = np.random.randint(len(v),size=k);
  return [v[i] for i in ind];
  #return np.array(v)[ind];

# 多項分布(偏ったサイコロ)
# p={k1:p, k2:p, ...}
def multinomial(n,p,hash=False):
  res = np.random.multinomial(n,[v for k,v in sorted(p.items(),key=lambda x:x[0])]);
  if hash:
    return {k:res[i] for i,(k,v) in enumerate(sorted(p.items(),key=lambda x:x[0]))};
  v = [];
  for i,(k,_) in enumerate(sorted(p.items(),key=lambda x:x[0])):
    v += [k]*res[i];
  return sample(v,n);
  

def listzero(M,N):
  ret = [];
  for i in range(M):
    ret.append([]);
    for j in range(N):
      ret[i].append(0);
  return ret;

# i,jを入れ替える
def vswap(v,i,j):
  tmp = v[i];
  v[i] = v[j];
  v[j] = tmp;
  return v;

# vのi番目をj番目に移動させる
def vmove(v,i,j):
  ret = list(v);
  ret.pop(i);
  ret.insert(j,v[i]);
  return ret;

# リストのi番目を0から順に後ろにずらすリスト
def orders(l,i):
  others = deepcp(l);
  others.pop(i);
  for j in range(len(l)):
    l2 = deepcp(others);
    l2.insert(j,l[i]);
    yield l2;

# ((a1,a2),(b1,b2),..)→((a1,b1,..),(a2,b2,..))
def vtranspose(v,tupl=False):
  ret = listzero(len(v[0]),len(v));
  for i,at in enumerate(v):
    for j,aat in enumerate(at):
      ret[j][i]=v[i][j];
  if tupl:
    return tuple(tuple(a) for a in ret);
  return ret;

# sublistを取得。空リストはなし
def sublist(l):
  n = len(l);
  for i in range(1,n+1):
    yield l[0:i];

def randsort(v):
  return sample(v,len(v));

# nをl個に配分する方法
def allocN2L(n,l):
  for a in itertools.product(*(range(0,n+1) for al in range(0,l))):
    if (sum(a) <= n):
      yield a;

# nをl個に配分する方法(和がnになる)
def allocN2Lexact(n,l):
  for a in itertools.product(*(range(0,n+1) for al in range(0,l))):
    if (sum(a) == n):
      yield a;

def intsqrt(i,l):
  k = 1.0/l;
  return int(np.round(float(i)**k));

# 非一様分布
def nnrandint(a,b,k):
  return intsqrt(np.random.randint(a**k,(b-1)**k+1),k);

def corrrand(a,b,k,corr,n):
  # 最初の一人は偏った純ランダム
  ret = np.zeros(n,dtype=np.int);
  ret[0] = nnrandint(a,b,k)-np.float(3*a+b)/2;
  for i in range(1,n):
    if i==1:
      j = 0;
    else:
      j = np.random.randint(0,i); # すでに値の決まってる誰か
    rtmp = np.random.randint(a,b)-np.float(3*a+b)/2;
    ret[i] = corr*ret[j]+np.sqrt(1.-corr**2)*rtmp;

  # [a,b]に戻す
  for i in range(0,n):
    ret[i] = int(np.round(ret[i]+np.float(3*a+b)/2));

  #print "fix correlation between player 0 and 1 in gb.corrrand";
  return np.array(list(ret));

  # 順序をランダムに入れ替える
  return np.array(sample(ret,n));
  

# リストをランダムに分割
def randsep(l):
  ret = [];
  n = len(l);

  #l = [nnrandint(1,4,6) for i in range(0,100)];
  #print sorted(l);
  #exit();

  rest = range(n);
  # restが無くなるまで
  while len(rest) > 0:
    #m = np.random.randint(1,n+1); # サブリストの長さ(一様)
    #m = nnrandint(1,n+1,6); # サブリストの長さ(非一様。長いほうが大きい)

    m = n; # 分割しない
    #print "randsep not separate!";
    
    #print "nnrand",n,m;
    subl = sample(rest,m);
    if len(subl) > 0:
      ret.append([l[a] for a in subl]);
    rest = list(set(rest)-set(subl));
  return ret;

#### 集合系: set ####

def powerset(s):
  return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1));

#### ハッシュ系:hash ####

# キーでソートした値ベクトル
def h2v(h):
  return [v for k,v in sorted(h.items(),key=lambda x:x[0])];

# キーでソートした値array
def h2a(h):
  return np.array(h2v(h));

# 値でソートしたキーベクトル
def h2k(h,coef=1):
  return [k for k,v in sorted(h.items(),key=lambda x:coef*x[1])];

# 値でソートした順序ハッシュ
def h2o(h,coef=1):
  return {k:i for i,(k,v) in enumerate(sorted(h.items(),key=lambda x:coef*x[1]))};

# キーでソートしたtuple
def h2t(h):
  return tuple((k,v) for k,v in sorted(h.items(),key=lambda x:x[0]));

def t2h(t):
  return {e[0]:e[1] for e in t};

# キーでの引き算。キーは同じ
def hdiff(h1,h2):
  return {k:h1[k]-h2[k] for k,v in sorted(h1.items(),key=lambda x:x[0])};

# キーでの足し算。キーは同じ
def hadd(h1,h2):
  return {k:h1[k]+h2[k] for k,v in sorted(h1.items(),key=lambda x:x[0])};

# 定数をかける
def hmul(h,a):
  return {k:v*a for k,v in h.items()};

# 値の和
def hsum(h):
  return np.sum(h2a(h));

# 値のwhere。boolianの時は値省略。キーリストを返す
def hwhere(h,v=None):
  if v is None:
    return [k for k,av in h.items() if av];
  else:
    return [k for k,av in h.items() if av==v];

# キーリストの値をキーでソートしてリストで取り出す
def hget(h,vk):
  return [v for k,v in sorted(h.items(),key=lambda x:x[0]) if k in vk];


#### 確率系:prob ####

# [0,1]の範囲内にする(主にceで使う)
def roundp(x):
  x = np.abs(x);
  if x > 1:
    return 1.;
  err = 0.0000001;
  if x < err:
    return 0.0;
  return x;

#### グラフ系:graph ####

# グラフ描画
def chkg(g,filename=None,flabel=None):
  plt.clf();
  labels = {};
  for n in g.nodes():
    labels[n] = n;
    if flabel is not None:
      labels[n] = flabel(n,g.node[n]);
    '''
    if (g.out_degree(n) == 0): # 葉
      labels[n] = n;
    else:
      t,i,loc,tmp,h,tmp,x,tmp = g.node[n]['obj'];
      labels[n] = "%d_%d_%d" % (i,h,n);
    '''
  #npos=nx.spring_layout(g,k=10.0,iterations=120);
  #nx.draw(g,npos,labels=labels);
  nx.draw(g,labels=labels);
  if filename is not None:
    plt.savefig(filename);
    print(filename);
  else:
    plt.show();

def undirected(g):
  ug = nx.Graph();
  for o,d in g.edges():
    ug.add_edge(o,d);
  return ug;

# 有向グラフのconnected
def connected(g):
  if g.order() > 0:
    return nx.is_connected(g.to_undirected());
  return False; # ノードなければfalse

# 幅優先探索
def bfsn(g):
  h = indexHash();
  for o,d in nx.bfs_edges(g,0):
    h.append(o);
    h.append(d); # leafも含む
  for k,v in h.each():
    yield k;

# DAG子孫グラフ(n含む)
def descendants(g,n):
  ret = [n];
  for o,d in nx.bfs_edges(g,n):
    ret.append(d);
  return g.subgraph(set(ret));

# DAG treewidth
# 全ノードの(子degree-親degreeを足す)
def treewidth(g):
  ret = 0;
  for n in g.nodes():
    w = g.out_degree(n)-g.in_degree(n);
    if w < 0:
      w = 0;
    ret += w;
  return ret;

# leafリスト
def leaves(g,loop=False):
  ret = [];
  for n in g.nodes():
    if g.out_degree(n)==0:
      ret.append(n);
    elif loop and (g.out_degree(n)==1):
      succ = g.successors(n);
      if not irid:
        succ = list(succ);
      if (n==succ[0]):
        ret.append(n);
  return set(ret);

# パスの中のエッジループ
def edgesin(path):
  for i in range(1,len(path)):
    yield path[i-1],path[i];

def lastedge(path):
  return path[-2],path[-1];

# o,d間の全てのパス
# まだ不完全。最短経路のみ
def allpaths(g,o,d):
  for path in nx.all_shortest_paths(g,o,d):
    yield path;

# 幅優先成長
def makeTree(gsize,maxdegree,g=None,n=None):
  if g is None:
    g = nx.Graph();
    n = "0";
    g.add_node(n); # ルート

  layer = [n];

  while g.order() < gsize:
    newlayer = [];
    # 成長させる
    #print "maketree same layer same action";
    mc = np.random.randint(2,maxdegree+1); # 同じレイヤは同一個数
    
    for an in layer: # この層の全ノードを処理(幅優先)
      #mc = np.random.randint(2,maxdegree+1); # 同じレイヤでも別個数
      #print mc;
      for i in range(0,mc):
        c = an+"_%d" % i;
        #print c;
        g.add_edge(an,c);
        newlayer.append(c);
        if g.order() >= gsize:
          return g;
    layer = newlayer;

  return g;


def makeTreeold(gsize,maxdegree,g=None,n=None):
  if g is None:
    g = nx.Graph();
    n = "0";
    g.add_node(n); # ルート

  if g.order() >= gsize:
    return g;

  # 成長させる
  mc = np.random.randint(2,maxdegree+1);
  cs = [];
  for i in range(0,mc):
    c = n+"_%d" % i;
    print(c);
    g.add_edge(n,c);
    cs.append(c);

  # 深さ優先成長
  for c in cs:
    g = makeTree(gsize,maxdegree,g,c);

  return g;

##### 最適化関係:optim #####

# x>0にするバリア
def positivebarrier(x):
  if x>0:
    return -np.log(x);
  else:
    return np.log(-x);

# デルタ関数
def delta(x):
  s = 0.5;
  return np.exp(-(x**2)/(2*s**2))/s-1.0/s;

##### 離散幾何関係:geo #####

# x軸との角度[0,2pi]
def azimuth(v):
  if np.abs(v[0]) < sys.float_info.epsilon:
    gamma = np.arctan(v[1])*2;
  else:
    gamma = np.arctan(v[1]/v[0]);
    if v[0] < 0.0:
      gamma = (np.pi + gamma);
  if gamma < 0.0:
    gamma += 2 * np.pi;
  return gamma

# 辞書(N)に対する交点
# A[N]y=b[N]
def vertex(A,b,ind):
  AN = A[ind,:];
  bN = b[:,ind];
  if np.linalg.matrix_rank(AN) < len(ind):
    return None;
  return (AN.I*bN.T).T; # 交点


# h[k] = id
class indexHash(dict):
  def __init__(self):
    pass;

  def append(self,k):
    if k not in self:
      self[k] = len(self);

  def tolist(self):
    ret = [];
    for k, v in sorted(self.items(), key=lambda x:x[1]):
      ret.append(k);
    return ret;

  def each(self):
    ret = [];
    for k, v in sorted(self.items(), key=lambda x:x[1]):
      yield k,v;

class setHash(dict):
  def __init__(self):
    pass;

  def append(self,k,v):
    if k not in self:
      self[k] = [v];
    else:
      self[k].append(v);      
      self[k] = list(set(self[k]));

  def each(self):
    ret = [];
    for k, v in sorted(self.items(), key=lambda x:x[0]):
      yield k,v;

###### normal game instance ########

def instanceEFCEnotCE():
  A = np.array([range(0,8),range(0,4)]);
  gmw = AtomG(A);
  gmw.u[(0,0)] = [4,10];
  gmw.u[(0,1)] = [4,10];
  gmw.u[(0,2)] = [0,6];
  gmw.u[(0,3)] = [0,6];
  gmw.u[(1,0)] = [4,10];
  gmw.u[(1,1)] = [4,10];
  gmw.u[(1,2)] = [0,6];
  gmw.u[(1,3)] = [0,6];
  gmw.u[(2,0)] = [4,10];
  gmw.u[(2,1)] = [0,6];
  gmw.u[(2,2)] = [4,10];
  gmw.u[(2,3)] = [0,6];
  gmw.u[(3,0)] = [4,10];
  gmw.u[(3,1)] = [0,6];
  gmw.u[(3,2)] = [4,10];
  gmw.u[(3,3)] = [0,6];
  gmw.u[(4,0)] = [6,0];
  gmw.u[(4,1)] = [6,0];
  gmw.u[(4,2)] = [0,6];
  gmw.u[(4,3)] = [0,6];
  gmw.u[(5,0)] = [6,0];
  gmw.u[(5,1)] = [0,6];
  gmw.u[(5,2)] = [6,0];
  gmw.u[(5,3)] = [0,6];
  gmw.u[(6,0)] = [6,0];
  gmw.u[(6,1)] = [6,0];
  gmw.u[(6,2)] = [0,6];
  gmw.u[(6,3)] = [0,6];
  gmw.u[(7,0)] = [6,0];
  gmw.u[(7,1)] = [0,6];
  gmw.u[(7,2)] = [6,0];
  gmw.u[(7,3)] = [0,6];
  return gmw;