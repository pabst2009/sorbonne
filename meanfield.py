# -*- coding: utf-8 -*-

from gridbru import GridBRU;
from groupgridgame import GroupGridGame;
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

class MeanField(GridBRU):
  def __init__(self,n,space,f,u,u_,exc=True,x0=None,dbgi=-1):
    GridBRU.__init__(self,n,space,exc,x0=x0);
    self.dbgi = dbgi;
    self.gso = None;
    self.f = f; # 場関数
    #self.ucore = u;
    #self.u = lambda i,x : u(i,x,f); # 利得関数。全員がui呼ぶたびに毎回fを計算するので遅い
    self.u_ = u_;


  def df(self,a,i):
    a_i = [a[j] for j in self.players() if j != i];
    f = self.f(a);
    f_i = self.f(a_i);
    return np.linalg.norm(f-f_i);

  def u(self,i,a):
    return self.ulocal(i,a);
    #return self.ucore(i,a,self.f);

  def ulocal(self,i,a):
    f = self.f(a);
    return self.u_(i,a[i],f);

  # 遅い:print文だけだから0返してもok
  def ext(self,a,i,prnt=False):
    #return 0;
    a_i = [a[j] for j in self.players() if j != i];
    f = self.f(a);
    f_i = self.f(a_i);
    ret = 0.;
    for j in self.others(i):
      #if a[j] != a[i]: # MFdistのみ
      #  continue;
      ext = self.u_(j,a[j],f)-self.u_(j,a[j],f_i);
      ret += ext;
      if prnt and (np.abs(ext) > 0):
        # MFdist用
        print "ext j",j,a[j],"Nx",f[a[j]]*self.N,"->",f_i[a[j]]*self.N,"fx",f[a[j]],"->",f_i[a[j]];
	print " uj",self.u_(j,a[j],f),"->",self.u_(j,a[j],f_i),"extj",self.u_(j,a[j],f)-self.u_(j,a[j],f_i);
      #ret += np.abs(self.u_(j,a[j],f)-self.u_(j,a[j],f_i)); # 絶対値取るとでかい
    return ret;

class ModularMFG(GroupGridGame):
  def __init__(self,n,space,ri,f,u,u_,exc=True,x0=None,dbgi=-1):
    GroupGridGame.__init__(self,n,space,ri,exc,x0=x0);
    self.dbgi=dbgi;
    self.gso = None; # so計算用
    self.f = f; # 場関数
    self.u = lambda i,x : u(i,x,f); # 利得関数。全員がui呼ぶたびに毎回fを計算するので遅い
    self.u_ = u_; # local項

  def ulocal(self,i,a):
    f = self.f(a);
    return self.u_(i,a[i],a,f);

  # local項のexternality
  def ext(self,a,i):
    a_i = [a[j] for j in self.players() if j != i];
    f = self.f(a);
    f_i = self.f(a_i);
    ret = 0.;
    for j in self.others(i):
      ret += self.u_(j,a[j],a,f)-self.u_(j,a[j],a_i,f_i);
      #ret += self.lmd*self.N*self.J_(a[j],a,f)-self.lmd*(self.N-1)*self.J_(a[j],a_i,f_i);
    return ret;

