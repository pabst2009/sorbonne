# -*- coding: utf-8 -*-

from gridgame import GridGame;
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

class GridBRU(GridGame):
  def __init__(self,n,space,exc=True,obstacle=None,x0=None):
    GridGame.__init__(self,n,space,exc=exc,obstacle=obstacle,x0=x0);

  # ポリシーではなくPFのBRU
  # meanfield用
  def bru(self,T,showres=True,stopcycle=True):
    ts = [];
    sss = [];
    rawss = [];
    pf = self.initpf();
    if self.gso is not None:
      for i in self.players():
        pf[i] = (self.gso[i],pf[i][1]);
    dbgi = self.dbgi;
    wupdated = True;
    cycle = False;
    hpf = {};
    t = 0;
    while wupdated: # だれかupdateしたら
      if cycle:
        break;
      if 0 and (t >= T):
        break;
      wupdated = False;
      ss = [];
      for i in self.players():
        tmppf = list(pf); # deepcp
        bestui = self.u(i,tmppf); # 各iごとにBRU
	bestuil = self.ulocal(i,tmppf);
	if i==dbgi:
	  print "#### initial i,ai,ui",tmppf[i],bestui;
	for ai in self.Ai(i): # 現在のxでの選択肢
          if self.gso is not None:
	    ai = (self.gso[i],ai[1]);
	  tmppf[i] = ai;
	  tmpui = self.u(i,tmppf);

	  if i == dbgi:
	    print "####### i",i,pf[i],"->",tmppf[i],"u",bestui,"->",tmpui;	  
	  dui = tmpui-bestui;
	  duil = self.ulocal(i,tmppf)-bestuil;
	  if dui>0.:
	    #self.du(i,pf,tmppf); # デバッグ用
	    dext = self.ext(tmppf,i)-self.ext(pf,i);
	    #df = self.df(tmppf,i);

	    #print "i,dui",i,dui,"dext",dext,"df,dext/df,Ndf,avg(dext/df)",np.array([df,dext/df,self.N*df,dext/df/(self.N-1)]);
	    if 0 and (i==14):
	      print "#######",pf[i],"->",tmppf[i],"u",bestui,"->",tmpui;
	      print self.f(tmppf)*self.N;
	    print "i",i,"dui,duil,dext=Nduj",np.array([dui,duil,dext]), "SS",(duil > -dext);
	    ss.append(duil>-dext);
	    if 0 and (duil < -dext):
	      print " externality cannot be ignored. sw:",self.sw(pf),"->",self.sw(tmppf);
	      print " i,ai,bi",i,pf[i],"->",tmppf[i],"ui",bestui,"->",tmpui;
	      if 1:
	        print self.f(pf)*self.N;
	        print " f",self.f(pf)[pf[i]]*self.N,"->",self.f(tmppf)[tmppf[i]]*self.N;
	        print "ext ai";
	        self.ext(pf,i,prnt=1);
	        print "ext bi";
	        self.ext(tmppf,i,prnt=1); exit();
	        #print self.f(tmppf)*self.N;
	    
	    bestui = tmpui;
	    pf = list(tmppf);
	    self.transition(pf);
	    if i==dbgi:
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
      print "t,ss",t,np.mean(ss);
      ts.append(t); sss.append(np.mean(ss));
      rawss += ss;
      yield t,i,pf; t+=1;

    if showres:
      print "avg rational rate",np.mean(rawss);

