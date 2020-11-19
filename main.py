#!c:/Python27/python -u
# -*- coding: utf-8 -*-
#c:/Python3564/python
from __future__ import division;

'''
Joint collaboration between Toyota and Sorbonne Université
Tatsuya Iwase (Toyota)
Aurélie Beynier (SU), Nicolas Bredeche (SU), Nicolas Maudet (SU)
'''

import numpy as np;
import sys;
import problem;
import solver;
import util;
import time;
import cProfile
import pstats
import gamebase as gb;

def meanfield(s,E,N=80,W=40,R=5):
  np.random.seed(s);
  #N = 40; T = 1; W = 30; space = (W,W);
  #N = 200; T = 1; W = 60; space = (W,W);
  #N = 100; T = 1; W = 40; space = (W,W);
  #prb = problem.MFmodular(N,space,wall=0,anime=2);
  T = 1; space = (W,W); xfood = np.array([(10,W/2),(30,W/2)]);
  xfood = np.array([(10,10),(30,10),(20,30)]);
  ##N = 40; T = 1; W = 30; space = (W,W); xfood = np.array([(7,7),(23,7),(15,23)]); # 汚い
  #prb = problem.MFforage(N,space,xfood,R,E,wall=0,anime=1);

  rbag = 7; rmax = 20; space = (W,W);

  #prb = problem.MForganize(N,space,R,rbag,rmax,E,wall=0,anime=1); # anime=0:なし, 1:png, 2:gifも

  #prb = problem.MFdrawing(N,(W,W),'input/star400',wall=0,anime=0); # N=200,W=60
  #prb = problem.MFdist(N,(W,W),R,wall=0,anime=1);
  prb = problem.MFcircle(N,(W,W),wall=0,anime=1); # N=80,W=40 # ai=xi # IAT2020
  #prb = problem.MFformation(N,(W,W),wall=0,anime=1); # N=80,W=40 # ai=xi


  #prb.so(); # gsoを計算
  #gb.myprofile(slv.runBRU,prb,T=1); exit();
  f,err = prb.showresult(prnt=0);
  sol = slv.runBRU(prb,T=T);
  print "before f";
  print f;
  print "before err",err;
  prb.showresult();

########## main ###########

irid = 0;
task = 2; # 0:mileage, 1:namecard, 2:aggregate

seed = 0;
np.random.seed(seed);
np.set_printoptions(precision=2);

slv = solver.Solver();

##### swarm grid problem #####

argv = [np.int(e) for e in sys.argv[1:]];
s,E,N,W,R = argv; # main.py 0 -1 80 40 5

meanfield(s,E,N,W,R); exit();

