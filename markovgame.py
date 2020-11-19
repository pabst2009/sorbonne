# -*- coding: utf-8 -*-

from gamebase import Game;
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

# 状態xによって利得関数が変わる
class MarkovGame(Game):
  def __init__(self,n,Ai,x0):
    Game.__init__(self,x0);
    self.A = Ai;
    self.x = x0;

  def initpf(self):
    return self.x;

  def getx(self):
    return self.x;

  def setplayer(self,N):
    self.x = {i:self.x[j] for j,i in enumerate(N)};
    self.setN = N;

  # policyでなくpfを列挙
  def pfs(self):
    1;

  def transition(self,pf):
    1;

  def sw(self,a):
    ret = 0.;
    for i in self.players():
      ret += self.u(i,a);
    return ret;

