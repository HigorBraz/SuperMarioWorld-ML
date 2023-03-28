import sys
import os
import retro
import numpy as np
import pickle

from rominfo import*
from utils import*

import random


raio = 6
lives = 4   #numero de vidas pra controlar quando o agente morre

#movimentos permitidos
moves = [128, 130, 131, 386, 64]


#retorna a melhor acao, caso empate ele sorteia
def QAction(q_table, estado):
    #caso o agente nao tenha treinado tal estado, retorna [0,0,0,0,0] e sorteia a acao
    qvals = np.array([q_table.get(str(estado),[0,0,0,0,0])[i]  for i in range(0, 5)])
       
    maxval = np.max(qvals)
    if (qvals == maxval).sum() > 1:
        idx = choice(np.nonzero(qvals==maxval)[0])
    else:
        idx = np.argmax(qvals)
    
    return idx

#calcula a distancia do agente ate a chegada
def distancia(estado, x):
    estNum = np.reshape(list(map(int, estado.split(','))), (2*raio+1,2*raio+1))
    dist = np.abs(estNum[:raio+1,raio+2:raio+7]).sum()
    return ((4800 - x)/8) + 0.3*dist


if sys.argv[1] == 'best':
    q_table = pickle.load(open('Q1.pkl', 'rb'))  
    
elif sys.argv[1] == 'train':
    q_table = pickle.load(open('Q2.pkl', 'rb'))

done = False

env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
env.reset() 
estado, x, y = getState(getRam(env), raio)
         
while not done:
    env.render()
    
    #distancia do estado atual
    old_dis = distancia(estado, x)
 
    #recupera a melhor acao da tabela para o estado atual
    action_idx = QAction(q_table, estado)
    action = moves[action_idx]
    
    #executa a acao
    done, info = performAction(action, env)
    
    estado, x, y = getState(getRam(env), raio)
    
    #distancia do prox estado
    new_dis = distancia(estado, x)   
    
    #se ele ficar travado nas mensagens "aperta A", ele nao vai pra frente so pula
    if old_dis - new_dis == 0:
        done, info = performAction(131, env)
    
    #se ele chegou no fim done = true
    if new_dis <= 0:
        done = False
        
    #se ele morreu reseta 
    #done nao vira true pq ele gera acao aleatoria para os estados que nao treinou e quando as recompensas empatam
    if info.get('lives') + 1 == lives:
            env.reset()
            estado, x, y = getState(getRam(env), raio)
            lives = 4
    