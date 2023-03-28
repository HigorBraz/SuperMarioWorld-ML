import sys
import os
import retro
import numpy as np

from rominfo import*
from utils import *

raio = 6

#parametros do reinforcement learning
gamma = 0.6
alpha = 0.6
epsilon = 0.1

lives = 4                           #numero de vidas pra controlar quando o agente morre (ele ganha uma vida nas tartarugas, caso as mate)
moves = [128, 130, 131, 386, 64]    #movimentos permitidos
rew = [2, 10, 3, 5, 1]              #recompensa pra cada movimento (direita vale mais, esquerda vale menos)

import random


#calcula a distancia do agente ate a chegada
def distancia(estado, x):
    estNum = np.reshape(list(map(int, estado.split(','))), (2*raio+1,2*raio+1))
    dist = np.abs(estNum[:raio+1,raio+2:raio+7]).sum()
    return ((4800 - x)/8) + 0.3*dist

#retorna a melhor acao, caso empate sera sorteada
def QAction(q_table, estado):
    qvals = np.array([q_table.get(str(estado))[i]  for i in range(0, 5)])
       
    maxval = np.max(qvals)
    if (qvals == maxval).sum() > 1:
        idx = choice(np.nonzero(qvals==maxval)[0])
    else:
        idx = np.argmax(qvals)
    
    return idx

if sys.argv[1] == 'train':
    if os.path.exists('Q2.pkl'):
        q_table = pickle.load(open('Q2.pkl', 'rb'))
    else:
        q_table = {}
    
else:
    #recupera a tabela salva
    if os.path.exists('Q1.pkl'):
        q_table = pickle.load(open('Q1.pkl', 'rb'))
    else:
        q_table = {}

env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)

            
for i in range(1, 2):
    env.reset()

    #distancias para controlar quanto o agete anda de estado a estado
    new_dis = 0
        
    estado, x, y = getState(getRam(env), raio)    

    maior_dis = distancia(estado, x)

    done = False
    
    
    #se ainda nao existe o estado inicial, inicializa
    if q_table.get(str(estado), 0) == 0:
        q_table[str(estado)] = [0,0,0,0,0]
        
    #enquanto o agente nao termina a fase:     
    while not done:
        env.render()
        
        dead = 0     
        
        print('Tamanho da tabela: ', len(q_table))
        
        #atualiza o valor de epsilon
        epsilon = (epsilon * 400) / (len(q_table) + 1)
        
        desconto = 1         
        
        old_dis = distancia(estado, x)
        
        #fator aleatorio, baixo pq ele ja tem aleatoriedade na funcao QAction, serve apenas caso o agente trave num loop
        if random.uniform(0,1) < epsilon:
            action_idx = random.choice([0,1,2,3,4])
            action = moves[action_idx]
        
        else:
            #recupera a melhor acao, caso exista
            action_idx = QAction(q_table, estado)
            action = moves[action_idx]
            desconto = rew[action_idx]
           
        #realiza a acao e gera o prox estado
        done, info = performAction(action, env)       
        prox_estado, x, y = getState(getRam(env), raio)
        
        #se o prox estado nao esta na tabela inicializa
        if q_table.get(str(prox_estado), 0) == 0:
            q_table[str(prox_estado)] = [0,0,0,0,0]
        
        #caso o agente morra reseta o env
        if info.get('lives') + 1 == lives:
            new_dis = distancia(prox_estado, x)
            env.reset()
            dead = 1
            lives = 4
            if  new_dis < maior_dis:
                maior_dis = new_dis             
         
        #caso ele nao morra:
        else:
            new_dis = distancia(prox_estado, x)               
              
            #se ele chegou no fim done = True
            if new_dis < 0:
                done = True
                env.reset()
                lives = 4
            
            #se ele voltou perde a recompensa, caso contrario calcula a recompensa com base na distancia percorrida
            else:
                if old_dis - new_dis <= 0:
                    reward = -100                    
                else:
                    reward =  desconto * (old_dis - new_dis)
                #atualiza as vidas
                lives = info.get('lives')
                
        #se ele tiver morrido nao ganha recompensa
        if dead == 1:
            new_value = 0
        
        #caso contrario calcula o new_value
        else:
            old_value = q_table[str(estado)][action_idx]
            next_max = np.max(q_table[str(prox_estado)])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        
         
        #preenche a tabela na posicao [estado][acao] com new_value
        q_table[str(estado)][action_idx] = new_value
        estado = prox_estado      
        
        if sys.argv[1] == 'best':
            #salva a tabela
            fw = open('Q1.pkl', 'wb')
            pickle.dump(q_table, fw)
            fw.close()
            
        if sys.argv[1] == 'train':
            #salva a tabela
            fw = open('Q2.pkl', 'wb')
            pickle.dump(q_table, fw)
            fw.close()