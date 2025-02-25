import math
import EoN
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import numpy as np
from multiprocessing import Process, Queue
import queue
import time
import os
import argparse
import sys


gamma = 1 
sigma1 = 0.4
sigma2 = 0.1
lambda1 = 0.1
lambda2 = 0.01

ITERATIONS= 50
RLX_TIME = 100
AV_TIME = 300
N_NODES = 1000
POOL_SIZE = 40
    

DEBUG = False
VERBOSE = False
QUEUE_TIMEOUT = 10


def get_c(w, lambda1, lambda2):
    f1 = (w*lambda2 - lambda1) / (2*(1-w))
    f2 = np.sqrt(1 + ( (4*w*lambda1*lambda2*(1-w)) / ((lambda1-w*lambda2)**2) ) )
    c1 = f1 * ( 1 - f2)
    c2 = f1 * ( 1 + f2)
    c = 0
    if c1 > 0:
        c = c1
    else:
        c = c2
    return c


def get_alpha(c, w, lambda1, lambda2):
    alpha = 0
    f1 = ((lambda1+c)/(c*w))
    f2 = np.sqrt(1 - (w/(lambda1+c)) * (c+lambda2*w))
    alpha1 = f1 * (1 + f2)
    alpha2 = f1 * (1 - f2)
    if alpha1 > 1:
        alpha = alpha2
    else:
        alpha = alpha1
    if alpha > 1:
        alpha = 1
    return alpha


def get_beat_c(c, alpha, lambda1, lambda2, sigma1, sigma2):
    num = c + lambda1 + ( (alpha*(1-alpha)*c**2) / lambda2  )
    den = (1-alpha)*c + lambda1 + alpha*sigma1*c + ( (sigma2*alpha*(1-alpha)*c**2) / lambda2 )
    beta_c = num/den
    return beta_c




def update_Pqs( data, Pqs, startTime, maxTime, n_nodes, n_infections, initInfected):
    # 1. count infection events
    for j in range(n_nodes):
        _, statusHistory = data.node_history(j)
        for h,k in enumerate(statusHistory):
            # h >= 1, to skip the initial status
            if h>=1 and k in ['I0', 'I1']:
                n_infections += 1           
    # 2. check reflecting boundary and update P_qs       
    initStatus = {}
    timeLine, statusDict = data.summary()
    infected = statusDict['I1']+statusDict['I0']
    lastTime = startTime
    lastInfected = initInfected
    for i,s in enumerate(timeLine):
        # We know there is at least 1 infected node in the Initiale State.
        # Thus (infected[0] != 0) it is always true and we always set lastTime=s
        if infected[i] == 0: 
            # we got 0 infected at time timeLine[i], we go back to timeLine[i-1]
            for j in range(n_nodes):
                initStatus[j] = data.node_status(j, lastTime)
                if initStatus[j] in ['I0', 'I1']:
                    initInfected += 1
            return lastTime, initStatus, n_infections, initInfected
        Pqs[lastInfected] = Pqs[lastInfected] + (s-lastTime)
        lastTime = s
        lastInfected = infected[i]
    Pqs[lastInfected] = Pqs[lastInfected] + (maxTime-lastTime)
    
    for j in range(n_nodes):
        initStatus[j] = data.node_status(j, maxTime)
        if initStatus[j] in ['I0', 'I1']:
            initInfected += 1
    if initInfected == 0:
        print(f"\t[ERROR] There are 0 infected nodes!!!")
        exit()
    return maxTime, initStatus, n_infections, initInfected




def apply_reflecting_boundary( data, maxTime, n_nodes ):
    # 1. check reflecting boundary (relaxation phase)
    initStatus = {}
    initInfected = 0
    timeLine, statusDict = data.summary()
    infected = statusDict['I1']+statusDict['I0']
    for i,s in enumerate(timeLine):
        # We know there is at least 1 infected node in the Initiale State.
        # Thus (infected[0] != 0) it is always true and we always set lastTime=s
        if infected[i] == 0: 
            # we got 0 infected at time timeLine[i], we go back to timeLine[i-1]
            for j in range(n_nodes):
                initStatus[j] = data.node_status(j, lastTime)
                if initStatus[j] in ['I0', 'I1']:
                    initInfected += 1
            return lastTime, initStatus, initInfected
        lastTime = s
        
    # 2. create the initial status for the averaging phase
    if DEBUG:
        print(f"\t[DEBUG] We start the AVERAGING PHASE with {infected[-1]} infected nodes.")
    for j in range(n_nodes):
        initStatus[j] = data.node_status(j, maxTime)
        if initStatus[j] in ['I0', 'I1']:
            initInfected += 1
    if initInfected == 0:
        print(f"\t[ERROR] There are 0 infected nodes!!!")
        exit()
    return maxTime, initStatus, initInfected



#
# Reflecting Boundary Condition (RBC)
#
def multi_run_RBC(queue_beta, pid, c, alpha, iterations, rlx_time, av_time, graph, output_dir):
    # For an ALPHA and a BETA we run iterations simulations
    v0 = c*alpha
    v1 = (1-alpha)*c
    n_nodes = graph.number_of_nodes()
    beta_dict = {}
    print(f"[START Process {pid}]")
    
    while True:
        try:
            beta_value = queue_beta.get(block=True, timeout=QUEUE_TIMEOUT)
            print(f"[START Process {pid}] Alpha: {alpha}, Beta: {beta_value}")
            H, J = get_H_J(v0, v1, gamma, lambda1, lambda2, beta_value, sigma1, sigma2)
            rebound_list = []
            infectionEvents_list = []
            Pqs_list = []
            rho_list = []
            rho_2_list = []
            X_list = []
            DELTA_list = []
            
            tic = time.perf_counter()
            for counter in range(iterations): #For ALPHA, BETA
                # Set-up Network
                IC = defaultdict(lambda: 'I0')
                return_statuses = ('S0', 'I0', 'S1', 'I1', 'S2')
                initStatus = IC

                # 1. RELAXATION TIME 
                n_rebound = -1
                endTime = rlx_time
                startTime = 0
                while startTime<endTime:
                    n_rebound += 1
                    simInv = EoN.Gillespie_simple_contagion(graph, H, J, initStatus, return_statuses, return_full_data=True, tmin = startTime, tmax = endTime)
                    startTime, initStatus, initInfected = apply_reflecting_boundary(simInv, endTime, n_nodes)
                if VERBOSE:
                    print(f"\t[RELAXATION TIME {counter}] rebounds: {n_rebound} (beta: {beta_value}, alpha: {alpha})")
                
                # 2. AVERAGING TIME
                n_rebound = -1
                n_infections = 0
                endTime = av_time
                startTime = 0
                Pqs = defaultdict(lambda: 0)
                while startTime<endTime:
                    n_rebound += 1
                    simInv = EoN.Gillespie_simple_contagion(graph, H, J, initStatus, return_statuses, return_full_data=True, tmin = startTime, tmax = endTime)
                    startTime, initStatus, n_infections, initInfected = update_Pqs(simInv, Pqs, startTime, endTime, n_nodes, n_infections, initInfected)
                if VERBOSE:
                    print(f"\t[AVERAGING TIME {counter}] rebounds: {n_rebound} - infections: {n_infections} (beta: {beta_value}, alpha: {alpha})")
                
                # 3. COMPUTE P_qs
                tot_time = sum(Pqs.values())
                for k in Pqs:
                    Pqs[k] = Pqs[k]/tot_time
                Pqs = dict(Pqs)
                if VERBOSE:
                    print(f"\ttot_time (av_time): {tot_time} ({av_time})")
                
                rebound_list.append(n_rebound)
                infectionEvents_list.append(n_infections)
                Pqs_list.append(Pqs)
                
                # 4. COMPUTE rho
                rho = 0
                rho_2 = 0
                for k in Pqs:
                    rho = rho + Pqs[k]*k
                    rho_2 = rho_2 + Pqs[k]*(k**2)
                rho = rho/n_nodes
                rho_2 = rho_2/(n_nodes**2)
                rho_list.append(rho)
                rho_2_list.append(rho_2)
                
                # 5. COMPUTE X and DELTA
                if (rho_2 - rho**2) <= 0:
                    if DEBUG:
                        print(f"[WARNING] (rho_2 - rho**2) == {(rho_2 - rho**2)} -- rho_2 is {rho_2} and rho is {rho}")
                    X = 0
                    DELTA = 0
                else:
                    DELTA = np.sqrt(rho_2 - rho**2) / rho 
                    X = n_nodes * (  (rho_2 - rho**2) / rho  )
                X_list.append(X)
                DELTA_list.append(DELTA)
                
                if VERBOSE:
                    print(f"X: {X} - DELTA: {DELTA} (beta: {beta_value}, alpha: {alpha})")
            toc = time.perf_counter()
            print(f"[STOP Process {pid}] Time for {iterations} simulations: {toc-tic:0.2f} sec - {(toc-tic)/60:0.2f} min - beta: {beta_value} - alpha: {alpha}")
            beta_dict[beta_value] = {}
            beta_dict[beta_value]['rebounds'] = rebound_list
            beta_dict[beta_value]['infections'] = infectionEvents_list
            beta_dict[beta_value]['Pqs'] = Pqs_list
            beta_dict[beta_value]['rho'] = rho_list
            beta_dict[beta_value]['rho^2'] = rho_2_list
            beta_dict[beta_value]['X'] = X_list
            beta_dict[beta_value]['DELTA'] = DELTA_list
            
        except queue.Empty:
            break    
    print(f"[END Process {pid}] Writing results")
    pickle.dump( beta_dict, open( f"{output_dir}/p{pid}_beta_dict.pickle", "wb" ) ) 

    

def single_run_RBC(queue_beta, pid, c, alpha, rlx_time, av_time, graph, output_dir):
    # For an ALPHA and a BETA we run iterations simulations
    v0 = c*alpha
    v1 = (1-alpha)*c
    n_nodes = graph.number_of_nodes()
    
    print(f"[START Process {pid}]")
    
    beta_dict = {}
    n_iterations = 0
    computed_beta = set()
    tic = time.perf_counter()
    while True:
        try:
            # get a value for BETA
            beta_value,_ = queue_beta.get(block=True, timeout=QUEUE_TIMEOUT)
            computed_beta.add(beta_value)
            n_iterations += 1
            H, J = get_H_J(v0, v1, gamma, lambda1, lambda2, beta_value, sigma1, sigma2)
            # Set-up Network
            IC = defaultdict(lambda: 'I0')
            return_statuses = ('S0', 'I0', 'S1', 'I1', 'S2')
            initStatus = IC

            # 1. RELAXATION TIME 
            n_rebound = -1
            endTime = rlx_time
            startTime = 0
            while startTime<endTime:
                n_rebound += 1
                simInv = EoN.Gillespie_simple_contagion(graph, H, J, initStatus, return_statuses, return_full_data=True, tmin = startTime, tmax = endTime)
                startTime, initStatus, initInfected = apply_reflecting_boundary(simInv, endTime, n_nodes)
                    
            # 2. AVERAGING TIME
            n_rebound = -1
            n_infections = 0
            endTime = av_time
            startTime = 0
            Pqs = defaultdict(lambda: 0)
            while startTime<endTime:
                n_rebound += 1
                simInv = EoN.Gillespie_simple_contagion(graph, H, J, initStatus, return_statuses, return_full_data=True, tmin = startTime, tmax = endTime)
                startTime, initStatus, n_infections, initInfected = update_Pqs(simInv, Pqs, startTime, endTime, n_nodes, n_infections, initInfected)
                
            # 3. COMPUTE P_qs
            tot_time = sum(Pqs.values())
            for k in Pqs:
                Pqs[k] = Pqs[k]/tot_time
            Pqs = dict(Pqs)
                
            # 4. COMPUTE rho
            rho = 0
            rho_2 = 0
            for k in Pqs:
                rho = rho + Pqs[k]*k
                rho_2 = rho_2 + Pqs[k]*(k**2)
            rho = rho/n_nodes
            rho_2 = rho_2/(n_nodes**2)
                
            # 5. COMPUTE X and DELTA
            if (rho_2 - rho**2) <= 0:
                if DEBUG:
                    print(f"[WARNING] (rho_2 - rho**2) == {(rho_2 - rho**2)} -- rho_2 is {rho_2} and rho is {rho}")
                DELTA = 0
                X = 0
            else:
                DELTA = np.sqrt(rho_2 - rho**2) / rho 
                X = n_nodes * (  (rho_2 - rho**2) / rho  )
            
            if beta_value not in beta_dict:
                beta_dict[beta_value] = {}
                beta_dict[beta_value]['rebounds'] = [n_rebound]
                beta_dict[beta_value]['infections'] = [n_infections]
                beta_dict[beta_value]['Pqs'] = [Pqs]
                beta_dict[beta_value]['rho'] = [rho]
                beta_dict[beta_value]['rho^2'] = [rho_2]
                beta_dict[beta_value]['X'] = [X]
                beta_dict[beta_value]['DELTA'] = [DELTA]
            else:
                beta_dict[beta_value]['rebounds'].append(n_rebound)
                beta_dict[beta_value]['infections'].append(n_infections)
                beta_dict[beta_value]['Pqs'].append(Pqs)
                beta_dict[beta_value]['rho'].append(rho)
                beta_dict[beta_value]['rho^2'].append(rho_2)
                beta_dict[beta_value]['X'].append(X)
                beta_dict[beta_value]['DELTA'].append(DELTA)
            
        except queue.Empty:
            toc = time.perf_counter()
            break    
    secondi = toc-tic
    minuti = secondi/60
    ore = minuti/60
    print(f"[END Process {pid}] iterations: {n_iterations} ({secondi:0.2f} sec) ({minuti:0.2f} min) ({ore:0.2f} ore)")
    print(f"[END Process {pid}] beta: {computed_beta} - alpha: {alpha}")
    pickle.dump( beta_dict, open( f"{output_dir}/p{pid}_beta_dict.pickle", "wb" ) ) 
    
    
    
    
    
    
    
    
    
    
def get_total_activity_time(simInv, initStatus, startTime, lastTime, nNodes):
    nodesActiveTime = [0]*nNodes
    for j in range(nNodes): 
        # For each node compute the active time
        timelist, statuslist = simInv.node_history(j)
        
        prevStatus = initStatus[j]
        if prevStatus in ['I0', 'I1']:
            lastInfectionTime = startTime
        else:
            lastInfectionTime = None
        
        totActiveTime = 0
        for i,s in enumerate(timelist):
            if i == 0:
                continue
            if statuslist[i] in ['I0', 'I1']: # the node is infected
                if prevStatus in ['I0', 'I1']:
                    print(f"[ERROR] current status: {statuslist[i]}({s}) - previous status: {prevStatus}({lastInfectionTime})")
                    exit()
                lastInfectionTime = s
            else: # the node is NOT infected
                if prevStatus in ['I0', 'I1']: # Check if the node just recovered (it was active)
                    if s >= lastTime:
                        totActiveTime += lastTime - lastInfectionTime
                    else:
                        totActiveTime += s - lastInfectionTime
                lastInfectionTime = None
            prevStatus = statuslist[i]
            if s >= lastTime:
                break
            
        nodesActiveTime[j] = totActiveTime
        if nodesActiveTime[j] > lastTime-startTime:
            print(f"[ERROR] active time: {nodesActiveTime[j]} - total time: {lastTime-startTime}")
            exit()
    return np.array(nodesActiveTime)    


def update_Pqs_RTA( data, Pqs, startTime, maxTime, nNodes, n_infections, initInfected, initStatus):
    # 1. count infection events
    for j in range(nNodes):
        _, statusHistory = data.node_history(j)
        for h,k in enumerate(statusHistory):
            # h >= 1, to skip the initial status
            if h>=1 and k in ['I0', 'I1']:
                n_infections += 1
            
    # 2. check absorbing state and update P_qs       
    timeLine, statusDict = data.summary()
    infected = statusDict['I1']+statusDict['I0']
    lastTime = startTime
    lastInfected = initInfected
    for i,s in enumerate(timeLine):
        # We know there is at least 1 infected node in the Initiale State.
        # Thus (infected[0] != 0) it is always true and we always set lastTime=s
        if infected[i] == 0: 
            # We got 0 infected at time s
            # We compute the active time for each node
            nodesActiveTime = get_total_activity_time(data
                                                      , initStatus, startTime, s, nNodes)
            totInfected = 0
            for j in range(nNodes):
                initStatus[j] = data.node_status(j, s)
                if initStatus[j] in ['I0', 'I1']:
                    totInfected += 1
            if totInfected != 0:
                print(f"[ERROR] Last Time number of infected nodes is: {totInfected}")
                exit()
            # We reactivate Na nodes choosen based on their activity time
            Na = sum(nodesActiveTime)/(s-startTime)
            # With probability  
            #  *  Na - np.floor(Na)   we reactivate    np.ceil(Na)
            #  *  np.ceil(Na) - Na    we reactivate    np.floor(Na)
            if np.random.uniform() <= np.ceil(Na)-Na:
                Na = np.floor(Na)
            else:
                Na = np.ceil(Na)
            Na = int(Na)
            
            nodesToActivate = np.random.choice(nNodes, Na, replace=False, p=nodesActiveTime/sum(nodesActiveTime))
            for k in nodesToActivate:
                if initStatus[k] == 'S0':
                    initStatus[k] = 'I0'
                elif initStatus[k] in ['S1','S2']:
                    initStatus[k] = 'I1'
                else:
                    print(f"[ERROR] initStatus[k]: {initStatus[k]}")
                    exit()
            return s, initStatus, n_infections, Na
        Pqs[lastInfected] = Pqs[lastInfected] + (s-lastTime)
        lastTime = s
        lastInfected = infected[i]
    Pqs[lastInfected] = Pqs[lastInfected] + (maxTime-lastTime)
    
    for j in range(nNodes):
        initStatus[j] = data.node_status(j, maxTime)
        if initStatus[j] in ['I0', 'I1']:
            initInfected += 1
    if initInfected == 0:
        print(f"\t[ERROR] There are 0 infected nodes!!!")
        exit()
    return maxTime, initStatus, n_infections, initInfected

 

def apply_reactivation_per_activity_time(simInv, initStatus, startTime, maxTime, nNodes): 
    # 1. check absorbing state (relaxation phase)
    timeLine, statusDict = simInv.summary()
    infected = statusDict['I1']+statusDict['I0']
    initInfected = 0
    for i,s in enumerate(timeLine):
        # We know there is at least 1 infected node in the Initiale State.
        # Thus (infected[0] != 0) it is always True 
        if infected[i] == 0: 
            # We got 0 infected at time s
            # We compute the active time for each node
            nodesActiveTime = get_total_activity_time(simInv, initStatus, startTime, s, nNodes)
            totInfected = 0
            for j in range(nNodes):
                initStatus[j] = simInv.node_status(j, s)
                if initStatus[j] in ['I0', 'I1']:
                    totInfected += 1
            if totInfected != 0:
                print(f"[ERROR] Last Time number of infected nodes is: {totInfected}")
                exit()
            # We reactivate Na nodes choosen based on their activity time
            Na = sum(nodesActiveTime)/(s-startTime)
            # With probability  
            #    Na - np.floor(Na)   we reactivate    np.ceil(Na)
            #    np.ceil(Na) - Na    we reactivate    np.floor(Na)
            if np.random.uniform() <= np.ceil(Na)-Na:
                Na = np.floor(Na)
            else:
                Na = np.ceil(Na)
            Na = int(Na)
            
            nodesToActivate = np.random.choice(nNodes, Na, replace=False, p=nodesActiveTime/sum(nodesActiveTime))
            for k in nodesToActivate:
                if initStatus[k] == 'S0':
                    initStatus[k] = 'I0'
                elif initStatus[k] in ['S1','S2']:
                    initStatus[k] = 'I1'
                else:
                    print(f"[ERROR] initStatus[k]: {initStatus[k]}")
                    exit()
            return s, initStatus, Na

        
    # 2. create the initial status for the averaging phase
    if DEBUG:
        print(f"\t[DEBUG] We start the AVERAGING PHASE with {infected[-1]} infected nodes.")
    for j in range(nNodes):
        initStatus[j] = simInv.node_status(j, maxTime)
        if initStatus[j] in ['I0', 'I1']:
            initInfected += 1
    if initInfected == 0:
        print(f"\t[ERROR] There are 0 infected nodes!!!")
        exit()
    return maxTime, initStatus, initInfected


#    
# Reactivation per Activity Time
#
def multi_run_RTA(queue_beta, pid, c, alpha, iterations, rlx_time, av_time, graph, output_dir):
    # For an ALPHA and a BETA we run iterations simulations
    v0 = c*alpha
    v1 = (1-alpha)*c
    n_nodes = graph.number_of_nodes()
    beta_dict = {}
    print(f"[START Process {pid}]")
    
    while True:
        try:
            # get a value for BETA
            beta_value = queue_beta.get(block=True, timeout=QUEUE_TIMEOUT)
            print(f"[START Process {pid}] Alpha: {alpha}, Beta: {beta_value}")
            H, J = get_H_J(v0, v1, gamma, lambda1, lambda2, beta_value, sigma1, sigma2)
            rebound_list = []
            infectionEvents_list = []
            Pqs_list = []
            rho_list = []
            rho_2_list = []
            X_list = []
            DELTA_list = []
            
            tic = time.perf_counter()
            for counter in range(iterations): #For ALPHA, BETA
                # Set-up Network
                IC = defaultdict(lambda: 'I0')
                return_statuses = ('S0', 'I0', 'S1', 'I1', 'S2')
                initStatus = IC

                # 1. RELAXATION TIME 
                n_rebound = -1
                endTime = rlx_time
                startTime = 0
                while startTime<endTime:
                    n_rebound += 1
                    simInv = EoN.Gillespie_simple_contagion(graph, H, J, initStatus, return_statuses, return_full_data=True, tmin = startTime, tmax = endTime)
                    startTime, initStatus, initInfected = apply_reactivation_per_activity_time(simInv, initStatus, startTime, endTime, n_nodes)
                if VERBOSE:
                    print(f"\t[RELAXATION TIME {counter}] rebounds: {n_rebound} (beta: {beta_value}, alpha: {alpha})")
                
                # 2. AVERAGING TIME
                n_rebound = -1
                n_infections = 0
                endTime = av_time
                startTime = 0
                Pqs = defaultdict(lambda: 0)
                while startTime<endTime:
                    n_rebound += 1
                    simInv = EoN.Gillespie_simple_contagion(graph, H, J, initStatus, return_statuses, return_full_data=True, tmin = startTime, tmax = endTime)
                    startTime, initStatus, n_infections, initInfected = update_Pqs_RTA( simInv, Pqs, startTime, endTime, n_nodes, n_infections, initInfected, initStatus)
                if VERBOSE:
                    print(f"\t[AVERAGING TIME {counter}] rebounds: {n_rebound} - infections: {n_infections} (beta: {beta_value}, alpha: {alpha})")
                
                # 3. COMPUTE P_qs
                tot_time = sum(Pqs.values())
                for k in Pqs:
                    Pqs[k] = Pqs[k]/tot_time
                Pqs = dict(Pqs)
                if VERBOSE:
                    print(f"\ttot_time (av_time): {tot_time} ({av_time})")
                
                rebound_list.append(n_rebound)
                infectionEvents_list.append(n_infections)
                Pqs_list.append(Pqs)
                
                # 4. COMPUTE rho
                rho = 0
                rho_2 = 0
                for k in Pqs:
                    rho = rho + Pqs[k]*k
                    rho_2 = rho_2 + Pqs[k]*(k**2)
                rho = rho/n_nodes
                rho_2 = rho_2/(n_nodes**2)
                rho_list.append(rho)
                rho_2_list.append(rho_2)
                
                # 5. COMPUTE X and DELTA
                if (rho_2 - rho**2) <= 0:
                    if DEBUG:
                        print(f"[WARNING] (rho_2 - rho**2) == {(rho_2 - rho**2)} -- rho_2 is {rho_2} and rho is {rho}")
                    X = 0
                    DELTA = 0
                else:
                    X = n_nodes * (  (rho_2 - rho**2) / rho  )
                    DELTA = np.sqrt(rho_2 - rho**2) / rho 
                X_list.append(X)
                DELTA_list.append(DELTA)
                
                if VERBOSE:
                    print(f"X: {X} - DELTA: {DELTA} (beta: {beta_value}, alpha: {alpha})")
            toc = time.perf_counter()
            print(f"[STOP Process {pid}] Time for {iterations} simulations: {toc-tic:0.2f} sec - {(toc-tic)/60:0.2f} min - beta: {beta_value} - alpha: {alpha}")
            beta_dict[beta_value] = {}
            beta_dict[beta_value]['rebounds'] = rebound_list
            beta_dict[beta_value]['infections'] = infectionEvents_list
            beta_dict[beta_value]['Pqs'] = Pqs_list
            beta_dict[beta_value]['rho'] = rho_list
            beta_dict[beta_value]['rho^2'] = rho_2_list
            beta_dict[beta_value]['X'] = X_list
            beta_dict[beta_value]['DELTA'] = DELTA_list
            
        except queue.Empty:
            break    
    print(f"[END Process {pid}] Writing results")
    pickle.dump( beta_dict, open( f"{output_dir}/p{pid}_beta_dict.pickle", "wb" ) )     

    
def single_run_RTA(queue_beta, pid, c, alpha, rlx_time, av_time, graph, output_dir):
    # For an ALPHA and a BETA we run iterations simulations
    v0 = c*alpha
    v1 = (1-alpha)*c
    n_nodes = graph.number_of_nodes()
    print(f"[START Process {pid}]")

    beta_dict = {}
    n_iterations = 0
    computed_beta = set()
    tic = time.perf_counter()
    while True:
        try:
            # get a value for BETA
            beta_value, itr  = queue_beta.get(block=True, timeout=QUEUE_TIMEOUT)
            computed_beta.add(beta_value)
            n_iterations += 1
            H, J = get_H_J(v0, v1, gamma, lambda1, lambda2, beta_value, sigma1, sigma2)
            # Set-up Network
            IC = defaultdict(lambda: 'I0')
            return_statuses = ('S0', 'I0', 'S1', 'I1', 'S2')
            initStatus = IC

            # 1. RELAXATION TIME 
            n_rebound = -1
            endTime = rlx_time
            startTime = 0
            while startTime<endTime:
                n_rebound += 1
                simInv = EoN.Gillespie_simple_contagion(graph, H, J, initStatus, return_statuses, return_full_data=True, tmin = startTime, tmax = endTime)
                startTime, initStatus, initInfected = apply_reactivation_per_activity_time(simInv, initStatus, startTime, endTime, n_nodes)
                
            # 2. AVERAGING TIME
            n_rebound = -1
            n_infections = 0
            endTime = av_time
            startTime = 0
            Pqs = defaultdict(lambda: 0)
            while startTime<endTime:
                n_rebound += 1
                simInv = EoN.Gillespie_simple_contagion(graph, H, J, initStatus, return_statuses, return_full_data=True, tmin = startTime, tmax = endTime)
                startTime, initStatus, n_infections, initInfected = update_Pqs_RTA( simInv, Pqs, startTime, endTime, n_nodes, n_infections, initInfected, initStatus)
                
            # 3. COMPUTE P_qs
            tot_time = sum(Pqs.values())
            for k in Pqs:
                Pqs[k] = Pqs[k]/tot_time
            Pqs = dict(Pqs)
                
            # 4. COMPUTE rho
            rho = 0
            rho_2 = 0
            for k in Pqs:
                rho = rho + Pqs[k]*k
                rho_2 = rho_2 + Pqs[k]*(k**2)
            rho = rho/n_nodes
            rho_2 = rho_2/(n_nodes**2)
                
            # 5. COMPUTE X and DELTA
            if (rho_2 - rho**2) <= 0:
                if DEBUG:
                    print(f"[WARNING] (rho_2 - rho**2) == {(rho_2 - rho**2)} -- rho_2 is {rho_2} and rho is {rho}")
                DELTA = 0
                X = 0
            else:
                DELTA = np.sqrt(rho_2 - rho**2) / rho 
                X = n_nodes * (  (rho_2 - rho**2) / rho  )
                
            if beta_value not in beta_dict:
                beta_dict[beta_value] = {}
                beta_dict[beta_value]['rebounds'] = [n_rebound]
                beta_dict[beta_value]['infections'] = [n_infections]
                beta_dict[beta_value]['Pqs'] = [Pqs]
                beta_dict[beta_value]['rho'] = [rho]
                beta_dict[beta_value]['rho^2'] = [rho_2]
                beta_dict[beta_value]['X'] = [X]
                beta_dict[beta_value]['DELTA'] = [DELTA]
            else:
                beta_dict[beta_value]['rebounds'].append(n_rebound)
                beta_dict[beta_value]['infections'].append(n_infections)
                beta_dict[beta_value]['Pqs'].append(Pqs)
                beta_dict[beta_value]['rho'].append(rho)
                beta_dict[beta_value]['rho^2'].append(rho_2)
                beta_dict[beta_value]['X'].append(X)
                beta_dict[beta_value]['DELTA'].append(DELTA)
            
        except queue.Empty:
            toc = time.perf_counter()
            break    
    secondi = toc-tic
    minuti = secondi/60
    ore = minuti/60
    print(f"[END Process {pid}] iterations: {n_iterations} ({secondi:0.2f} sec) ({minuti:0.2f} min) ({ore:0.2f} ore)")
    print(f"[END Process {pid}] beta: {computed_beta} - alpha: {alpha}")    
    pickle.dump( beta_dict, open( f"{output_dir}/p{pid}_beta_dict.pickle", "wb" ) )       

    
    
    
    
    
    
    
    
    
def get_H_J(v0, v1, gamma, lambda1, lambda2, beta, sigma1, sigma2):
    H = nx.DiGraph()  #DiGraph showing possible transitions that don't require an interaction
    H.add_edge('S0' , 'S1', rate = v0)   # S0  -> S1
    H.add_edge('S1' , 'S2', rate = v1)   # S1  -> S2
    H.add_edge('I0' , 'S1', rate = gamma)   # I0  -> S1
    H.add_edge('I1' , 'S2', rate = gamma)   # I1  -> S2
    H.add_edge('S1' , 'S0', rate = lambda1)   # S1 -> S0
    H.add_edge('S2' , 'S0', rate = lambda2)   # S2 -> S0
    J = nx.DiGraph()    #DiGraph showing transition that does require an interaction.
    J.add_edge(('I0', 'S0'), ('I0', 'I0'), rate = beta)         
    J.add_edge(('I1', 'S0'), ('I1', 'I0'), rate = beta)        
    J.add_edge(('I0', 'S1'), ('I0', 'I1'), rate = beta*sigma1)  
    J.add_edge(('I1', 'S1'), ('I1', 'I1'), rate = beta*sigma1)  
    J.add_edge(('I0', 'S2'), ('I0', 'I1'), rate = beta*sigma2)  
    J.add_edge(('I1', 'S2'), ('I1', 'I1'), rate = beta*sigma2)  
    return H, J



def gen_s1_graph( graph_type, N, beta, gamma, mean_deg, output_dir ):
    print(f"Parameters:")
    print(f" - beta: {beta}")
    print(f" - gamma: {gamma}")
    print(f" - mean deg: {mean_deg}")
    G = nx.geometric_soft_configuration_graph(beta=beta, n=N, gamma=gamma, mean_degree=mean_deg)
    pickle.dump( G, open( f"{output_dir}/{graph_type}_graph.pickle", "wb" ) )
    return G, graph_type
    
    
    
def gen_random_graph( graph_type, N, output_dir ):
    edge_p = math.log(N)/N
    n_edges = N*(N-1)/2 
    expected_mean_deg = 2*edge_p*n_edges/N
    G = nx.fast_gnp_random_graph(N, edge_p )
    while not nx.is_connected(G):
        print("The GRAPH is not CONNECTED!!!")
        G = nx.fast_gnp_random_graph(N, edge_p )
    pickle.dump( G, open( f"{output_dir}/{graph_type}_graph.pickle", "wb" ) )
    print(f"expected mean deg: {expected_mean_deg}")
    return G, graph_type


def read_graph_from_file( file_name, start_from_one, output_dir):
    G = nx.Graph()
    with open(file_name,"r") as f:
        for line in f:
            if '%' in line or '#' in line:
                continue
            values = line.split()
            v1 = values[0]
            v2 = values[1]
            if start_from_one:
                G.add_edge(int(v1)-1, int(v2)-1)
            else:
                G.add_edge(int(v1), int(v2))   
    g_name = file_name.split("/")[-1].split(".")[0]
    pickle.dump( G, open( f"{output_dir}/{g_name}_graph.pickle", "wb" ) )
    return G, g_name


    
def run_new(alpha_list, beta_list, c, output_dir, g_name, mode, G):
    for alpha in alpha_list:
        q_beta = Queue()
        beta_list.sort(reverse=True)
        beta_dict = {}
        for beta_value in beta_list:
            beta_dict[beta_value] = {}
            beta_dict[beta_value]['rebounds'] = []
            beta_dict[beta_value]['infections'] = []
            beta_dict[beta_value]['Pqs'] = []
            beta_dict[beta_value]['rho'] = []
            beta_dict[beta_value]['rho^2'] = []
            beta_dict[beta_value]['X'] = []
            beta_dict[beta_value]['DELTA'] = []
            for i in range(ITERATIONS):
                try:
                    q_beta.put((beta_value,i))
                except queue.Full:
                    sys.exit("[ERROR] The queue is Full!")
        p_pool = []    
        for j in range(POOL_SIZE):
            if mode == 'rbc':
                p = Process(target=single_run_RBC, args=(q_beta, j, c, alpha, RLX_TIME, AV_TIME, G, output_dir))
            elif mode == 'rta':
                p = Process(target=single_run_RTA, args=(q_beta, j, c, alpha, RLX_TIME, AV_TIME, G, output_dir))
            p.start()
            p_pool.append(p)
        for j,p in enumerate(p_pool):
            p.join()
            print(f"Start collecting results of process {j}")
            p_beta_dict = pickle.load( open( f"{output_dir}/p{j}_beta_dict.pickle", "rb" ) )
            for k in p_beta_dict:
                beta_dict[k]['rebounds'].extend(p_beta_dict[k]['rebounds'])
                beta_dict[k]['infections'].extend(p_beta_dict[k]['infections'])
                beta_dict[k]['Pqs'].extend(p_beta_dict[k]['Pqs'])
                beta_dict[k]['rho'].extend(p_beta_dict[k]['rho'])
                beta_dict[k]['rho^2'].extend(p_beta_dict[k]['rho^2'])
                beta_dict[k]['X'].extend(p_beta_dict[k]['X'])
                beta_dict[k]['DELTA'].extend(p_beta_dict[k]['DELTA'])                
            os.remove(f"{output_dir}/p{j}_beta_dict.pickle")
        data = {}
        data['alpha'] = alpha
        data['beta_dict'] = beta_dict
        pickle.dump( data, open( f"{output_dir}/{g_name}_data_alpha{alpha}.pickle", "wb" ) )  
    print(f"Done!")
        
    

    
    
    
def main():
    parser=argparse.ArgumentParser(description="products")
    parser.add_argument('-t', '--graph-type', dest='graph_type', required=True, type=str, help='[random, pickle, s1, file]')
    parser.add_argument('-f', '--edge-list-file', dest='edge_list_file',default=None, required=False, type=str, help='The edge list of the graph. It works with the [file] option.')
    parser.add_argument('-s', '--start-from-one', dest='start_from_one', action='store_true', default=False, required=False, help='Specify this option if the node IDs in the Edge List file start from 1 instead of 0. It works with the [file] option.')
    parser.add_argument('-p', '--pickle-file', dest='pickle_file', required=False, help='The Networkx graph stored in a pickle file. It works with the [pickle] option.')
    parser.add_argument('-m', '--mode', dest='mode', required=True, type=str, help='[rbc, rta]')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True, type=str, help='Directory in which results will be written')
    
    args = parser.parse_args()
    graph_type = args.graph_type
    edge_list_file = args.edge_list_file
    start_from_one = args.start_from_one
    pickle_file = args.pickle_file
    mode = args.mode
    output_dir = args.output_dir
    
    print(f"start_from_one: {start_from_one}")
        
    try:  
        os.mkdir(output_dir)  
    except OSError as error:  
        print(error)
    
    G = None
    if graph_type == "random":
        G, g_name = gen_random_graph( graph_type, N_NODES, output_dir )
    elif graph_type == "pickle":
        G = pickle.load( open( pickle_file, "rb" ) )
        g_name = "pickle"
    elif graph_type == "s1":
        BETA=2
        GAMMA=2.5
        MEAN_DEG=10
        G, g_name = gen_s1_graph( graph_type, N_NODES, BETA, GAMMA, MEAN_DEG, output_dir )
    elif graph_type == "file" and edge_list_file != None:
        G, g_name = read_graph_from_file(edge_list_file, start_from_one, output_dir)
    else:
        print("Check the command line!")
        return
 
    print(f"nodes: {G.number_of_nodes()}")
    print(f"edges: {G.number_of_edges()}")
    print(f"max degree: {max([d for n, d in G.degree()])}")
    meanDeg = sum([deg for node, deg in list(G.degree)])/G.number_of_nodes() 
    print(f"mean degree: {meanDeg}")
    varDeg = sum([ (meanDeg-deg)**2 for node, deg in list(G.degree)])/G.number_of_nodes() 
    print(f"variance degree: {varDeg}")
        
    # Compute parameters c, alpha and beta
    w = (1-sigma1)/(1-sigma2)        
    c = get_c(w, lambda1, lambda2)
    c += 0.05429
    alpha_star = get_alpha(c, w, lambda1, lambda2)
    beta_c = get_beat_c(c, alpha_star, lambda1, lambda2, sigma1, sigma2)
    print(f"\nc: {c}")
    print(f"alpha_star: {alpha_star}")
    print(f"beta_c: {beta_c}")
    meanDeg = sum([deg for node, deg in list(G.degree)])/G.number_of_nodes() 
    print(f"meanDeg: {meanDeg}")
    
    beta_c_alpha_0 = get_beat_c(c, 0, lambda1, lambda2, sigma1, sigma2)
    beta_c_alpha_1 = get_beat_c(c, 1, lambda1, lambda2, sigma1, sigma2)
    print(f"\t expected beta_c (alpha_star): {round(beta_c/meanDeg,4)}")
    print(f"\t expected beta_c (alpha_0): {round(beta_c_alpha_0/meanDeg,4)}")
    print(f"\t expected beta_c (alpha_1): {round(beta_c_alpha_1/meanDeg,4)}\n")

    ### S^1
    #beta_start = 0.01
    #beta_list = []
    #for i in range(30): #50
    #    beta_start = round(beta_start+0.001, 3)
    #    beta_list.append(beta_start)
    ### ENRON
    beta_start = 0.003
    beta_list = []
    for i in range(30): #50
        beta_start = round(beta_start+0.001, 3)
        beta_list.append(beta_start)
    print(f"beta_list: {beta_list}")
    print(f"\tlen: {len(beta_list)} - min: {beta_list[0]} - max: {beta_list[-1]}\n")    

    alpha_val = 0.0
    alpha_list = []
    while alpha_val <= 1:
        alpha_list.append(alpha_val)
        alpha_val = round(alpha_val+0.2, 1)
    print(f"alpha_list: {alpha_list}")
    print(f"\tlen: {len(alpha_list)} - min: {alpha_list[0]} - max: {alpha_list[-1]}\n")

    pickle.dump( alpha_list, open( f"{output_dir}/{g_name}_alpha_list.pickle", "wb" ) )   
    pickle.dump( beta_list, open( f"{output_dir}/{g_name}_beta_list.pickle", "wb" ) )   
    pickle.dump( [c, alpha_star, beta_c, G.number_of_nodes(), ITERATIONS, RLX_TIME, AV_TIME], open( f"{output_dir}/{g_name}_values.pickle", "wb" ) )   
     
    run_new(alpha_list, beta_list, c, output_dir, g_name, mode, G)

    
    
if __name__ == "__main__":
    tic = time.perf_counter()
    main()   
    toc = time.perf_counter()
    totTime = toc-tic
    print(f"Simulation time: {totTime:0.2f} s ({totTime/60:0.2f} m) ({totTime/(60*60):0.2f} h)")
