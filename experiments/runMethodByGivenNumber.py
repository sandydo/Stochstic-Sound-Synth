
# The Monte Carlo Simulation
def generateTrigonometricTrajectoriesMonteCarlo(parameter, variables, shared_result_dict):
    '''
    returns two vectors.
    '''
    # The decomposition of the stiffness matrix
    Q, sigma, QT = variables['alphaOperDecomp']
    
    # Time variables
    timeDiscr = parameter[2]
    Tmax = parameter[0]
    k = Tmax/timeDiscr
    
    # Space variables.
    spaceDiscr = parameter[1]
    
    # The covariance Operator
    eigV, eigF = variables['convOp']
    
    # The noise seed
    noiseSeed = parameter[3]
    
    ## The random field ##
    randomField = GenBrownianMotion(eigV,
                                    eigF,
                                    timeDiscr,
                                    spaceDiscr,
                                    spaceDiscr +1,
                                    Tmax,
                                    noiseSeed)
    
    
    
    # Starting with the initial condition
    u = variables['initVals'][0]
    v = variables['initVals'][1]
    
    
    # Some concrete functions we might need:
    abstrLamb = lambda Q, s, QT, func, vec : np.dot( np.dot(Q, np.dot(np.diag(func(s)), QT)) , vec)
    # Some operator on the spectrum we will use.
    sink = lambda t : lambda x : np.sin( t*x**(1./2) )
    cosk = lambda t : lambda x : np.cos( t*x**(1./2) )
    Lambda = lambda s:  lambda x : x**(s)

    ### Frequent operators

    # C_h(k)
    cosOp = np.dot(Q, np.dot(np.diag(cosk(k)(sigma)), QT))
    # S_h(k)
    sinOp = np.dot(Q, np.dot(np.diag(sink(k)(sigma)), QT))
    # \Lambda^{-1/2}
    LambdaNegPow = np.dot(Q, np.dot(np.diag(Lambda(-1. / 2)(sigma)), QT))
    # \Lambda^{1/2}
    LambdaPosPow = np.dot(Q, np.dot(np.diag(Lambda(1. / 2)(sigma)), QT))

    
    
    ### Let's iterate. ###
    for n in range(1,timeDiscr+1): # Not at time 0
        # Saving the temporal step
        uPast = u
        vPast = v
        u = np.dot(cosOp, uPast) + np.dot(LambdaNegPow, np.dot(sinOp, vPast)) \
               + np.dot(LambdaNegPow, np.dot(sinOp, uPast)) * (randomField[n, :] - randomField[n - 1, :])

        v = - np.dot(LambdaPosPow, np.dot(sinOp, uPast)) + np.dot(cosOp, vPast) \
               + np.dot(cosOp, uPast )* (randomField[n, :] - randomField[n - 1, :])
        
    
    shared_result_dict[parameter] = (u,v)


# Some helping functions

# The Monte Carlo Simulation
def generateTrigoCohenAnsatz(parameter, variables, shared_result_dict):
    '''
    returns two vectors.
    '''
    # The decomposition of the stiffness matrix
    alpha = variables['alpha']
    
    # Time variables
    timeDiscr = parameter[2]
    Tmax = parameter[0]
    k = Tmax/timeDiscr
    
    # Space variables.
    spaceDiscr = parameter[1]
    
    # The covariance Operator
    eigV, eigF = variables['convOp']
    
    # The noise seed
    noiseSeed = parameter[3]
    
    ## The random field ##
    randomField = GenBrownianMotion(eigV,
                                    eigF,
                                    timeDiscr,
                                    spaceDiscr,
                                    spaceDiscr +1,
                                    Tmax,
                                    noiseSeed)
    
    
    
    # Starting with the initial condition
    u = variables['initVals'][0]
    v = variables['initVals'][1]
    

    ### Frequent operators
    sqrtOp = sp.linalg.sqrtm(alpha.toarray())
    # C_h(k)
    cosOp = sp.linalg.cosm(sqrtOp*k)
    # S_h(k)
    sinOp = sp.linalg.sinm(sqrtOp*k)
    # \Lambda^{-1/2}
    LambdaNegPow = sp.linalg.inv(sqrtOp)
    # \Lambda^{1/2}
    LambdaPosPow = sqrtOp

    
    
    ### Let's iterate. ###
    for n in range(1,timeDiscr+1): # Not at time 0
        # Saving the temporal step
        uPast = u
        vPast = v
        u = np.dot(cosOp, uPast) + np.dot(LambdaNegPow, np.dot(sinOp, vPast)) \
               + np.dot(LambdaNegPow, np.dot(sinOp, uPast)) * (randomField[n, :] - randomField[n - 1, :])

        v = - np.dot(LambdaPosPow, np.dot(sinOp, uPast)) + np.dot(cosOp, vPast) \
               + np.dot(cosOp, uPast )* (randomField[n, :] - randomField[n - 1, :])
        
    
    shared_result_dict[parameter] = (u,v)





# The Explicite Euler Method
def generateTrajectoriesCrankNicolsonMaruyama(parameter, variables, shared_result_dict):
    '''
    This is the Crank-Nicolson-Maruyama method to generate the endpoint in time of trajectories of 
	wave equation with multiplicative nose.
	1) The parameter is of the form
	parameter = (timeMax,
		spaceDiscr,
		timeDiscr,
		noiseSeed,
		noiseSmoothParam,
		initialCondByName,
		'LaplaceCovarianceOp')
	2) In variables we give the initial condition, the covariance operator by eigenvalues
	and eigenfunctions and the \alpha operator for the particular setting
	returns two vectors, which represent the trajectories at the end time point.
    '''

    # The alpha operator with the Lumped Mass matrix
    alpha = variables['alphaLumped']
    
    ## Time variables
    timeDiscr = parameter[2]
    Tmax = parameter[0]
    k = Tmax/timeDiscr
    
    ## Space variables.
    spaceDiscr = parameter[1]
    
    ## The covariance Operator
    eigV, eigF = variables['convOp']
    
    ## The noise seed
    noiseSeed = parameter[3]
    
    ## The random field ##
    randomField = GenBrownianMotion(eigV,
                                    eigF,
                                    timeDiscr,
                                    spaceDiscr,
                                    spaceDiscr +1,
                                    Tmax,
                                    noiseSeed)
    
    
    
    # Starting with the initial condition
    u = variables['initVals'][0]
    v = variables['initVals'][1]
 
    ### Let's iterate. ###
    for n in range(1,timeDiscr+1): # Not at time 0
        # Saving the temporal step
        uPast = u
        vPast = v
        
        # The noise term is.
        deltaW = (randomField[n, :] - randomField[n - 1, :])
        
        # RHS for the second component is:
        rhsVSolve = vPast - k* alpha.dot(uPast) - (k/2.)**2 * alpha.dot(vPast) + uPast*deltaW
        
        v = ssp.linalg.spsolve(ssp.identity(spaceDiscr) + (k/2.)**2 * alpha, rhsVSolve)
        
        u = uPast + (k/2.)*(v + vPast)
        
        
        
        
    
    
    shared_result_dict[parameter] = (u,v)

    



# The random setting
def GenBrownianMotion(eigV, eigF, M, N, J, timeMax,  seed=0):
    '''
    This generates a matrix randomField of shape (M+1)x(N) so that for j = 1,...,M
                randomField[j,:]
    is the values of a Wiener process due to the given covariance operator for the timestep timeMax*j/M.

    param: eigV: lambda function for the eigenvalues of the covariance operator.
    param: eigF: eigenfunctions of the covariance operator.
    param: M: Number of time discretizations.
    param: N: Number of spatial discretizations.
    param: J: Number of the truncations
    param: seed:
    '''
    randomField = np.zeros((M+1,N))
    if seed != 0:
        np.random.seed(seed=seed)

    # Let's generate J standard brownian motion trajectories on the time interval (0, timeMax)
    brownianMotions = np.zeros((J,M+1))
    for j in range(J):
        for i in range(M):
            # scale is the standard deviation.
            brownianMotions[j,i+1] = brownianMotions[j,i] + norm.rvs(scale=math.sqrt(timeMax/M*1.))

    # Let's generate the whole random field.
    # Time
    for m in range(M+1):
        # Space
        for n in range(N):
            # 'complexity'
            # We just sum over the noise values.
            tempValue = 0.
            for k in range(J):
                # The point in space
                x_n = (n+1)/(N+1)
                
                # We just sum over the noise values.
                tempValue = tempValue + math.sqrt(eigV(k+1)) * eigF(k+1)(x_n) * brownianMotions[k,m]
            
            
            randomField[m,n] = tempValue

    return randomField


# The stiffness matrix
def generateStiffness(N):
    # We will just give the stiffness matrix explicite for the particular example.
    # The source of this definition is Strong and Weak Approximation of Semilinear Stochastic Evolution Equations
    # From Raphael Kruse
    A = ssp.lil_matrix((N,N))
    h = 1./(N+1)
    
    # We give it just explicite

    for i in range(N):
        for j in range(N):
            if i == j:
                A[i,j] = 2/h
            elif abs(i-j) == 1:
                A[i,j] = -1/h
            else:
                pass
    A = A.tocsc()
    return A

def generateMassMatrix(N):
    # We will just give the mass matrix explicite for the particular example.
    # The source of this definition is Strong and Weak Approximation of Semilinear Stochastic Evolution Equations
    # From Raphael Kruse
    M = ssp.lil_matrix((N,N))
    h = 1./(N+1)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i,j] = 2*h/3
            elif abs(i-j) == 1:
                M[i,j] = h/6
            else:
                pass
    M = M.tocsc()
    return M

def generateSumLumpedMassMatrix(N):
    
    M= ssp.lil_matrix((N,N))
    h = 1./(N+1)
    
    for i in range(N):
        if i!=0 and i!=N-1:
            M[i,i] = h
        elif i==0 or i==N-1:
            M[i,i] = h*5/6
    
    M = M.tocsc()
    
    return M



def generateSumLumpedMassMatrixInv(N):
    
    M= ssp.lil_matrix((N,N))
    h = 1./(N+1)
    
    for i in range(N):
        if i!=0 and i!=N-1:
            M[i,i] = 1/h
        elif i==0 or i==N-1:
            M[i,i] = (h*5/6)**(-1)
    M = M.tocsc()
    return M

def numberToString(number, lenght):
    '''
    Number is the value given. Length denotes the length of the ouput string.
    we expect, that 
    '''
    nString  = str(number)
    if len(nString) < length:
        # Add zeros at the beginning
        for i in range(length - len(nString)):
            nString = '0' + nString
    else:
        # Cut off the end
        nString = nString[:length]

    return nString





def calculateTrajectories(parameter):
    '''
    Calculates trajectories with the given parameter setting.
    :param: parameter: just all parameters
    :returs: A dictionary with the given trajectories and the corresponding parameters
    '''
    # First we give an dictionary for the results
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    tempVariables = {}
    warming = None

    # Current parameter setting.


    ###########################
    ##### Space parameter #####
    ###########################
    for spaceDiscr in parameter['spaceDiscr']:
        
        
        ############################
        #### Noise Smoothness ######
        ############################
        for noiseSmoothParam in parameter['noiseSmoothParam']:
            #####################################
            ##### The covariance operator #######
            #####################################

            for covarianceOpByName in parameter['covarianceOpByName']:
                # The covariance operator, given by explice eigenvalues and eigenfunctions as lambda functions.
                if covarianceOpByName == 'LaplaceCovarianceOp':

                    ### Color for the noise. These are the eigenvalues and eigenfunctions of (-\Delta)^{-s}, where \Delta is the laplacian.
                    eigV = lambda s :lambda j : (j* math.pi)**(-s*2)
                    eigF = lambda j : lambda y: math.sqrt(2)*np.sin(j * math.pi * y)
                else:
                    raise NameError('No covariance operator given')

                tempVariables['convOp'] = (eigV(noiseSmoothParam), eigF)
                
                ###################################### 
                ##### The initial condition loop #####
                ######################################
                for initialCondByName in parameter['initialCondByName']:

                        #### The initial condition #####
                    if initialCondByName == 'eigenFunctionInitCondition':
                        # A very smooth init condition
                        h=1./(spaceDiscr + 1)
                        u_0 = np.sin(2 * math.pi * np.linspace(h,1-h, spaceDiscr))
                        v_0 = np.sin(3 * math.pi * np.linspace(h,1-h, spaceDiscr))
                    elif initialCondByName == 'unsmoothInitCond':
                        # A very unsmooth init condition
                        u_0 = np.concatenate((np.linspace(0, 5, np.floor((spaceDiscr) / 4.)), np.linspace(5, 0, np.floor((spaceDiscr) / 4)),
                                          np.zeros(int((spaceDiscr) - 2 * np.floor((spaceDiscr) / 4.)))))  # in H^1
                        # So this is just the weak derivative of u_0
                        v_0 = np.concatenate((np.full(int((spaceDiscr) - 2 * np.floor((spaceDiscr) / 4.)), 0),
                                                np.full(int(np.floor((spaceDiscr) / 4.)), 5. ),
                                                np.full( int(np.floor((spaceDiscr) / 4.)), 0)))  # in H^0
                    else:
                        raise NameError('No initial condition given')

                    tempVariables['initVals'] = [u_0, v_0]


                    #############################
                    ##### Time parameter ########
                    #############################
                    for timeDiscr in parameter['timeDiscr']:
                        #####################
                        ##### Time Max ######
                        #####################
                        for timeMax in parameter['timeMax']:
                            
                            # We prepare the last prameter before sending the Monte Carlo iterations:
                            targetFunc = None
                            # We assign the target function per algorithm.
                            if parameter['methodByName'] == 'trigonometricSVD':
                                # The trigonomtric method with sigular value decomposition.
                                targetFunc = generateTrigonometricTrajectoriesMonteCarloBasisTransform
                                # We take the alpha full:
                                # First the general alpha
                                A = generateStiffness(spaceDiscr)
                                MFull = generateMassMatrix(spaceDiscr)
                                MFullinv = ssp.linalg.inv(MFull)
                                alphaFull = MFullinv.dot(A)

                                tempVariables['alphaFull'] = alphaFull
                                # Now the docomposition of alphaFull
                                Q, sigma, QT = np.linalg.svd(alphaFull.toarray(), full_matrices=True)
                                tempVariables['alphaOperDecomp'] = (Q, sigma, QT)

                            elif parameter['methodByName'] == 'EigenValFullBasisTrafo':
                                targetFunc = generateTrigoEigenvalueFuncBasisTransform
                                A = generateStiffness(spaceDiscr)
                                MFull = generateMassMatrix(spaceDiscr)
                                MFullinv = ssp.linalg.inv(MFull)
                                
                                alphaFull = MFullinv.dot(A)
                                tempVariables['alphaFull'] = alphaFull
                                
                                # Now the docomposition of alphaFull
                                v, mat = np.linalg.eig(alphaFull.toarray())
                                tempVariables['eigenDecompo'] = v, mat
                                
                            elif parameter['methodByName'] == 'EigenValFull':
                                print('test')
                                targetFunc = generateTrigoEigenDecompo
                                A = generateStiffness(spaceDiscr)
                                MFull = generateMassMatrix(spaceDiscr)
                                MFullinv = ssp.linalg.inv(MFull)
                                
                                alphaFull = MFullinv.dot(A)
                                tempVariables['alphaFull'] = alphaFull
                                
                                # Now the docomposition of alphaFull
                                v, mat = np.linalg.eig(alphaFull.toarray())
                                tempVariables['eigenDecompo'] = v, mat
                                
                            elif parameter['methodByName'] == 'trigonemtricExp':
                                # The trigonometric method with build in exponent function.
                                targetFunc = generateTrajectoriesWithBuildInExp
                            elif parameter['methodByName'] == 'expliciteEuler':
                                # The explicite Euler Method.
                                targetFunc = generateTrajectoriesEulerMethod
                            elif parameter['methodByName'] == 'trigonometricSVDWithoutBT':
                                # The tridonometric Method with the SVD of alpha, but not basis transformation.
       
                                targetFunc = generateTrigonometricTrajectoriesMonteCarlo
                                # First the alpha.
                                A = generateStiffness(spaceDiscr)
                                MFull = generateMassMatrix(spaceDiscr)
                                MFullinv = ssp.linalg.inv(MFull)
                                alphaFull = MFullinv.dot(A)

                                tempVariables['alphaFull'] = alphaFull
                                
                                # Now the docomposition of alphaFull
                                Q, sigma, QT = np.linalg.svd(alphaFull.toarray(), full_matrices=True)
                                tempVariables['alphaOperDecomp'] = (Q, sigma, QT)
                                
                            elif parameter['methodByName'] == 'CrankNicolson':
                                # The Crank-Nicolson-Maruyama method.
                                targetFunc = generateTrajectoriesCrankNicolsonMaruyama
                                A = generateStiffness(spaceDiscr)
                                # Now the mass Lumped:
                                MLumpedInv = generateSumLumpedMassMatrixInv(spaceDiscr)
                                alphaLumped = MLumpedInv.dot(A)
                                
                                tempVariables['alphaLumped'] = alphaLumped
                            elif parameter['methodByName'] == 'CohensMethodFull':
                                # Cohens Ansatz with full alpha
                                targetFunc = generateTrigoCohenAnsatz
                                A = generateStiffness(spaceDiscr)
                                MFull = generateMassMatrix(spaceDiscr)
                                MFullinv = ssp.linalg.inv(MFull)
                                alphaFull = MFullinv.dot(A)

                                tempVariables['alpha'] = alphaFull
                                
                            elif parameter['methodByName'] == 'CohensMethodLumped':
                                # Cohens Ansatz with lumped alpha
                                targetFunc = generateTrigoCohenAnsatz
                                A = generateStiffness(spaceDiscr)
                                # Now the mass Lumped:
                                MLumpedInv = generateSumLumpedMassMatrixInv(spaceDiscr)
                                alphaLumped = MLumpedInv.dot(A)
                                tempVariables['alpha'] = alphaLumped
                                
                            
                                                          
                                
                            ############################
                            ### The noise seed loop ####
                            ############################
                            ## Wee send all the processes due to a noise seed in parralel,
                            ## and wait until they are done.
                            
                            currentProcesses = []
                            now = time.time()
                            for noiseSeed in parameter['noiseSeed']:
                                
                                ### We save the current parameter ####
                                tempParameter = (timeMax,
                                                 spaceDiscr,
                                                 timeDiscr,
                                                 noiseSeed,
                                                 noiseSmoothParam,
                                                 initialCondByName,
                                                 'LaplaceCovarianceOp')


                                # And we start the process
                                process = multiprocessing.Process(target=targetFunc, 
                                                                  args=(tempParameter, tempVariables, result_dict))
                                process.start()

                                currentProcesses.append(process)
                                
                                if len(currentProcesses) >= multiprocessing.cpu_count() or noiseSeed == parameter['noiseSeed'][-1]:
                                    # Lets wait untill all processes are done:
                                    for process in currentProcesses:
                                        process.join()
                                    # And empty it:
                                    currentProcesses = []
                            
                            

    # Ok, done.
    return dict(result_dict)









if __name__ == '__main__':
    import sys
    
    from scipy.stats import norm
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import math
    import time
    import scipy.sparse as ssp

    from mpl_toolkits.mplot3d import axes3d

    import multiprocessing
    import copy
    import pickle
    
    
    setting = sys.argv[1]
    timeAnalysis = {}
    if setting == '0':
        parameter = {'timeMax' : [1.],
                 'spaceDiscr' : 2**(np.array([2])),
                 # 'spaceDiscr' : 2**(np.array([2,3])),
                 'timeDiscr' : [2**9],
                 'noiseSeed' : range(1,2501),
                 'noiseSmoothParam': [1, 1/2, 1/3, 1/4],
                 'initialCondByName': ['eigenFunctionInitCondition'],
                 'covarianceOpByName': ['LaplaceCovarianceOp'],
                 'methodByName': 'CrankNicolson',
                 'fileName': 'testRun_traj.p'}


        now = time.time()
        result = calculateTrajectories(parameter)
        print(len(result))
        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'space')] = time.time() - now
        print('Claclulating DryRun took ' + str(time.time() - now) + ' s')
        
        

        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/DryRun.p', 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))
    

    elif setting == '1':
        # Crank-Nicolosion space convergence without referenc traj
        parameter = {'timeMax' : [1.],
                     'spaceDiscr' : 2**(np.array([2,3,4,5,6,7,8])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'timeDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/2, 1/3, 1/4],
                     'initialCondByName': ['eigenFunctionInitCondition', 'unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'CrankNicolson',
                     'fileName': 'Space_Convergence_Trajectories_CrankNicolson.p'}


        now = time.time()
        result = calculateTrajectories(parameter)
        print(len(result))
        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'space')] = time.time() - now
        print('Claclulating '+str(parameter)+ ' took ' + str(time.time() - now) + ' s')


        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))
    
    elif setting == '2':
        # Fancy Trig method without space convergence
        parameter = {'timeMax' : [1.],
                     'spaceDiscr' : 2**(np.array([2,3,4,5,6,7,8])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'timeDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/2, 1/3, 1/4],
                     'initialCondByName': ['eigenFunctionInitCondition', 'unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'trigonometricSVDWithoutBT',
                     'fileName': 'Space_Convergence_Trajectories_FancySVD.p'}

        timeAnalysis = {}
        now = time.time()
        result = calculateTrajectories(parameter)

        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'space')] = time.time() - now

        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))
    
    elif setting == '3':
        # Cohens Trig method without space convergence
        parameter = {'timeMax' : [1.],
                     'spaceDiscr' : 2**(np.array([2,3,4,5,6,7,8])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'timeDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/2, 1/3, 1/4],
                     'initialCondByName': ['eigenFunctionInitCondition', 'unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'CohensMethodLumped',
                     'fileName': 'Space_Convergence_Trajectories_CohensMeth.p'}

        timeAnalysis = {}
        now = time.time()
        result = calculateTrajectories(parameter)

        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'space')] = time.time() - now

        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))
    
    elif setting == '4':
        #Reference trajectories Crank_Nicolson.
        parameter = {'timeMax' : [1.],
                     'spaceDiscr' : 2**(np.array([9])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'timeDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/2, 1/3, 1/4],
                     'initialCondByName': ['eigenFunctionInitCondition', 'unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'CrankNicolson',
                     'fileName': 'ReferenceTraj_CrankNicolson.p'}


        now = time.time()
        result = calculateTrajectories(parameter)

        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'space')] = time.time() - now
        print('Claclulating '+str(parameter)+ ' took ' + str(time.time() - now) + ' s')


        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))
    
    elif setting == '5':
        # Reference trajectories for my fancy svd method.
        parameter = {'timeMax' : [1.],
                     'spaceDiscr' : 2**(np.array([9])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'timeDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/2, 1/3, 1/4],
                     'initialCondByName': ['eigenFunctionInitCondition', 'unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'trigonometricSVDWithoutBT',
                     'fileName': 'ReferenceTraj_FandySVD.p'}

        timeAnalysis = {}
        now = time.time()
        result = calculateTrajectories(parameter)

        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'space')] = time.time() - now

        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))
    
    elif setting == '6':
        # Reference trajectories Cohen.
        parameter = {'timeMax' : [1.],
                     'spaceDiscr' : 2**(np.array([9])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'timeDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/2, 1/3, 1/4],
                     'initialCondByName': ['eigenFunctionInitCondition', 'unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'CohensMethodLumped',
                     'fileName': 'ReferenceTraj_CohensMeth.p'}

        timeAnalysis = {}
        now = time.time()
        result = calculateTrajectories(parameter)

        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'space')] = time.time() - now

        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))
    
    elif setting == '7':
        # Time convergence Crank Nicolson.
        parameter = {'timeMax' : [1.],
                     'timeDiscr' : 2**(np.array([2,3,4,5,6,7,8])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'spaceDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/4],
                     'initialCondByName': ['eigenFunctionInitCondition', 'unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'CrankNicolson',
                     'fileName': 'Time_Convergence_Trajectories_CrankNic.p'}


        now = time.time()
        result = calculateTrajectories(parameter)

        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'time')] = time.time() - now
        print('Claclulating '+str(parameter)+ ' took ' + str(time.time() - now) + ' s')


        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))
    
    elif setting == '8':
        # Time convergence Fancy SVD.
        parameter = {'timeMax' : [1.],
                     'timeDiscr' : 2**(np.array([2,3,4,5,6,7,8])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'spaceDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/4],
                     'initialCondByName': ['eigenFunctionInitCondition', 'unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'trigonometricSVDWithoutBT',
                     'fileName': 'Time_Convergence_Trajectories_FancySVD.p'}

        timeAnalysis = {}
        now = time.time()
        result = calculateTrajectories(parameter)

        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'time')] = time.time() - now

        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))

    elif setting == '9':
        # Time Convergence Cohen.
        parameter = {'timeMax' : [1.],
                     'timeDiscr' : 2**(np.array([2,3,4,5,6,7,8])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'spaceDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/4],
                     'initialCondByName': ['eigenFunctionInitCondition', 'unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'CohensMethodLumped',
                     'fileName': 'Time_Convergence_Trajectories_CohensMeth.p'}

        timeAnalysis = {}
        now = time.time()
        result = calculateTrajectories(parameter)

        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'time')] = time.time() - now

        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))
    
    
    
    elif setting == '10':
        parameter = {'timeMax' : [1.],
                     'timeDiscr' : 2**(np.array([2,3,4,5,6,7,8,9])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'spaceDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/2, 1/3, 1/4],
                     'initialCondByName': ['unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'CrankNicolson',
                     'fileName': 'simulationExperimentConvergenceRate_time_UnSmooth_Init_Cond_CrankNicolson.p'}


        now = time.time()
        result = calculateTrajectories(parameter)

        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'time')] = time.time() - now
        print('Claclulating '+str(parameter)+ ' took ' + str(time.time() - now) + ' s')


        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))
    
    elif setting == '11':
        parameter = {'timeMax' : [1.],
                     'timeDiscr' : 2**(np.array([2,3,4,5,6,7,8,9])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'spaceDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/2, 1/3, 1/4],
                     'initialCondByName': ['unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'trigonometricSVDWithoutBT',
                     'fileName': 'simulationExperimentConvergenceRate_time_UnSmooth_Init_Cond_TrigSVD.p'}

        timeAnalysis = {}
        now = time.time()
        result = calculateTrajectories(parameter)

        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'time')] = time.time() - now

        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))
        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))
    
    elif setting == '12':
        parameter = {'timeMax' : [1.],
                     'timeDiscr' : 2**(np.array([2,3,4,5,6,7,8,9])),
                     # 'spaceDiscr' : 2**(np.array([2,3])),
                     'spaceDiscr' : [2**9],
                     'noiseSeed' : range(1,2501),
                     'noiseSmoothParam': [1, 1/2, 1/3, 1/4],
                     'initialCondByName': ['unsmoothInitCond'],
                     'covarianceOpByName': ['LaplaceCovarianceOp'],
                     'methodByName': 'CohensMethodLumped',
                     'fileName': 'simulationExperimentConvergenceRate_time_UnSmooth_Init_Cond_CohenLumpedMeth.p'}

        timeAnalysis = {}
        now = time.time()
        result = calculateTrajectories(parameter)

        timeAnalysis[(parameter['methodByName'], parameter['initialCondByName'][0], 'time')] = time.time() - now

        pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/'+parameter['fileName'], 'wb'))

        pickle.dump(timeAnalysis, open('/home/thomas_schnake/Masterthesis_Simulation/timeAnalyser.p', 'wb'))








