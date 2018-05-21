

## Some Utilities:
def generateStiffness(N):
    # N is the number of discretisation where N+1 is the number of values on the interval
    A = np.zeros((N+1,N+1))
    h = 1./(N+1)

    # We give it just explicite

    for i in range(N+1):
        for j in range(N+1):
            if i == j:
                A[i,j] = 2/h
            elif abs(i-j) == 1:
                A[i,j] = -1/h
            else:
                pass
    return A

def generateRealBrownianMotionField(J, length, k, seed=-1):
    '''
    To generates J  realvalued brownian motion of length M with equidistant time steps of size k.
    We consider the initial value being 0.
    The result will be of shape, (J,length+1) where result[:,0] == 0
    '''

    # Look if we seed the process.
    if seed != -1:
        np.random.seed(seed=seed)


    brownianMotions = np.zeros((J,length + 1))
    # And let's generate.
    for i in range(length):
        for j in range(J):
            brownianMotions[j,i+1] = brownianMotions[j,i] + norm.rvs(scale=k)

    return brownianMotions

def generateColoredFunctValuedBrownianMotion(brownianMotions, eigV, eigF, spaceDiscret):
    '''
    We generate a colored function valued brownian motion, where the function is on (0,1). The color is given from the eigen value eigenV and
    eigen function eigenF which are lambda functions. We discretize the domain of the functions into an
    equidistant grid of grid size spaceDiscrete. If timeLength == brownianMotions.shape[1], then the result is of 
    shape (timeLength, spaceDiscret+1)
    '''

    timeLength = brownianMotions.shape[1]
    trunc = brownianMotions.shape[0]
    randomField = np.zeros((timeLength, spaceDiscret+1))

    # Let's generate the random field.
    # Time
    for i in range(timeLength):
        # Space
        for n in range(spaceDiscret+1):
            x = 1.*n/(spaceDiscret + 1)

            # 'complexity'
            tempValue = 0.
            for j in range(trunc):
                tempValue = tempValue + math.sqrt(eigV(j+1)) * eigF(j+1)(x) * brownianMotions[j,i]


            randomField[i,n] = tempValue

    return randomField



# The Monte Carlo Simulation
def generateTrigonometricTrajectoriesMonteCarlo(tempVariables, shared_result_dict, pos):
    '''
    returns two vectors.
    '''

    # The parameter.
    truncMax = tempVariables['truncMax']
    M = tempVariables['M']
    k = tempVariables['k']
    seed = tempVariables['seed']

    eigV = tempVariables['eigV']
    eigF = tempVariables['eigF']
    N = tempVariables['N']

    # Starting with the initial condition
    u = tempVariables['initVect1']
    v = tempVariables['initVect2']



    # 1) First we generate the random Field:
    # 1.1) First the brownian Motion:
    brownianMotion = generateRealBrownianMotionField( truncMax, M, k, seed)
    # 1.2) And the colored random field.
    randomField = generateColoredFunctValuedBrownianMotion( brownianMotion, eigV, eigF, N)

    # 2) We generate the stiffness matrix and the corresponding decomposition.
    A = generateStiffness(N)
    Q, sigma, QT = np.linalg.svd(A, full_matrices=True)



    # Some concrete functions we might need:
    abstrLamb = lambda C, s, D, func, vec : np.dot( np.dot(C, np.dot(np.diag(func(s)), D)) , vec)
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



    ### 3) Let's iterate.
    for n in range(1,M+1): # Not at time 0
        # Saving the temporal step
        uPast = u
        vPast = v
        u = np.dot(cosOp, uPast) + np.dot(LambdaNegPow, np.dot(sinOp, vPast)) \
               + np.dot(LambdaNegPow, np.dot(sinOp, uPast * (randomField[n, :] - randomField[n - 1, :])))

        v = - np.dot(LambdaPosPow, np.dot(sinOp, uPast)) + np.dot(cosOp, vPast) \
               + np.dot(cosOp, uPast * (randomField[n, :] - randomField[n - 1, :]))

    if seed % 100 == 0:
        print('check status: (hIter, kIter, smoothIter, seedIter)=' + str(pos))
    shared_result_dict[pos] = (u,v)

def runTrigonometricSimulation(parameter, options):
    '''
    returns a list two vecorts (u,v). The length of the list is len(parameter['MonteCarloIter'])
    '''
    # First we give an dictionary for the results
    manager = multiprocessing.Manager()
    result = manager.dict()

    # To fix some temporal variables 
    processes = []
    hSpan = parameter['hSpan']
    kSpan = parameter['kSpan']
    noiseSpan = parameter['noiseSpan']
    MonteCarloIter = parameter['MonteCarloIter']

    for hIter in range(len(hSpan)):
        # Space discretization:
        h = hSpan[hIter]
        N = int((1-h)/h)

        # The initial Condition:
        if options['InitSetting'] == 1:
            # A very unsmooth init condition
            initVect1 = np.concatenate((np.linspace(0, 5, np.floor((N+1) / 4.)), np.linspace(5, 0, np.floor((N+1) / 4)),
                              np.zeros(int((N+1) - 2 * np.floor((N+1) / 4.)))))  # in H^1
            # So this is just the weak derivative of u_0
            initVect2 = np.concatenate((np.full(int((N+1) - 2 * np.floor((N+1) / 4.)), 0),
                                    np.full(int(np.floor((N+1) / 4.)), 5. ),
                                    np.full( int(np.floor((N+1) / 4.)), 0)))  # in H^0
        elif options['InitSetting'] == 2:
            # A very smooth init condition
            initVect1 = np.sin(2 * math.pi * np.linspace(0,1, N+1))
            initVect2 = np.sin(3 * math.pi * np.linspace(0,1, N+1))

        elif options['InitSetting'] == 3:
            initVect1 = np.zeros(N+1)
            initVect2 = np.zeros(N+1)

        for kIter in range(len(kSpan)):
            # Time discretization:
            k = kSpan[kIter]
            timeMax = options['TimeMax']
            M = int(timeMax/k)


            for smoothIter in range(len(noiseSpan)):
                # Extremeness of noise. Let's fix the eigen function and eigen values.
                smooth = noiseSpan[smoothIter]
                eigVal = options['eigVal'](smooth)
                eigFunc = options['eigFunc']

                for seedIter in range(len(MonteCarloIter)):
                    # Monte Carlo simulation.
                    seed = MonteCarloIter[seedIter]

                    # Let's do the whole simulation:
                    # First need fix the rest of the varialbes:
                    trunc = N+1

                    tempVariables = {
                        'truncMax' : trunc,
                        'M': M,
                        'k': k,
                        'seed': seed,
                        'eigF': eigFunc,
                        'eigV': eigVal,
                        'N': N,
                        'initVect1': initVect1,
                        'initVect2': initVect2


                    }


                    p = multiprocessing.Process(target=generateTrigonometricTrajectoriesMonteCarlo,
                                               args=(tempVariables.copy(), 
                                                     result,
                                                     (hIter, kIter, smoothIter, seedIter)))
                    p.start()
                    processes.append(p)



    # Wait until the processes are done.
    for pro in processes:
        pro.join()

    return dict(result)


if __name__ == '__main__':
    # The initial condition:
    import numpy as np
    from scipy.stats import norm
    import math
    import time
    import multiprocessing
    import dill as pickle

    ### And some color for the noise
    ## This is one example of how tow generate noise due to explicit eigenfunctions.
    eigV = lambda s :lambda j : (j* math.pi)**(-s*2)
    eigF = lambda j : lambda y: math.sqrt(2)*np.sin(j* math.pi * y)


    options = { 'TimeMax': 1,
                'InitSetting': 2,
                'eigFunc': eigF,
                'eigVal': eigV
              }



    parameter = { 'hSpan': 2.**(-np.array([2., 3., 4., 5., 6., 7., 9.])),
                  'kSpan': 2.**(-np.array([9.])),
                  'MonteCarloIter': range(2500),
                  'noiseSpan': [ 1., 1/2., 1/3., 1/4., 0. ]
    }


    now = time.time()
    result = runTrigonometricSimulation(parameter, options)
    print('Calculation took {0} s'.format(time.time() - now))
    result['parameter'] = parameter
    result['options'] = options
    pickle.dump(result, open('/home/thomas_schnake/Masterthesis_Simulation/waveTrajectories/convergence_analysis_Space_adapted_Trundation_InitSetting2.p', 'wb'))

 
