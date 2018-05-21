# Some helping functions
import scipy.sparse as ssp
import numpy as np
from scipy.stats import norm
import math

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
