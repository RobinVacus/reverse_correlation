import matplotlib.pyplot as plt
import math
import random
import os
import sys

plt.rcParams['font.size'] = '21'
plt.rcParams['text.usetex'] = 'True'
plt.rcParams["font.family"] = "CMU"

class BernoulliSet:
    """ Iterates through all possible realisations of a set {X_i} of n independent Bernoulli random variables
        represented by the array 'probabilities'. For each realisation, returns a boolean array
        whose i-th element represents the success or failure of X_i, as well as the corresponding probability.

        Parameters
        ----------

        probabilities : float array
            The i-th element represents the probability that X_i = 1. """

    def __init__(self,probabilities):

        self.probabilities = probabilities
        self.n = len(probabilities)
        self.stop = False
        self.current = [False]*self.n
    
    def __iter__(self):
        return self

    def __next__(self):

        if self.stop:
            raise StopIteration

        p = 1
        for i in range(self.n):
            if self.current[i]:
                p *= self.probabilities[i]
            else:
                p *= 1.-self.probabilities[i]
        
        result = self.current[:]
        i = self.n-1
        while i >= 0 and self.current[i]:
            i -= 1
        if i < 0:
            self.stop = True
        else:
            self.current[i] = True
            for j in range(i+1,self.n):
                self.current[j] = False
        
        return result,p


def computeExpectation(probabilities,function):
    """ Computes the average of 'function' on a set {X_i} of n indepedent Bernoulli random variables.

        Parameters
        ----------

        probabilities : float array
            The i-th element represents the probability that X_i = 1.

        function : function
            Maps a boolean array whose i-th element represents the success of X_i,
            to a tuple containing the payoff of the first player and the total production. """

    averagePayoff,averageTotalProduction = 0,0

    for (realisation,proba) in BernoulliSet(probabilities):

        payoff,totalProduction = function(realisation)
        averagePayoff += payoff * proba
        averageTotalProduction += totalProduction * proba

    return averagePayoff,averageTotalProduction


def foragingGamePayoff(strategyProfile,s=0.35,gamma=0.5):
    """ Returns the payoff of the first player and the total production in the Foraging game,
        given the specified strategy profile.

        Parameters
        ----------

        strategyProfile : float array
            The i-th element represents the probability that Player i is a producer.

        s : float , default 0.35
            The finder's share.

        gamma : float, default 0.5
            A parameter representing individual capabilities or food abundance and positively
            correlated with the payoffs. """

    n = len(strategyProfile)
    
    def aux(realisation):

        # The boolean 'realisation[i]" represents whether Player i is a producer
        nProducers = sum(realisation)
        payoff = 0
        
        if realisation[0]: # If the first player is a producer
            payoff = s*(1+gamma) + (1-s)*(1+gamma)/(1+n-nProducers)

        else: # If the first player is a scrounger
            payoff = gamma + nProducers*(1-s)*(1+gamma)/(1+n-nProducers)

        totalProduction = nProducers * (1+gamma) + (n - nProducers) * gamma
        return payoff,totalProduction

    return computeExpectation(strategyProfile,aux)


def companyGamePayoff(strategyProfile,phi=(lambda x:1-math.exp(-2*x)),p=0.5,a=0.5,delta=0.6,c=0.1,gamma=1):
    """ Returns the payoff of the first player and the total production in the Company game,
        given the specified strategy profile.

        Parameters
        ----------

        strategyProfile : float array
            The i-th element represents the probability that Player i is a producer.

        phi : function , default x -> 1-exp(-2x)
            Maps salaries to payoffs. Should be concave and non-decreasing.

        p : float , default 0.5
            The probability that a player founds food.

        a : float , default 0.5
            The fraction of the producers production reached by scroungers.

        delta : float , default 0.6
            The extent to which the salary of each player depends on the total production.

        c : float , default 0.1
            The energetic cost paid by producers.

        gamma : float, default 1
            A parameter representing individual capabilities or food abundance and positively
            correlated with the payoffs. """

    n = len(strategyProfile)
    
    def aux(realisation):

        # The first half of the realisations represents the strategies
        strategies = realisation[:n]
        # The second half represents success in finding food
        success = realisation[n:]
        production = [0]*n
        
        for i in range(n):
            if success[i]: # If Player i was able to produce
                if strategies[i]: # If Player i is a producer
                    production[i] = gamma
                else: # If Player i is a scrounger 
                    production[i] = a*gamma

        totalProduction = sum(production)

        salary = delta*production[0] + (1-delta)*(totalProduction-production[0])/(n-1)
        payoff = phi(salary)
        
        if strategies[0]:
            # If the first player is a producer, she must pay an energetic cost
            payoff -= c

        return payoff,totalProduction

    return computeExpectation(strategyProfile+[p]*n,aux)


def findESS(payoffFunction,nPlayers,eps=1e-4,steps=10):
    """ Computes the approximate ESS of a game with the specified payoff function, in the case that it exists and is unique.
        Returns the ESS, the corresponding payoff, and the corresponding total production.
        Returns None if no unique ESS is found.

        Parameters
        ----------

        payoffFunction : function
            Maps a strategy profile to a tuple containing the payoff of the first player and the total production.

        nPlayers : int
            The number of players competing in the game.

        eps : float , default 1e-4
            The precision to which the ESS is computed.

        steps : int, default 10
            The number of points over which the unicity of the ESS is checked. """
    
    def payoff(q,p):
        """ Returns the payoff of the first player when its strategy is q, and all other players have
            strategy p."""

        strategyProfile = [q]+[p]*(nPlayers-1)
        return payoffFunction(strategyProfile)[0]

    def totalProduction(p):
        """ Returns the total production when all players are producers with probability p."""

        strategyProfile = [p]*nPlayers
        return payoffFunction(strategyProfile)[1]

    # We select 'steps' point equally spaced within [0,1]
    X = [i/(steps-1) for i in range(steps)]
    
    # We check the existence of an index i0 such that:
    #   for all j <= i0, payoff(1,X[j]) > payoff(0,X[j])
    #   for all j >  i0, payoff(1,X[j]) < payoff(0,X[j])
    #
    # If such index exists, we consider that the ESS exists and is unique.
    i0 = None

    for j in range(steps):

        if payoff(1,X[j]) < payoff(0,X[j]):
            
            if i0 == None:
                i0 = j-1

        else:

            # There is no such index. The function terminates with an error.
            if i0 != None:
                return None,None,None

    # If for all p, payoff(1,p) > payoff(0,p), then the only ESS is p_star = 1
    if i0 == None:
        return 1,payoff(1,1),totalProduction(1)
    
    # If for all p, payoff(1,p) < payoff(0,p), then the only ESS is p_star = 0
    if i0 < 0:
        return 0,payoff(0,0),totalProduction(0)

    # We refine the search within the interval [X[i0], X[i0+1]] to find the exact p for which payoff(1,p) = payoff(0,p)
    pmin = X[i0]
    pmax = X[i0+1]
    
    while pmax-pmin > eps:

        pmed = (pmin+pmax)/2.

        if payoff(1,pmed) > payoff(0,pmed):
            pmin = pmed
        else:
            pmax = pmed

    ESS = (pmin+pmax)/2.
    return ESS, payoff(ESS,ESS), totalProduction(ESS)


class Parameter:
    """ Can be passed to a function to specify what should be the X or Y-axis label and range."""

    def __init__(self,label,variable,valueMin,valueMax):

        self.valueMin = valueMin
        self.valueMax = valueMax
        self.label = label
        self.variable = variable

    def getRange(self,iterations):

        return [self.valueMin+(self.valueMax-self.valueMin)*i/(iterations-1) for i in range(iterations)]

    def getFullLabel(self):

        return self.label + " $" + self.variable + "$"

class Quantity:
    """ Can be passed to a function to specify what should be computed (payoff or total production)."""

    def __init__(self,key,label,variable,height):

        self.key = key
        self.label = label
        self.variable = variable
        self.height = height

    def getFullLabel(self):

        return self.label + " $" + self.variable + "$"
    
FRACTION_PRODUCER = Quantity(0,"fraction of\nproducers","p_\\star",2)
PAYOFF = Quantity(1,"payoff","\\pi_\\star",5)
TOTAL_PRODUCTION = Quantity(2,"total production","\\Gamma_\\star",5)


def progressBar(t,T,length=30):
    """ Returns a simple string to indicate the progress of a computation. """
    
    x = int(length*t/(T-1))
    return "|"+"="*x+" "*(length-x)+"|"
        
    
def figure2D(payoffFunctionMap,quantities,nPlayers,xParameter,yParameter,yValues,iterations=50,saveToFile=""):
    """ Generates a figure depicting some quantities at equilibrium.
        The X-axis corresponds to 'xParameter'.
        Creates one subfigure for each element in 'quantities'.
        In each subfigure, creates one plot for each element in 'yValues'.

        Parameters
        ----------

        payoffFunctionMap : function
            Maps a float tuple (x,y) to a payoff function.

        quantities : Quantity array
            The list of quantities to be computed and shown.

        nPlayers : int
            The number of players competing in the game.

        xParameter : Parameter
            The parameter corresponding to the X-axis.

        yParameter : Parameter
            The parameter corresponding to each plot.

        yValues : float array
            The list of values of the variable y for which different plots must be created.

        iterations : int , default 50
            The number of points of the X-axis to be processed.

        saveToFile : str , default ""
            The file in which to save the figure. If equal to None, the figure is immediately displayed instead. """

    if saveToFile != "" and os.path.exists(saveToFile):
        print("File {} already exists".format(saveToFile))
        return
    
    print("Generating figure2D : {}".format(saveToFile))

    X = xParameter.getRange(iterations)
    Q = len(quantities)

    Y = [[[None]*iterations for y in yValues] for quantity in quantities]
    
    for i in range(iterations):

        print("Processing {}/{}\t".format(i+1,iterations)+progressBar(i,iterations),end="\r")

        for k in range(len(yValues)):

            tmp = findESS(payoffFunctionMap(X[i],yValues[k]),nPlayers)
            for j in range(Q):
                Y[j][k][i] = tmp[quantities[j].key]

            
    print("\n")

    heights = [quantity.height for quantity in quantities]
    fig,axs = plt.subplots(Q,1,figsize=(8,sum(heights)),sharex=True,height_ratios=heights)

    #axs[0].set(**{"ylabel":"fraction of\nproducers $p_\\star$","ylim":(-0.1,1.1)})

    for j in range(Q):

        #axs[j].set_prop_cycle(color=['#fcbf49','#f77f00','#d62828'])
        #axs[j].set_prop_cycle(color=['#ffcc08','#f48c06','#9d0208'])

        axs[j].set_prop_cycle(color=['#b5e48c','#4cc9f0','#1e6091'])
        axs[j].set(**{"ylabel":quantities[j].getFullLabel()})

        for k in range(len(yValues)):

            label = "$"+yParameter.variable+" = "+str(yValues[k])+"$"
            axs[j].plot(X,Y[j][k],label = label)
            
        

    axs[1].legend(fontsize=18)
    axs[-1].set(**{"xlabel":xParameter.getFullLabel()})

    if saveToFile != "":
        plt.savefig(saveToFile,dpi=300,bbox_inches="tight")
    else:
        plt.show()


def figure3D(payoffFunctionMap,quantity,nPlayers,xParameter,yParameter,iterations=50,saveToFile=""):
    """ Generates a figure depicting some quantity at equilibrium as a function of some variables x and y.

        Parameters
        ----------

        payoffFunctionMap : function
            Maps a float tuple (x,y) to a payoff function.

        quantity : Quantity
            The quantity to be computed and shown.

        nPlayers : int
            The number of players competing in the game.

        xParameter : Parameter
            The parameter corresponding to the X-axis.

        yParameter : Parameter
            The parameter corresponding to the Y-axis.

        iterations : int , default 50
            The number of points of the X-axis to be processed.

        saveToFile : str , default None
            The file in which to save the figure. If equal to None, the figure is immediately displayed instead. """

    if saveToFile != "" and os.path.exists(saveToFile):
        print("File {} already exists".format(saveToFile))
        return
    
    print("Generating figure3D : {}".format(saveToFile))
    
    X = xParameter.getRange(iterations)
    Y = yParameter.getRange(iterations)

    Z = [[None for j in range(iterations)] for i in range(iterations)]
    
    for i in range(iterations):

        print("Processing {}/{}\t".format(i+1,iterations)+progressBar(i,iterations),end="\r")
        
        for j in range(iterations):

            tmp = findESS(payoffFunctionMap(X[j],Y[iterations-i-1]),nPlayers)
            Z[i][j] = tmp[quantity.key]

    print("\n")

    fig,ax = plt.subplots(1,1,figsize=(8,6.5))
        
    plt.xlabel(xParameter.getFullLabel())
    plt.ylabel(yParameter.getFullLabel())

    im = ax.imshow(Z,
               extent=(xParameter.valueMin,xParameter.valueMax,yParameter.valueMin,yParameter.valueMax),
               aspect="auto",
               cmap="coolwarm")
    plt.colorbar(im)

    if saveToFile != "":
        plt.savefig(saveToFile,dpi=300,bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":

    ITERATIONS = 20

    if len(sys.argv) > 1:
        ITERATIONS = int(sys.argv[1])
        
    # Figures for the Foraging game

    n = 4
    GAMMA = Parameter("low-hanging fruits","\\gamma",0,0.5)
    FINDERS_SHARE = Parameter("finder's share","s",0.3,0.4)
    MAP = (lambda x,y: (lambda strategyProfile : foragingGamePayoff(strategyProfile,s=y,gamma=x)) )
    
    figure2D(payoffFunctionMap = MAP,
             quantities = [FRACTION_PRODUCER,PAYOFF],
             nPlayers = n,
             xParameter = GAMMA,
             yParameter = FINDERS_SHARE,
             yValues = [0.2,0.3,0.4],
             iterations = ITERATIONS,
             saveToFile = "foraging_2D_payoff.pdf")

    
    figure3D(payoffFunctionMap = MAP,
             quantity = PAYOFF,
             nPlayers = n,
             xParameter = GAMMA,
             yParameter = FINDERS_SHARE,
             iterations = ITERATIONS,
             saveToFile = "foraging_3D_payoff.pdf")
    
    
    # Figures for the Company game with phi : x -> 1-exp(-2x)

    n = 4
    GAMMA = Parameter("individual capability","\\gamma",1,2)
    COST = Parameter("energetic cost","c",0.06,0.1)
    phi = (lambda x: 1-math.exp(-2*x))
    MAP = (lambda x,y: (lambda strategyProfile : companyGamePayoff(strategyProfile,delta=0.6,c=y,gamma=x,phi=phi)) )

    figure2D(payoffFunctionMap = MAP,
             quantities = [FRACTION_PRODUCER,PAYOFF,TOTAL_PRODUCTION],
             nPlayers = n,
             xParameter = GAMMA,
             yParameter = COST,
             yValues = [0.07,0.08,0.09],
             iterations = ITERATIONS,
             saveToFile = "company_phiExp_2D.pdf")

    
    figure3D((lambda x,y: (lambda strategyProfile : companyGamePayoff(strategyProfile,delta=0.6,c=y,gamma=x,phi=phi)) ),
             quantity = PAYOFF,
             nPlayers = n,
             xParameter = GAMMA,
             yParameter = COST,
             iterations = ITERATIONS,
             saveToFile = "company_phiExp_3D_payoff.pdf")


    figure3D((lambda x,y: (lambda strategyProfile : companyGamePayoff(strategyProfile,delta=0.6,c=y,gamma=x,phi=phi)) ),
             quantity = TOTAL_PRODUCTION,
             nPlayers = n,
             xParameter = GAMMA,
             yParameter = COST,
             iterations = ITERATIONS,
             saveToFile = "company_phiExp_3D_totalProduction.pdf")

    # Figures for the Company game with phi : x -> min(1,x)

    n = 4
    GAMMA = Parameter("individual capability","\\gamma",1.25 , 1.75)
    COST = Parameter("energetic cost","c",0.12,0.18)
    phi = (lambda x: min(1,x))
    MAP = (lambda x,y: (lambda strategyProfile : companyGamePayoff(strategyProfile,c=y,gamma=x,phi=phi,delta=0.7)) )
    
    figure2D(payoffFunctionMap = MAP,
             quantities = [FRACTION_PRODUCER,PAYOFF,TOTAL_PRODUCTION],
             nPlayers = n,
             xParameter = GAMMA,
             yParameter = COST,
             yValues = [0.1,0.15,0.2],
             iterations = ITERATIONS,
             saveToFile = "company_phiMin_2D.pdf")
    

    figure3D(payoffFunctionMap = MAP,
             quantity = PAYOFF,
             nPlayers = n,
             xParameter = GAMMA,
             yParameter = COST,
             iterations = ITERATIONS,
             saveToFile = "company_phiMin_3D_payoff.pdf")


    figure3D(payoffFunctionMap = MAP,
             quantity = TOTAL_PRODUCTION,
             nPlayers = n,
             xParameter = GAMMA,
             yParameter = COST,
             iterations = ITERATIONS,
             saveToFile = "company_phiMin_3D_totalProduction.pdf")
    
    
    # Figures for the Company game with phi : x -> x

    n = 2
    GAMMA = Parameter("individual capability","\\gamma",1,2)
    COST = Parameter("energetic cost","c",0.06,0.1)
    phi = (lambda x: x)
    MAP = (lambda x,y: (lambda strategyProfile : companyGamePayoff(strategyProfile,c=y,gamma=x,phi=phi,delta=0.7)) )

    figure2D(payoffFunctionMap = MAP,
             quantities = [FRACTION_PRODUCER,PAYOFF,TOTAL_PRODUCTION],
             nPlayers = n,
             xParameter = GAMMA,
             yParameter = COST,
             yValues = [0.2,0.25,0.3],
             iterations = ITERATIONS,
             saveToFile = "company_phiLinear_2D.pdf")
    
    
    



















