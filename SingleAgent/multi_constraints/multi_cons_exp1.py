"""
This file implements the multi-density-constraint for discrete cases
We have : 
1) Atomic constraint in forms of sum(Ax+b)<=0
2) Formula in forms of Atomic | Negation | Conjunction | Disjunction
We have one multiplier for each atomic constraint, and the multiplier for each formula follows the rule:
1) phi=alpha -> sigma(phi)=max(sigma(alpha),0)
2) phi=negation phi' -> sigma(phi)=max(-sigma(phi'),0)
3) phi=phi_1 and phi_2 -> sigma(phi)=max(sigma(phi_1),sigma(phi_2))
4) phi=phi_1 or phi_2 -> sigma(phi)=min(sigma(phi_1),sigma(phi_2))
"""
import numpy as np


class Atomic(object):
    def __init__(self, n_states, A, b, default_multiplier=0, default_beta = 1):
        # Initialize with a given A and b
        self.n_states = n_states
        self.A = A
        self.b = b
        self.beta = default_beta
        self.sigma = default_multiplier # Initialize the multiplier with a default value
        assert len(A) == self.n_states
        assert len(b) == self.n_states

    def calculate(self, rhos):
        # Calculate the value of A*rho+b
        # The result should be a vector with length n_states
        return self.A * rhos + self.b

    def process_bool(self, rhos):
        # Return the result of sum(A*rho+b) <= 0
        # The result should be a single boolean value
        assert len(rhos) == self.n_states
        # val = self.calculate(rhos)
        return sum(self.calculate(rhos)) <= 0

    def process_val(self, rhos):
        # Return the result of sum(A*rho+b)
        # The result should be a single real
        assert len(rhos) == self.n_states
        return np.sum(self.calculate(rhos))
    
    def update_multiplier(self, rhos,min_val=0,max_val=None):
        # Update the multiplier by: sigma <- max(0,sigma+beta(rho))
        assert len(rhos) == self.n_states
        if max_val is not None:
            self.sigma = min(max_val, max(min_val, self.sigma+self.beta*self.process_val(rhos)))
        else:
            self.sigma = max(min_val, self.sigma+self.beta*self.process_val(rhos))
        return 1

    def get_multiplier(self, state_id):
        #TODO: Use weighted multiplier for now
        total_weight = sum(np.abs(self.A))
        # weighted_sigma = self.sigma*self.A[state_id]/total_weight
        weighted_sigma = self.sigma*np.abs(self.A[state_id])/total_weight
        return weighted_sigma

class Formula(object):
    """
    Abstract class for all formulae
    """

    def __init__(self) -> None:
        self.sigma = 0

    def get_multiplier(self, state_id):
        pass

    def get_bool(self):
        pass

    def get_val(self):
        pass


class AtomFormula(Formula):
    """
    phi = alpha
    sigma(phi)=max(sigma(alpha),0)
    """

    def __init__(self, atom: Atomic):
        super(Formula, self).__init__()
        self.alpha = atom

    def get_multiplier(self, state_id):
        # TODO: May need to update this
        return self.alpha.get_multiplier(state_id)

    def get_bool(self, rhos):
        return self.alpha.process_bool(rhos)
    
    def get_val(self, rhos):
        return self.alpha.process_val(rhos)
    
    def update_multiplier(self,rhos,min_val=0,max_val=None):
        self.alpha.update_multiplier(rhos,min_val=min_val,max_val=max_val)
        return 1

    
class NegFormula(Formula):
    """
    phi=negation phi' -> sigma(phi)=max(-sigma(phi'),0)
    """

    def __init__(self, formula):
        super(Formula, self).__init__()
        self.phi_1 = formula
        pass

    def get_multiplier(self, state_id):
        # TODO: Decide whether we have a multiplier for each state or each formula
        sigma = self.phi_1.get_multiplier(state_id)
        return -sigma
        # return max(-sigma, 0)

    def get_bool(self, rhos):
        return (not self.phi_1.get_bool(rhos))
    
    def get_val(self, rhos):
        return -self.phi_1.get_val(rhos)
    
    def update_multiplier(self,rhos,min_val=0,max_val=None):
        self.phi_1.update_multiplier(rhos,min_val=min_val,max_val=max_val)
        return 1

class AndFormula(Formula):
    """
    phi=phi_1 and phi_2 -> sigma(phi)=max(sigma(phi_1),sigma(phi_2))
    """

    def __init__(self, formula1, formula2):
        super(Formula, self).__init__()
        self.phi_1 = formula1
        self.phi_2 = formula2
        pass

    def get_multiplier(self, state_id):
        # TODO: Decide whether we have a multiplier for each state or each formula
        sigmas = np.array([self.phi_1.get_multiplier(state_id), self.phi_2.get_multiplier(state_id)])
        return max(sigmas,key=abs)
        # return max(sigmas)

    def get_bool(self, rhos):
        return (self.phi_1.get_bool(rhos) and self.phi_2.get_bool(rhos))
    
    def get_val(self, rhos):
        return max(self.phi_1.get_val(rhos), self.phi_2.get_val(rhos))
    
    def update_multiplier(self,rhos,min_val=0,max_val=None):
        self.phi_1.update_multiplier(rhos,min_val=min_val,max_val=max_val)
        self.phi_2.update_multiplier(rhos,min_val=min_val,max_val=max_val)
        return 1

class OrFormula(Formula):
    """
    phi=phi_1 or phi_2 -> sigma(phi)=min(sigma(phi_1),sigma(phi_2))
    """

    def __init__(self, formula1, formula2):
        super(Formula, self).__init__()
        self.phi_1 = formula1
        self.phi_2 = formula2

    def get_multiplier(self, state_id):
        # TODO: Decide whether we have a multiplier for each state or each formula
        sigmas = np.array([self.phi_1.get_multiplier(state_id), self.phi_2.get_multiplier(state_id)])
        # return min(sigmas)
        return min(sigmas,key=abs)
    
    def get_bool(self, rhos):
        return (self.phi_1.get_bool(rhos) or self.phi_2.get_bool(rhos))
    
    def get_val(self, rhos):
        return min(self.phi_1.get_val(rhos), self.phi_2.get_val(rhos))

    def update_multiplier(self,rhos,min_val=0,max_val=None):
        self.phi_1.update_multiplier(rhos,min_val=min_val,max_val=max_val)
        self.phi_2.update_multiplier(rhos,min_val=min_val,max_val=max_val)
        return 1

if __name__ == "__main__":
    # This will be a testing of rho(s1)==rho(s2), implemented by two atomics, two atomic formulae and an and operation
    a1 = Atomic(2,np.array([1,-1]),np.zeros(2))
    a2 = Atomic(2,np.array([-1,1]),np.zeros(2))
    phi1 = AtomFormula(a1)
    phi2 = AtomFormula(a2)
    phi = AndFormula(phi1,phi2)
    
    print(phi.get_bool(np.array([5,5])))
    print(phi.get_bool(np.array([5,-10])))

    # Test the multiplier part
    rhos = [5,-10]
    phi.update_multiplier(rhos)

    sig = phi.get_multiplier(1)
    print(sig)
