"""
For continuous cases, we need a real mapping from s \in S to 2^AP
So we define various things on 2^AP.
suppose we have {a,b,c} as ap, 
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
        v = self.A * rhos + self.b
        return v

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
        weighted_sigma = self.sigma*self.A[state_id]/total_weight
        # Exp1, for convience of coding, we use abs here to make calculation consistent
        # weighted_sigma = self.sigma*np.abs(self.A[state_id])/total_weight 
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
    
    def in_atom(self,sid):
        return self.alpha.A[sid]!=0

    
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
        # # return min(sigmas)
        # weight = sum(abs(sigmas))
        # if weight == 0:
        #     return min(sigmas,key=abs)
        # else:
        #     weights = np.array([abs(sigmas[1]) / weight, abs(sigmas[0]) / weight])
        #     result = sum(abs(sigmas)*weights)
        #     if sum(sigmas < 0):
        #         return -result
        #     else:
        #         return result
        # if not self.phi_1.in_atom(state_id):
        #     return sigmas[1]
        # elif not self.phi_2.in_atom(state_id):
        #     return sigmas[0]
        # else:
        vals = []
        for p in [self.phi_1,self.phi_2]:
            if hasattr(p, 'alpha'): # atom
                vals.append(p.alpha.sigma)
            else: # Another formula
                vals.append(p.get_multiplier(state_id))
        # vals = np.array([self.phi_1.alpha.sigma,self.phi_1.alpha.sigma])
        vals = np.array(vals)
        inverse_values = 1 / vals
        total_inverse = np.sum(inverse_values)
        probabilities = inverse_values / total_inverse
        if np.random.random_sample() <= probabilities[0]:
            return sigmas[0]
        else:
            return sigmas[1]
        # return min(sigmas,key=abs)
    
    def get_bool(self, rhos):
        return (self.phi_1.get_bool(rhos) or self.phi_2.get_bool(rhos))
    
    def get_val(self, rhos):
        v1,v2 = self.phi_1.get_val(rhos), self.phi_2.get_val(rhos)
        return min(self.phi_1.get_val(rhos), self.phi_2.get_val(rhos))

    def update_multiplier(self,rhos,min_val=0,max_val=None):
        self.phi_1.update_multiplier(rhos,min_val=min_val,max_val=max_val)
        self.phi_2.update_multiplier(rhos,min_val=min_val,max_val=max_val)
        return 1
    def in_atom(self, sid):
        return self.phi_1.in_atom(sid) or self.phi_2.in_atom(sid)
    
if __name__ == '__main__':
    a1 = Atomic(5,np.array([0,0,0,0,1]),np.array([0,0,0,50,0]),default_beta=0.25) # l4 <= 0
    phia = AtomFormula(a1)

    a2 = Atomic(5,np.array([0,0,0,1,0]),np.array([0,0,0,-100,0]),default_beta=0.25) # l3 < 10
    phib = AtomFormula(a2)

    a3 = Atomic(5,np.array([0,0,1,0,0]),np.array([0,0,0,0,-100]),default_beta=0.25) # l2.5 < 10
    phid = AtomFormula(a3)

    a4 = Atomic(5,np.array([0,0,0,0,1]),np.array([0,0,0,0,-100]),default_beta=0.25) # l4 < 5
    phic = AtomFormula(a4)
        # x=[0.15, 0.2, 0.25, 0.3, 1.0],
        # y=[120, 80, 10, 10, 10],
    phi_ad = OrFormula(phia,phid)
    phi_abd = OrFormula(phi_ad,phib)
    print(phi_ad.get_multiplier(3))
    print(phib.get_multiplier(3))
    print(phi_abd.get_multiplier(3))
    