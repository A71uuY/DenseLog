import numpy as np

class ConsFormula:
    def __init__(self, state_size, A,b):
        # Ax+b<=0
        self.A=A
        self.b=b
        assert len(A)==state_size
    def get_val(self,density_vec):
        value = np.sum(self.A*density_vec)+self.b
        return value
    def get_bool(self,density_vec):
        return self.get_val(density_vec)<=0

class OrFormula:
    def __init__(self, phi1,phi2):
        # Ax+b<=0
        self.phi1=phi1
        self.phi2=phi2
    def get_val(self,density_vec):
        val1 = max(0,self.phi1.get_val(density_vec))
        val2 = max(0,self.phi2.get_val(density_vec))
        
        # if val1 == 0:
        #     return val2*0.5
        # elif val2 == 0:
        #     return val1*0.5
        # elif (val1+val2) == 0:
        #     return 0
        # else:
        #     # all_weight=val2+val1
        #     # w1,w2 = val2/all_weight,val1/all_weight
        #     # return w1*val1+w2*val2
        #     weighted_val = val2*val1*2/(val2+val1)
        #     return weighted_val
        return min(val1, val2)
    def get_bool(self,density_vec):
        return np.logical_or(self.phi1.get_bool(density_vec),self.phi2.get_bool(density_vec))
    def get_max_vio(self,density_vec):
        val1 = max(0,self.phi1.get_val(density_vec))
        val2 = max(0,self.phi2.get_val(density_vec))
        if val1 == val2 and val1 < 0.00000000001:
            print("No violation")
            return 0
        if val1 > val2:
            print("Max violation from val1 is %f" % val1 )
        else:
            print("Max violation from val2 is %f" % val2 )

def pdl_objective_func(constraints,density_vec):
    # The objective should be a conjunction, so I take sum
    return np.sum([constraint.gev_val(density_vec) for constraint in constraints])



