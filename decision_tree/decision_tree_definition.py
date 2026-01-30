import numpy as np


def deg_2_poly(x):
    return x[1]*x[5] - x[3]*x[4] + x[0]*x[4] - x[1]*x[3] \
    + x[3]*x[5] - x[0]*x[2] - x[1]*x[2] - x[0]*x[1] \
    + x[0]*x[3] - x[2]*x[5] - x[4]*x[5] - x[2]*x[4]

def deg_2_poly_2(x):
    return x[4]**2 + x[2]**2 + x[5]**2 + x[1]**2 + x[0]**2 + x[3]**2

def deg_4_poly(x):
    return -x[0]*x[1]*x[4]*x[5] -x[1]*x[2]*x[3]*x[4] + x[0]*x[2]*x[3]*x[5] 

def deg_4_poly_2(x):
    return (x[1]**2)*(x[4]**2) + (x[0]**2)*(x[5]**2) + (x[2]**2)*(x[3]**2) 

def deg_4_poly_3(x): #This is probably not correct, it only helps classify around 20 examples
    return -x[0]*x[1]*(x[3]**2) - (x[0]**2)*x[3]*x[4] + x[0]*(x[3]**2)*x[4] - (x[0]**2)*x[1]*x[3]

def term1(x):
    return -x[0]*x[1]*(x[2]**2)*(x[3]**2)*x[4]*x[5]

def term2(x):
    return x[0]*(x[1]**2)*x[2]*x[3]*(x[4]**2)*x[5]

def term3(x):
    return - (x[0]**2)*x[1]*x[2]*x[3]*x[4]*(x[5]**2) 

def deg_6_poly(x):
    return term1(x) + term2(x) + term3(x)

def _unhandled(context_dict):
    """Helper to print unhandled cases and return None."""
    msg_parts = [f"Case not accounted for:"]
    for k, v in context_dict.items():
        msg_parts.append(f"Value of {k} is {v}")
    print(" ".join(msg_parts))
    return None


def apply_deg_2_decision_criterion(x):
    deg_6_polynomial_value = deg_6_poly(x)

    assert deg_6_polynomial_value in [-256, -192, -112, -64, -40,  -24, -16, -12, -4, 8],\
        f"Invalid polynomial value: {deg_6_polynomial_value}. Expected one of [-256, -192, -112, -64, -40, -24, -12, -4, 8]"
    
    deg_2_poly_value = deg_2_poly(x)
    
    deg_6_value_to_deg_2_value_dict = {-256: [[16], [-16]], 
                                       -192: [[12], [-12]],
                                       -112: [[8], [-8]],
                                       -40: [[2, 8, 10], [-2, -8, -10]],
                                       -64: [[13, 16], [-16, -13, -11, -8, -5, 5, 8, 11]],
                                       -24: [[12], [-12.0, -6.0, 6.0]], 
                                       -12: [[9], [-9, -3, 3]], 
                                       -4: [[8], [-11.0, -8.0, -7.0, -5.0, -4.0, 4.0, 5.0, 7.0, 11.0] ], 
                                       8: [[-2, 2, 14], [-14.0, -10.0, -8.0, 8.0, 10.0] ]}    

    for (key, val_lists) in deg_6_value_to_deg_2_value_dict.items():
        if key == deg_6_polynomial_value:
            if deg_2_poly_value in val_lists[0]:
                return 1
            elif deg_2_poly_value in val_lists[1]:
                return -1
            else:
                print(f"Value of degree 6 polynomial is {deg_6_polynomial_value} and value of degree 2 polynomial is {deg_2_poly_value}, this case is not accounted for")
                return None

            
    if deg_6_polynomial_value == -16:
        if deg_2_poly_value in [4, 13, 16]:
            return 1
        elif deg_2_poly_value in [-16.0, -13.0, -11.0, -8.0, -7.0, -5.0, -4.0, 5.0, 7.0, 11.0]:
            return -1
        else:
            deg_4_poly_value = deg_4_poly(x)
            if deg_4_poly_value == -1:
                return 1
            elif deg_4_poly_value == -16:
                return -1
            else:
                print(f"Value of degree 6 polynomial is {deg_6_polynomial_value} and value of degree 2 polynomial is {deg_2_poly_value}, this case is not accounted for")
                return None

def case_0(x):
    d6 = 0
    num_zeros = len(np.where(np.array(x) == 0)[0])
    
    if num_zeros == 3:
        return -1
    
    d2 = deg_2_poly(x)
    if num_zeros == 0:
        if d2 == 15: return 1
        elif d2 in {-15, -9, -6, 6, 9}: return -1
        else:
            _unhandled({'d6': d6, 'num_zeros': num_zeros, 'd2': d2})
    
    elif num_zeros == 2:
        if d2 in {12, 16}: return 1
        if d2==8:
            d4 = deg_4_poly(x)
            if d4 == -4: return 1
            elif d4 == 0: return -1
            else:
                _unhandled({'d6': d6,'num_zeros': num_zeros, 'd2': d2, 'd4': d4})
        if d2==5:
            d4_2 = deg_4_poly_2(x)
            if d4_2 == 16: return 1
            elif d4_2 == 4: return -1
            else:
                return _unhandled({'d6': d6,'num_zeros': num_zeros, 'd2': d2, 'd4_2': d4_2})
        
        if d2 in {-16, -12, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 7, 9, 10}: return -1
        return _unhandled({'d6': d6, 'num_zeros': num_zeros, 'd2': d2})
    
    elif num_zeros == 1:
        if deg_4_poly_2(x) == 2.0: return -1
        
        d4 = deg_4_poly(x)
        if d4 == -2.0: return -1
        
        if d2 in {12, 13, 14, 16}: return 1
        if d2 in set(range(-14, -1)) | {-16, 0, 9}: return -1

        # Check deg 4 sets
        if d4 == 16: return 1
        if d4 in {-16, -8}: return -1


        d4 = deg_4_poly(x)
        if d4 == 16: return 1
        if d4 in {-16, -8}: return -1
        

        if d2 in {3, 10}: return 1
        if d4 == 8: 
            if d2 in {1, 2, 6}: return -1
            elif d2 in {-1, 5, 7}: return 1
            else:
                return _unhandled({'d6': d6,'num_zeros': num_zeros, 'd2': d2, 'd4': d4})

        elif d4 == 2: 
            if d2 in {-1, 1, 2, 4, 6, 8}: return -1
            elif d2==5: return 1

        elif d4 == -4:
            if d2 in {-1, 1, 2, 4, 5, 7}: return -1
            elif d2 in [6]: return 1
            elif d2 in {8, 11}:
                d2_2 = deg_2_poly_2(x)
                if d2_2 == 11: return 1
                elif d2_2 == 14: return -1
                else:
                    _unhandled({'d6': d6,'num_zeros': num_zeros, 'd2': d2, 'd2_2': d2_2, 'd4': d4})

        elif d4 == 4:
            if d2 in {-1, 8}: return -1
            if d2 in {2, 11}: return 1
            if d2 in {4, 5, 7}:
                d2_2 = deg_2_poly_2(x)
                if d2_2 == 14: return -1
                if d2_2 == 11: return 1
                else:
                    return _unhandled({'d6': d6, 'd4': d4, 'd2': d2, 'd2_2': d2_2})

            if d2 == 1:
                d2_2 = deg_2_poly_2(x)
                if d2_2 == 14: return -1

                #Note: this only helps with around 20 cases, this is probably not the best polynomial
                d4_3 = deg_4_poly_3(x)
                if d4_3 in {2, 4, 6, 16}: return 1
                if d4_3 in {-10, -6}: return -1
                return _unhandled({'d6': d6, 'd4': d4, 'd2': d2, 'd2_2': d2_2, 'd4_3': d4_3})
        
        else:
            return _unhandled({'d6': d6, 'd2': d2, 'd4': d4})

def decision_tree(x):
    determined_by_deg_6_and_deg_2_poly_list = [-256, -192, -112, -64,  -40, -24,-16, -12, -4, 8]
    
    deg_6_poly_value = deg_6_poly(x)

    if deg_6_poly_value > 8:
        return 1

    elif deg_6_poly_value in [3]:
        return 1

    elif deg_6_poly_value in [-1]:
        return -1

    elif deg_6_poly_value in determined_by_deg_6_and_deg_2_poly_list:
        pred = apply_deg_2_decision_criterion(x)
        return pred

    elif deg_6_poly_value == 0:
        pred = case_0(x)
        return pred
    else: 
        print(f"The value of degree 6 polynomial is {deg_6_poly_value}, need to add this case!")
        return None
        
