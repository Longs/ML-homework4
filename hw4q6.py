import numpy as np

def rv(value_list):
    return np.array([value_list])

def cv(value_list):
    return np.transpose(rv(value_list))

def f1(x):
    return float((2 * x + 3)**2)

def df1(x):
    return 2 * 2 * (2 * x + 3)

def f2(v):
    x = float(v[0]); y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def df2(v):
    x = float(v[0]); y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])

def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]


#########################################
#Gradient descent with analytic gradient:
#########################################

def gd(f, df, x0, step_size_fn, max_iter):
    """
    f: a function whose input is an x, a column vector, and returns a scalar.
    df: a function whose input is an x, a column vector, and returns a column vector representing the gradient of f at x.
    x0: an initial value of xx, x0, which is a column vector.
    step_size_fn: a function that is given the iteration index (an integer) and returns a step size.
    max_iter: the number of iterations to perform
    """
    x = x0
    xs=[x]
    fs=[f(x)]
    
    for t in range(0,max_iter):
        
        temp_x = x.copy() - (step_size_fn(t)*df(x))

        xs.append(temp_x)
        fs.append(f(temp_x))
        x = temp_x.copy()

    return x,fs,xs


# Test case 1
ans=package_ans(gd(f1, df1, cv([0.]), lambda i: 0.1, 1000))
print(ans)
# Test case 2
ans=package_ans(gd(f2, df2, cv([0., 0.]), lambda i: 0.01, 1000))
print(ans)

#########################################
#Numerical gradient:
#########################################

def num_grad(f, delta=0.001):
    def df(x):
        out = []

        def delta_v(i,size,delta):
            out = np.zeros(size)
            out[i] = delta
            return cv(out)

        for i in range(len(x)):
            dv = delta_v(i,len(x),delta)
            out.append((f(x+dv) - f(x-dv))/(2*delta))
        return cv(np.array(out))

    return df


#numerical gradient test cases
x = cv([0.])
ans=(num_grad(f1)(x).tolist(), x.tolist())
print(ans)

x = cv([0.1])
ans=(num_grad(f1)(x).tolist(), x.tolist())
print(ans)

x = cv([0., 0.])
ans=(num_grad(f2)(x).tolist(), x.tolist())
print(ans)

x = cv([0.1, -0.1])
ans=(num_grad(f2)(x).tolist(), x.tolist())
print(ans)


#########################################
#Gradient descent with numerical gradient:
#########################################

def minimize(f, x0, step_size_fn, max_iter):

    df = num_grad(f) #numerical gradient function for f
    return gd(f, df, x0, step_size_fn, max_iter) #gradient function is passed in


ans = package_ans(minimize(f1, cv([0.]), lambda i: 0.1, 1000))
ans = package_ans(minimize(f2, cv([0., 0.]), lambda i: 0.01, 1000))

#########################################
# Support Vector Machine objective
#########################################

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

# Test case 1
x_1, y_1 = super_simple_separable()
th1, th1_0 = sep_e_separator

def hinge(v):
    return np.where(v<1,1-v,0)

print(f"hinge 1 {hinge(1)} hinge 1.1 {hinge(1.1)} hinge 0.5 {hinge(0.5)} hinge -0.5 {hinge(-0.5)}")

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    sum_loss = 0
    #v = np.dot(y,np.dot(x.T,th)+th0)
    v = (np.dot(x.T,th)+th0)
    for i in range(len(v)):
        sum_loss +=hinge(v[i][0]*y[0][i])
    return sum_loss/len(v)


# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    return (hinge_loss(x, y, th, th0) + lam*(np.dot(th.T,th))).item()
    pass


"""ans = svm_obj(x_1, y_1, th1, th1_0, .1)
print(ans)

# Test case 2
ans = svm_obj(x_1, y_1, th1, th1_0, 0.0)
print(ans)"""

ans=svm_obj(x_1, y_1, 0.1*th1, th1_0, 0.0)
print(ans)