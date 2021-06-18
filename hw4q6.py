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