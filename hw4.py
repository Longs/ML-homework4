
import numpy as np

data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5

def margin(data,labels,th,th0):
    norm = (np.sum(th**2))**0.5
    print(f"norm: {norm}")
    return labels*(np.dot(th.T,data) + th0)/norm

def s_sum(data,labels,th,th0):
    margins = margin(data,labels,th,th0)
    return np.sum(margins[0])

def s_min(data,labels,th,th0):
    margins = margin(data,labels,th,th0)
    return np.amin(margins[0])

def s_max(data,labels,th,th0):
    margins = margin(data,labels,th,th0)
    return np.amax(margins[0])

d = (data,labels,red_th,red_th0)

print(f"red margins: {margin (data,labels,red_th,red_th0)}")
print(f"ssum: {s_sum(data,labels,red_th,red_th0)}, s_min: {s_min(data,labels,red_th,red_th0)}, s_max: {s_max(data,labels,red_th,red_th0)}")
d = (data,labels,blue_th,blue_th0)
print(f"blue margins: {margin(data,labels,blue_th,blue_th0)}")
print(f"ssum: {s_sum(data,labels,blue_th,blue_th0)}, s_min: {s_min(data,labels,blue_th,blue_th0)}, s_max: {s_max(data,labels,blue_th,blue_th0)}")

data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4

def hinge_loss(margin, gamma_ref):
    if margin > gamma_ref:
        return 0
    else:
        return 1-margin/gamma_ref

vectorised_hinge = np.vectorize(lambda margin : hinge_loss(margin,(1/2**0.5)))

print(f"hinge loss \n")
raw_margins = margin(data,labels,th,th0)
print(f"{raw_margins}")
print(f"hinge_loss: {vectorised_hinge(raw_margins)}")

