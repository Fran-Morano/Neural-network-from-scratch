x = [1, -2, 3] #inputs
w = [-3, -1, 2] #weights
b = 1 # bias

xw0 = x[0]*w[0]
xw1 = x[1]*w[1]
xw2 = x[2]*w[2]

z = xw0 +xw1 +xw2 + b

y = max(z, 0) #relu
print(y)

#backward pass

dvalue = 1 #derivative from the next layer

#derivative of reLU and the chain rule
drelu_dz = dvalue*(1. if z>0 else 0.)
print(drelu_dz)


# Partial derivatives of the multiplication, the chain rule
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
#everything is 1 because de output changes just as much as the inputs of the sum changes
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)


# Partial derivatives of the multiplication, the chain rule
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
#when you derive a multiplication of 2 variables the result is the variable that you didnt derive
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)


'''
now that we have every derivative needed we could calculate every derivate of relu for each input/weight/bias variable
'''
#We can also create arrays for the gradient with all the partial derivatives we have calculated
dx = [drelu_dx0, drelu_dx1, drelu_dx2] # gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] # gradients on weights
db = drelu_db # gradient on bias...just 1 bias here

#We could use this to decrease the outpur of the ReLU function by applying a fraction of the gradients

w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

z2 = xw0 + xw1 + xw2 + b
# ReLU activation function
y2 = max(z2, 0)
print(y2)