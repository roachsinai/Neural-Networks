import theano
import theano.tensor as T
import random
import numpy

x = T.vector()
w = theano.shared(numpy.array([-1., 1.]))
b = theano.shared(0.)

z = T.dot(w,x)+b
y = 1/(1+T.exp(-z))

neuron = theano.function(inputs=[x],outputs=y)

y_hat = T.scalar() # 正确答案
cost = T.sum((y-y_hat)**2)

dw,db = T.grad(cost,[w,b])

# Effective Way:
gradient = theano.function(
		inputs=[x, y_hat],
		updates=[(w,w-0.1*dw),(b,b-0.1*db)]
		)
x = [1, -1]
y_hat = 1

for i in range(100):
	print(neuron(x))
	gradient(x,y_hat)
	print(w.get_value(),b.get_value())
