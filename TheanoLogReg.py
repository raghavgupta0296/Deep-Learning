import numpy
import theano
import theano.tensor as T

N=320
ips=780

D=(numpy.random.randn(N,ips),numpy.random.randint(0,2,N))

iter = 10000

x=T.dmatrix("x")
y=T.dvector("y")

w=theano.shared(numpy.random.randn(ips),name="w")
b=theano.shared(0.,name="b")

print(" Random Weights and Bias  :   ")

print w.get_value()
print b.get_value()

print " ---------------------------------------  "

fn = 1/(1+T.exp(-T.dot(x,w)-b))
threshold = fn > 0.5
crossEntropy = -y*T.log(fn)-(1-y)*T.log(1-fn)
cost = crossEntropy.mean()+0.01*(w**2).sum()
gw, gb = T.grad(cost, [w, b])

train = theano.function(
    inputs=[x, y],
    outputs=[threshold, crossEntropy],
    updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb))
)

predict=theano.function(inputs=[x],outputs=threshold)

for i in range(iter):
    pred, err = train(D[0], D[1])
    if i % 1000 == 0:
        print pred, err
        print " ____________"

print " Final Model : "
print w.get_value()
print b.get_value()

print "_____________________________"

print " target values  for D : "
print D[1]

print " prediction : "
print predict(D[0])
