import mxnet as mx

'''
Implementation of https://medium.com/@julsimon/an-introduction-to-the-mxnet-api-part-2-ce761513124e
'''
a = mx.symbol.Variable('A')
b = mx.symbol.Variable('B')
c = mx.symbol.Variable('C')
d = mx.symbol.Variable('D')

e = (a + b) * (c + d)

print(e.list_arguments())
print(e.list_outputs())
print(e.get_internals().list_outputs())

a_data = mx.nd.uniform(low=0, high=1, shape=(1000, 1000))
b_data = mx.nd.uniform(low=0, high=1, shape=(1000, 1000))
c_data = mx.nd.uniform(low=0, high=1, shape=(1000, 1000))
d_data = mx.nd.uniform(low=0, high=1, shape=(1000, 1000))

executor = e.bind(mx.cpu(), {
    'A': a_data,
    'B': b_data,
    'C': c_data,
    'D': d_data,
})

e_data = executor.forward()

print(e_data)
