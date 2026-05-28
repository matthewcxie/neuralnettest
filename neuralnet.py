#subscribing to pytorch API

class Neuron:
    def __init__(self, numinputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range (numinputs)]
        self.b = Value(random.uniform(-1, 1))
    def __call__(self, x):
        #multiply pairwise, use HOFs
        value = self.b
        for wi, xi in zip(self.w, x):
            value += wi * xi
        out = value.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]
class Layer:
    #list of neurons, just create a parameter for #
    def __init__(self, numinputs, numoutputs):
        self.neurons = [Neuron(numinputs) for _ in range(numoutputs)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
    def parameters(self):
        params = []
        for neuron in self.neurons:
            pars = neuron.parameters()
            params.extend(pars)
        return params
class multilayer:
    def __init__(self, numinputs, numoutputs):
        sizes = [numinputs] + numoutputs
        self.layers = []
        for i in range(len(numoutputs)):
            new_layer = Layer(sizes[i], sizes[i+1])
            self.layers.append(new_layer)
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        params = []
        for layer in self.layers:
            pars = layer.parameters()
            params.extend(pars)
        return params
        