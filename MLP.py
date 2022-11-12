

class MLP(object):

  def __init__(self, lr, layers):
    self.lr = lr
    self.layers = []
    n_input = 1
    for n_units in layers:
       self.layers.append(Layer(n_units, n_input))
       n_input = n_units

  def forward_step(self, linput):

    layer_input = linput
    for layer in self.layers:
      layer_input = layer.forward_step(layer_input)
    return layer_input

   def backpropagation(self, linput, llp):
        delta = llp - linput
        for layer in reversed(self.layers):
            delta = layer.backward_step(delta, self.lr)
    


