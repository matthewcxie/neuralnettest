Following Kaparthy, I implemented a version of a basic neural network. This uses backpropogation to train neural networks incrementally. Given target values and initial values, we can interatively tweak our weights and understand how we tweak each weight using back propogation, which tells us how each neuron will affect the loss function. We do this iteratively to minimize the loss function. We minimize the MSE by tweaking neurons in each layer until we achieve the desired result.

Here's an example of how this works:

1. Initialize the network

```python
model = multilayer(5, [3, 3, 1])
```

We make a simple model of five inputs, two hidden layers of three neurons each, and a single output neuron. We define a dataset that is given to us, and the target outputs we want

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]
```

We can now calculate the MSE loss by iterating through our data and adjusting the weights using gradient descent

```python
numsteps = 100
rateoflearning = 0.01 #delta of what we tweak our rates by each iteration


for k in range(numsteps):
    ypred = [model(x)[0] for x in xs]

    #calculate MSE loss
    loss = sum(((out - ygt) ** 2 for out, ygt in zip(ys, ypred)), Value(0.0)) #uses HOF to combine two values in order to track 
    loss = loss * (1.0/len(ys))

    #reset gradients back to zero in order for them not to affect the next iteration
    for p in model.parameters():
        p.grad = 0.0
    
    #back propogation using chain rule and rules given in engine
    loss.backward()

    #gradient descent
    for p in model.parameters():
        p.data -= rateoflearning * p.grad
    

    #print out the progress made: closer to zero, the smaller the loss function is
    print(f"{k:02d}, Loss: {loss.data:.4f}")

```

We can also inspect specific neurons to find their weight and their gradient, which tells us how much changing them will change the final output during back propogation

```python
# first weight of first neuron in layer 0
first_weight = model.layers[0].neurons[0].w[0]
print("Weight data:", first_weight.data)
print("Weight gradient:", first_weight.grad)

# bias of neuron in final layer (layer -1)
final_bias = model.layers[-1].neurons[0].b
print("Output Bias data:", final_bias.data)
print("Output Bias gradient:", final_bias.grad)
```
