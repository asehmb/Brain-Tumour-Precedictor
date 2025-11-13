import torch

data = [[1,2,3],[4,5,6]]
my_tensor = torch.tensor(data)
print(my_tensor)

shape = (2,3)
ones = torch.ones(shape)
zeros = torch.zeros(shape)
random = torch.rand(shape)

print(f'random tensor:\n{random}')
#creation by mimicking
template = torch.tensor([[1,2],[3,4]])
rand_like = torch.randn_like(template, dtype=torch.float)

print(f'Template tensor:\n{template}\n')
print(f'Rand_like:\n {rand_like}')

#what's inside a tensor
tensor = torch.randn(2,3)

print(f'shape: {tensor.shape}')
print(f'datatype: {tensor.dtype}')
print(f'device: {tensor.device}')

# .shape: a tuple describe the dim. best debugging tool so far
# 90% of errors in pytorch will be shape mismatches
# .device: where the tensor lives. cpu of CUDA (gpu)
# .dtype: the data type of the numbers

# gradients: tiny, cts adjustment on the total weight
# Rule model parameters (weight, biases) MUST be in float type. float32 is a standard
# AUTOGRAD: automatic differentiation
# requires_grad = True (most important)
# sends a message to autograd: 'track every single operation that happens to it'

# build z = xy, where y = a + b
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(4.0, requires_grad=True)

y = a + b   # first operation
z = x * y   # 2nd operation
print(f'z: {z.grad_fn}')
print(f'x: {y.grad_fn}')
print(f'y: {a.grad_fn}')

'''
    Core verbs:
    "*" vs "@"
    element-wise multiplication ('*')
    matrix mult. ('@')
'''

a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[10,20],[30,40]])
print(a * b)

m1 = torch.tensor([[1,2,3],[4,5,6]])
m2 = torch.tensor([[7,8],[9,10],[11,12]])

print(m1 @ m2)

'''when building a linear layer, always use @ operator'''
#reduction operations and the dim argument
scores = torch.tensor([[10.,20.,30.],[5.,10.,15.]])
avg_scores = scores.mean()

print(f'overall mean: {avg_scores}')

'''dim arg let us control *which direction to collapse*'''

'''
    Rule:
    dim = 0, collapse rows. operates "vertically"
    dim = 1, collapse columns. operates "horizontally"

'''

avg_per_assignment = scores.mean(dim = 0)
avg_per_Student = scores.mean(dim = 1)
print(avg_per_assignment)
print(avg_per_Student)

#basic indexing
x = torch.arange(12).reshape(3,4)
'''
    tensor([[0.,1.,2.,3.],
            [4.,5.,6.,7.],
            [8.,9.,10.,11.]])
'''
#get the 3rd column (at index 2)
col_2 = x[:,2]
print(col_2)
#dynamic selection: 'argmax'
scores = torch.tensor([
    #best score is at index 3
    [10,0,5,20,1],
    #best score is at index 1
    [1,30,2,5,0]
])
best_indices = torch.argmax(scores, dim = 1)
print(best_indices)

# torch.gather()
data = torch.tensor([
    [10,11,12,13],
    [20,21,22,23],
    [30,31,32,33]
])
#choose which column to get from each row
indices_to_select = torch.tensor([[2],[0],[3]])
selected_values = torch.gather(data, dim = 1, index = indices_to_select)
print(selected_values)

#building model from scratch
'''
    forward pass: first guess
    
    
    
    '''
#batch
N = 10
D_in = 1
D_out = 1
#create input data
X = torch.randn(N, D_in)

true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
y_true = X@ true_W + true_b + torch.randn(N, D_out) * 0.1 #add a little noise

W = torch.randn(D_in,D_out,requires_grad=True)
b = torch.randn(1,requires_grad=True)

print(f'initial weight:\n{W}\n')
print(f'initial bias:\n{b}\n')
# model initial hypotheses
# model = X @ W + b
# X_train = torch.tensor(0.5)
# y_hat = model(X_train)
# print(f'prediction y_hat (first 3 rows):\n{y_hat[:3]}\n')
# print(f'prediction y_true (first 3 rows):\n{y_true[3:]}')
#
# #MSE
# error = y_hat - y_true
# squared_error = error**2
# loss = squared_error.mean()
# print(f'loss: {loss}')
#
# # most important commands (gradient descent)
# loss.backward()
'''
    w.grad = -const.
    (-) gradient -> increasing W decreases loss.
    gradient point toward steepest increase
    go OPPOSITE direction to minimize loss
    
    b.grad = -const.
    larger magnitude = steeper slope
    
    Measure error: loss
    know direction: .grad
    '''
#gradient descent
# W_new = W_old - learning_rate * W.grad
# b_new = b_old - learning_rate * b.grad

'''torch.no_grad(): don't track parameter updates
    .grad.zero_(): reset gradients each iteration'''
#hyperparameters
learning_rate, epochs = 0.01, 100
#initialize parameters
w, b = torch.randn(1,1,requires_grad=True), torch.randn(1,requires_grad=True)

#training loop
for epoch in range(epochs):
    # forward pass and loss
    y_hat = X @ w + b
    loss = torch.mean((y_hat - y_true)**2)

    # backward pass
    loss.backward()

    #update params
    with torch.no_grad():
        w -= learning_rate * w.grad; b -= learning_rate * b.grad

    #zero gradients
    w.grad.zero_(); b.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch:02d}: Loss={loss.item():.4f}, W={w.item():.3f}, b={b.item():.3f}')
print(f'\nfinal parameters: W={w.item():.3f}, b={b.item():.3f}')
print(f'True parameters: W=2.000, b=1.000')

'''torch.nn.linear: '''

D_in = 1
D_out = 1

linear_layer = torch.nn.Linear(in_features = D_in, out_features = D_out)

print(f"Layer's weight:\n{linear_layer.weight}\n")
print(f"Layer's bias:\n{linear_layer.bias}\n")

y_hat_nn = linear_layer(X)

print(f'Output of nn.Linear (first 3 rows):\n {y_hat_nn[:3]}')

'''nn.ReLU (Rectified Linear Unit): 
    If an input is negative, make it zero. "ReLU(x) = max(0, x)"'''

relu = torch.nn.ReLU()
sample_data = torch.tensor([-2.0,-0.5,0.0,0.5,2.0])
ReLU_data = relu(sample_data)
print(f'Original Data:\n {sample_data}')
print(f'Data after ReLU:\n {ReLU_data}')

'''nn.GELU (Gaussian ErrorLinear Unit): 
    The modern standard for Transformers (GPT, Llama). A smoother, gently curving version of ReLU'''
gelu = torch.nn.GELU()
GELU_data = gelu(sample_data)
print(f'Original Data:\n {sample_data}')
print(f'Data after GELU:\n {GELU_data}')

'''nn.SOFTMAX: Used on final output layer for classification
    converts logits to probabilities distribution (output in [0,1] and sum = 1)'''
softmax = torch.nn.Softmax(dim=-1)
logits = torch.tensor([[1.0,3.0,0.5,1.5],[-1.0,2.0,1.0,0.0]])
probabilities = softmax(logits)
print(f'Output Probabilities:\n {probabilities}\n')
print(f'Sum of probabilities for item 1:\n {probabilities[0].sum()}')

'''nn.embedding:
    words --> numbers
    learnable lookup table
    Each word gets a unique vector'''

vocab_size = 10
emb_dim = 3
emb_layer = torch.nn.Embedding(vocab_size, emb_dim)

input_ids = torch.tensor([[1,5,0,8]])
word_vectors = emb_layer(input_ids)
print(input_ids)
print(f'Word vectors:\n {word_vectors}\n')

'''nn.layernorm: 
    Prevents values from exploding/vanishing
    rescales to stable range
    essential for deep networks'''

norm_layer = torch.nn.LayerNorm(normalized_shape=3)
input_features = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
normalize_features = norm_layer(input_features)

print(normalize_features)
print(f'Mean (should be ~0): {normalize_features.mean(dim=-1)}')
print(f'std dev (should be ~1): {normalize_features.std(dim=-1)}')

'''nn.dropout: 
    prevent overfitting
    randomly zeros neurons during training
    forces network robustness
    most important thing: it only happens during training'''

#train vs eval mode for nn.dropout
dropout_layer = torch.nn.Dropout(p=0.5)
input_tensor = torch.ones(1,10)

#activate dropout for training
dropout_layer.train()
dropout_during_train = dropout_layer(input_tensor)

#deactivate dropout for eval/pred
dropout_layer.eval()
dropout_during_eval = dropout_layer(input_tensor)
print(dropout_during_train)
print(dropout_during_eval)

'''nn.Module: to organize our model
    torch.optim: to automate the learning'''

import torch.nn as nn
'''Rebuild our model'''
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # in the constructor, we define the layer we will use
        self.linear_layer = nn.Linear(in_features,out_features)

    def forward(self, x):
        # in the forward pass, we CONNECT the layers
        return self.linear_layer(x)

# instantiate the model
model = LinearRegressionModel(in_features=1, out_features=1)
print('Model Architecture:')
print(model)

import torch.optim as optim

#hyperparameter
learning_rate = 0.01

# create an ADAM optimizer
# we pass model.parameters() to tell it which tensor to manage
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# we'll also grab a pre-bult loss function from torch.nn
loss_fn = nn.MSELoss()      # Mean squared error loss


'''Three line MANTRA:
    1. optimizer.zero_grad()
    2. loss.backward()
    3. optimizer.step()'''

#final clean training loop
'''
epochs = 100
for epoch in range(epochs):
    ### FORWARD PASS ###
    y_hat = model(x)

    ### CALCULATE LOSS ###
    loss = loss_fn(y_hat, y_true)

    ### THREE LINE MANTRA ###
    #1. Zero the gradients
    optimizer.zero_grad()
    #2. Compute gradients
    loss.backward()
    #3. update the parameters
    optimizer.step()

    # optimizer print process
    if epoch % 10 == 0:
        print(f'Epoch {epoch:02d}: Loss: {loss.item():.4f}')
        '''

# The transformer feed-forward network

class FeedForwardNetwork(nn.Module):
    def __init__(self, emb_dim, ffn_dim):
        super().__init__()
        #in an LLM, emb_dim might be 4096, ffn_dim might be 14336

        self.layer1 = nn.Linear(emb_dim, ffn_dim)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(ffn_dim, emb_dim)

    def forward(self, x):
        # the data flow is exactly what you'll expect
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


'''
    A typical LLM:
    Model: Transformer
    Layer: nn.Linear (inside an FFN)
    Weight matrix "W" shape: (4096, 14336)
    Matrix multiplication: X @ W
    Total params: ~8 billions
'''

''' 5-steps logic is UNIVERSAL (even for LLM):
    y_hat = model(x)    (bunches of linear layers)
    loss = loss_fn(...)     (Cross-Entropy Loss)
    optimizer.zero_grad()   (identical)
    loss.backward()     (identical)
    optimizer.step()    (identical)
'''