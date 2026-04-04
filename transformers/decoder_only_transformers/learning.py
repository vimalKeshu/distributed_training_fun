import torch # create tensors and provides helper functions
import torch.nn as nn # for nn.Module(), nn.Embedding() and nn.Linear()
import torch.nn.functional as F # gives us the softmax() and argmax()
from torch.optim import Adam # Adam optimizer, stochastic gradient descent
from torch.utils.data import TensorDataset, DataLoader # for storing data loader

# first, create a dict that maps vocabulary tokens to id numbers 
token_to_id = ({
    'what': 0,
    'is': 1,
    'your': 2,
    'name': 3,
    'gpt': 4,
    'my': 5,
    '<EOS>': 10, # END OF SEQUENCE
    '<PAD>': 11, # PADDING 
})

## create the dict that maps the ids to tokens, for interpretintg the model output.
id_to_token = dict(map(reversed, token_to_id.items()))
VOCAB_SIZE = len(token_to_id)
SEQ_LEN = 6
D_MODEL = 2
# we use decoder only transformer, the inputs contain
# the questions followed by <EOS> token followed by the response 'gpt'
# this is because all of the tokens will be used as inputs to the decoder only
# transformer during training. 
# it's called teacher forcing 
# teacher forcing helps us train the neural network faster 

inputs = torch.tensor([
    [
        token_to_id['what'],
        token_to_id['is'],
        token_to_id['your'],
        token_to_id['name'],
    ],
    [
        token_to_id['gpt'],
        token_to_id['is'],
        token_to_id['my'],
    ]
])

# we are using decoder only transformer the outputs, or
# the predictions, are the input questions (minus the first word) followed by
# <EOS> gpt <EOS>. the first <EOS> means we are dong processing the input question
# and the second means we are done generating the output.
labels = torch.tensor([
    [
        token_to_id['is'],
        token_to_id['your'],
        token_to_id['name'],
        token_to_id['<EOS>'],
        token_to_id['gpt'],
        token_to_id['<EOS>'],
    ],
    [
        token_to_id['is'],
        token_to_id['my'],
        token_to_id['<EOS>'],
        token_to_id['name'],
        token_to_id['<EOS>'],
        token_to_id['<PAD>'],
    ]
])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset=dataset)

print(f'Shape of the input: {inputs.shape}')
print(f'Shape of the labels: {labels.shape}')

x = inputs.unsqueeze(0)
y = labels.unsqueeze(0)

print(f'Batch input: {x.shape}')
print(f'Batch labels: {y.shape}')