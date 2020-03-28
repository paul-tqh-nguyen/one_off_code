#!/usr/bin/python3

import torch

batch_size, input_size, hidden_size, output_size = 3, 6, 4, 2

model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, output_size),
)

input = torch.rand(batch_size, input_size)

print(f'input {input}')
print(f'input.shape {input.shape}')
print(f'batch_size {batch_size}')
print(f'input_size {input_size}')

output = model(input)

print(f'output {output}')
print(f'output.shape {output.shape}')
