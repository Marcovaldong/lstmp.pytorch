import math
import torch
from torch.nn import init
import torch.jit as jit
from torch.nn import Parameter
# from torch.jit import Tensor  # there is an error
from torch import Tensor
from typing import List, Tuple

class LSTMPCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, projection_size):
        super(LSTMPCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, projection_size))
        self.weight_hr = Parameter(torch.randn(projection_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.init_weights()

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # input: batch_size * input_size
        # state: hx -> batch_size * projection_size 
        #        cx -> batch_size * hidden_size 
        # state cannot be None
        '''
        if state is not None:
            hx, cx = state
        else:
            hx = input.new_zeros(input.size(0), self.projection_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        '''
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih + 
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        hy = torch.mm(hy, self.weight_hr.t())

        return hy, (hy, cy)
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        init.uniform_(self.weight_ih, -stdv, stdv)
        init.uniform_(self.weight_hh, -stdv, stdv)
        init.uniform_(self.weight_hr, -stdv, stdv)
        init.uniform_(self.bias_ih)
        init.uniform_(self.bias_hh)

class LSTMPLayer(jit.ScriptModule):
    # def __init__(self, cell, *cell_args):
    def __init__(self, input_size, hidden_size, projection_size):
        super(LSTMPLayer, self).__init__()
        # self.cell = cell(*cell_args)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.cell = LSTMPCell(input_size=input_size, hidden_size=hidden_size, projection_size=projection_size)
    
    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # state cannot be None
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.init_weights()

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        init.uniform_(self.weight_ih, -stdv, stdv)
        init.uniform_(self.weight_hh, -stdv, stdv)
        init.uniform_(self.bias_ih)
        init.uniform_(self.bias_hh)
        
        

class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
    # def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)
        # self.cell = LSTMCell(input_size, hidden_size)
        # print('initial params of weight_ih: ')
        # print(self.cell.weight_ih)
        # print('initial params of weight_hh: ')
        # print(self.cell.weight_hh)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

def test():
    input_size = 320
    hidden_size = 768
    projection_size=256
    rnn = LSTMPLayer(input_size=input_size, hidden_size=hidden_size, projection_size=projection_size)
    x = torch.rand((50, 4, 320))
    hx = x.new_zeros(x.size(1), projection_size, requires_grad=False)
    cx = x.new_zeros(x.size(1), hidden_size, requires_grad=False)
    state = [hx, cx]
    y, h = rnn(x, state)