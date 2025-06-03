import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.draw import disk
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sympy import symbols, And, Or, Not, simplify_logic
from itertools import product

# Your original BNN code (unchanged)
class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x.abs() <= 1).float()

def binarize(x):
    return SignSTE.apply(x)

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        bin_w = binarize(self.weight)
        return F.linear(x, bin_w, self.bias)

class XNORClassifier(nn.Module):
    def __init__(self, inference_mode=False):
        super().__init__()
        self.inference_mode = inference_mode
        self.fc1 = BinaryLinear(256, 128)
        self.fc2 = BinaryLinear(128, 64)
        self.fc3 = BinaryLinear(64, 1)

    def forward(self, x):
        x = binarize(x)
        x = binarize(self.fc1(x))
        x = binarize(self.fc2(x))
        x = self.fc3(x)
        if self.inference_mode:
            return x.sign()
        else:
            return torch.sigmoid(x)

class CanvasDataset(Dataset):
    def __init__(self, n=1000, size=16, fixed_shape=None):
        self.X = []
        self.y = []
        for _ in range(n):
            canvas = np.zeros((size, size), dtype=np.uint8)
            shape = fixed_shape or np.random.choice(["circle", "rect"])
            if shape == "circle":
                rr, cc = disk((size//2, size//2), size//3)
                canvas[rr, cc] = 1
                label = 1
            else:
                canvas[size//4:3*size//4, size//4:3*size//4] = 1
                label = 0
            self.X.append(canvas.flatten())
            self.y.append(label)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# NEW: BNN to Boolean Logic Converter
class BNNToLogic:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.model.inference_mode = True
        
    def extract_binary_weights(self):
        """Extract binarized weights from trained model"""
        weights = {}
        with torch.no_grad():
            # Layer 1: 256 -> 128
            w1 = torch.sign(self.model.fc1.weight).int()  # 128 x 256
            b1 = self.model.fc1.bias
            weights['layer1'] = {'weight': w1, 'bias': b1}
            
            # Layer 2: 128 -> 64  
            w2 = torch.sign(self.model.fc2.weight).int()  # 64 x 128
            b2 = self.model.fc2.bias
            weights['layer2'] = {'weight': w2, 'bias': b2}
            
            # Layer 3: 64 -> 1
            w3 = torch.sign(self.model.fc3.weight).int()  # 1 x 64
            b3 = self.model.fc3.bias
            weights['layer3'] = {'weight': w3, 'bias': b3}
            
        return weights
    
    def simulate_layer(self, inputs, weight, bias):
        """Simulate one binary layer using XNOR operations"""
        # inputs: binary tensor
        # weight: binary weight matrix
        outputs = []
        
        for neuron_weights in weight:
            # XNOR operation: ~(input XOR weight) 
            xnor_result = ~(inputs ^ neuron_weights)
            # Count 1s (popcount)
            popcount = xnor_result.sum().item()
            # Apply threshold (bias determines threshold)
            output = 1 if popcount > (len(inputs) / 2 + bias) else -1
            outputs.append(output)
            
        return torch.tensor(outputs, dtype=torch.int)
    
    def generate_truth_table(self, input_size=8):
        """Generate truth table for first 8 inputs (for demonstration)"""
        weights = self.extract_binary_weights()
        truth_table = {}
        
        print(f"Generating truth table for {input_size} inputs...")
        
        for i in range(2**input_size):
            # Create input pattern (only first input_size bits)
            input_pattern = torch.zeros(256, dtype=torch.int)
            for bit in range(input_size):
                input_pattern[bit] = 1 if (i >> (input_size-1-bit)) & 1 else -1
            
            # Forward pass through network
            x = input_pattern.float()
            with torch.no_grad():
                output = self.model(x.unsqueeze(0)).item()
                binary_output = 1 if output > 0 else 0
                
            truth_table[i] = binary_output
            
        return truth_table
    
    def create_boolean_expression(self, truth_table, input_size=8):
        """Convert truth table to simplified Boolean expression"""
        # Create symbolic variables
        vars = symbols(f'x0:{input_size}')
        
        # Find minterms where output is 1
        minterms = []
        for input_val, output in truth_table.items():
            if output == 1:
                # Create minterm for this input combination
                literals = []
                for bit in range(input_size):
                    bit_val = (input_val >> (input_size-1-bit)) & 1
                    if bit_val:
                        literals.append(vars[bit])
                    else:
                        literals.append(Not(vars[bit]))
                minterms.append(And(*literals))
        
        if not minterms:
            return False  # Always false
        
        # Combine minterms with OR
        boolean_expr = Or(*minterms)
        
        # Simplify the expression
        simplified = simplify_logic(boolean_expr, form='dnf')
        
        return simplified
    
    def count_gates(self, expr):
        """Count the number of logic gates in the expression"""
        if expr in [True, False]:
            return {"AND": 0, "OR": 0, "NOT": 0, "TOTAL": 0}
            
        counts = {"AND": 0, "OR": 0, "NOT": 0}
        
        def count_recursive(e):
            if isinstance(e, Or):
                counts["OR"] += len(e.args) - 1 if len(e.args) > 1 else 0
                for arg in e.args:
                    count_recursive(arg)
            elif isinstance(e, And):
                counts["AND"] += len(e.args) - 1 if len(e.args) > 1 else 0
                for arg in e.args:
                    count_recursive(arg)
            elif isinstance(e, Not):
                counts["NOT"] += 1
                count_recursive(e.args[0])
        
        count_recursive(expr)
        counts["TOTAL"] = sum(counts.values())
        return counts

def demonstrate_simplification():
    """Train a model and demonstrate Boolean simplification"""
    print("Training BNN...")
    torch.manual_seed(42)
    
    # Train the model
    ds = CanvasDataset(n=500)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = XNORClassifier(inference_mode=False)
    
    # Quick training
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    model.train()
    
    for epoch in range(10):  # Reduced epochs for demo
        for xb, yb in dl:
            xb = xb.float()
            yb = yb.float().unsqueeze(1)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    print("Converting to Boolean logic...")
    
    # Convert to Boolean logic
    converter = BNNToLogic(model)
    
    # Generate truth table for first 8 inputs
    truth_table = converter.generate_truth_table(input_size=8)
    
    # Create Boolean expression
    boolean_expr = converter.create_boolean_expression(truth_table, input_size=8)
    
    # Count gates
    gate_counts = converter.count_gates(boolean_expr)
    
    print(f"\n=== Results ===")
    print(f"Original BNN parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Truth table entries: {len(truth_table)}")
    print(f"Simplified Boolean expression:")
    print(f"  {boolean_expr}")
    print(f"\nGate counts:")
    for gate_type, count in gate_counts.items():
        print(f"  {gate_type}: {count}")
    
    # Test a few examples
    print(f"\n=== Verification ===")
    vars = symbols('x0:8')
    for test_val in [5, 10, 15, 20]:
        if test_val < 256:  # Within our truth table
            # Original model prediction
            test_input = torch.zeros(256)
            for bit in range(8):
                test_input[bit] = 1 if (test_val >> (7-bit)) & 1 else -1
            
            with torch.no_grad():
                model_output = model(test_input.unsqueeze(0)).item() > 0
            
            # Boolean expression evaluation
            bit_values = {vars[i]: bool((test_val >> (7-i)) & 1) for i in range(8)}
            bool_output = bool(boolean_expr.subs(bit_values))
            
            print(f"Input {test_val:3d}: Model={model_output}, Boolean={bool_output}, Match={model_output==bool_output}")
    
    return model, boolean_expr, gate_counts

if __name__ == "__main__":
    model, expr, counts = demonstrate_simplification()
