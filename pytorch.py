import torch

x = torch.empty(3,3)

y = torch.zeros(3,3)

z = torch.ones(3,3)

a = torch.rand(3,3)
b = torch.rand(3,3)
print(torch.add(a,b))
a.add_(b) # modify a

a - b
torch.sub(a,b) # div, mul
a.sub_(b)

c = torch.rand(5,5)
print(c[1,1].item())

# reshape, view() method
d = torch.rand(4,4)
e = d.view(2, -1)
print(e, e.size())

# torch to numpy
a = torch.ones(5)
d = a.numpy()
print(d)
print(torch.cuda.is_available())
