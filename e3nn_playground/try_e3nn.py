import e3nn.o3 as o3
import e3nn.io as io
import e3nn
import torch
import plotly.graph_objects as go

irreps = o3.Irreps("5x0e + 10x1o")

for i, (mul, ir) in enumerate(irreps):
    print(mul)

s = io.SphericalTensor(10, +1, -1)  # 这像是创建了一个空壳，其实里面并没有数值
print(f's:{s.dim}')

pos = torch.tensor([  # 这里这个pos的模长其实是多少都无所谓，球谐函数这里只考虑方向
    [0.1, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])
val = torch.tensor([
    -0.5,
    2.0,
])

x = s.sum_of_diracs(pos, val)
print(f'x:{x.shape}')

print(s.signal_xyz(x, torch.eye(3)))

traces = s.plotly_surface(x)
# traces = s.plot(x)
traces = [go.Surface(**d) for d in traces]
fig = go.Figure(data=traces)
fig.show()

g = e3nn.nn.Gate("16x0o", [torch.tanh], "32x0o", [torch.tanh], "16x1e+16x1o")
print(f'irreps_in:{g.irreps_in}')
print(f'irreps_out:{g.irreps_out}')

print(o3.Irreps.spherical_harmonics(3).randn(-1).shape)