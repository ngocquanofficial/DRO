bz = 24 
particles = 4
lst = [i for i in range(bz)]
import torch
lst = torch.zeros((24, 2))
print(lst)
for i in range(particles) :
    print(lst[int(i/particles * bz): int( (i+1)/particles * bz)].shape)
