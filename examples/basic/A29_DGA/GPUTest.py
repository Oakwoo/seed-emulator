import torch
flag = torch.cuda.is_available()
if flag:
    print("CUDA available")
else:
    print("CUDA unavailable")
    
ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Drive: ",device)
print("GPU Version: ",torch.cuda.get_device_name(0))

