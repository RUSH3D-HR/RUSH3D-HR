import torch
import AIRecon

# First we need to load the pytorch model and export it into onnx format

model_path = './reconstruction/source/recon_model/default.pth'
model_weight = torch.load(model_path)['model']
model = AIRecon.make(model_weight, load_sd=True).cuda()
model.eval()

input_name = ['input','psf', 'transition']
output_name = ['output']
input = torch.rand(81, 477, 477).cuda()#1,v,d,h, w
psf = torch.rand(81,91,2).cuda()
alpha = torch.linspace(0, 1, 50)
transition = (1 / (1 + torch.exp(-10 * (alpha - 0.5)))).cuda()
with torch.no_grad():
    torch.onnx.export(model,
                    (input, psf, transition),
                    './default.onnx',
                    input_names=input_name,
                    output_names=output_name,
                    # dynamic_axes={'psf':{1:'depth'}},
                    opset_version=16)
    
# Then we can use trtexec to convert the onnx model into tensorrt engine

# trtexec --onnx=default.onnx --saveEngine=./reconstruction/source/recon_model/default.engine


# Finally, enable it by settting "recon_trt" to True in the RCconfig file. 
# NOTICE: The relative import and methods were commented out in 'recon_torch.py' for easier demo configuration.
# Run without enable it will raise NotImplementedError. Please uncomment the relative import and methods in 'recon_torch.py' to enable it.