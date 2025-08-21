import torch
import AIRecon

model_path = './reconstruction/source/recon_model/0510x15.pth'
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
                    './0510x15_91layer_400c.onnx',
                    input_names=input_name,
                    output_names=output_name,
                    # dynamic_axes={'psf':{1:'depth'}},
                    opset_version=16)

# ./trtexec.exe --onnx=RUSH3D_20240530_ideal_dz1.onnx --saveEngine=RUSH3D_20240530_ideal_dz1.engine --minshapes=psf:81x61x2 --maxshapes=psf:81x73x2 --optShapes=psf:81x73x2
# trtexec --onnx=serenet_ly6g_1104.onnx --saveEngine=./reconstruction/source/recon_model/serenet_ly6g_1104_A100.engine
# trtexec --onnx=serenet_scanvoltage.onnx --saveEngine=./reconstruction/source/recon_model/serenet_scanvoltage_3090.engine
# trtexec --onnx=cbbx3.onnx --saveEngine=./reconstruction/source/recon_model/cbbx3.engine
# trtexec --onnx=0510x15_91layer_400c.onnx --saveEngine=./reconstruction/source/recon_model/0510x15_91layer_400c.engine