"""BF16 inference fusion; FMA disabled, original cast points explicit.

Torch pow/mean is retained verbatim: no alternate RMS reduction is substituted.
The Q/K path materializes normalized pre-RoPE keys because meta pooling uses them.
"""
import torch
import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib


@tr.jit
def norm_tail(X,V,W,Y,N:tl.constexpr,D:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 x=tl.load(X+i,i<N,0).to(tl.float32)
 v=tl.load(V+i//D,i<N,0)
 scale=lib.rsqrt(v+EPS)
 n=(x*scale).to(tl.bfloat16).to(tl.float32)
 w=tl.load(W+i%D,i<N,0).to(tl.float32)
 tl.store(Y+i,n*w,i<N)

_norm_launchers={}
_rope_launchers={}

def tail(x,variance,weight,eps,fast=False):
 y=torch.empty_like(x);n=x.numel()
 key=(x.device.index,n,x.shape[-1],eps)
 if fast and key in _norm_launchers:
  _norm_launchers[key](x,variance,weight,y)
 else:
  compiled=norm_tail[(tr.cdiv(n,256),)](x,variance,weight,y,n,x.shape[-1],eps,256,enable_fp_fusion=False)
  if fast:_norm_launchers[key]=compiled[(tr.cdiv(n,256),1,1)]
 return y

def rms(x,weight,eps):
 # EXACT original FP32 pow + torch.mean dispatch and shape.
 f=x.to(torch.float32);variance=f.pow(2).mean(-1,keepdim=True)
 return tail(x,variance,weight,eps,fast=True)


@tr.jit
def rope_kernel(X,C,S,Y,N:tl.constexpr,H:tl.constexpr,T:tl.constexpr,D:tl.constexpr,ST0:tl.constexpr,ST1:tl.constexpr,ST2:tl.constexpr,ST3:tl.constexpr,SY0:tl.constexpr,SY1:tl.constexpr,SY2:tl.constexpr,SY3:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);d=i%D;t=i//D%T;h=i//(D*T)%H;b=i//(D*T*H)
 ptr=b*ST0+h*ST1+t*ST2+d*ST3
 rd=tl.where(d<D//2,d+D//2,d-D//2)
 x=tl.load(X+ptr,i<N,0).to(tl.float32)
 rot=tl.load(X+b*ST0+h*ST1+t*ST2+rd*ST3,i<N,0).to(tl.float32)
 rot=tl.where(d<D//2,-rot,rot)
 c=tl.load(C+(b*T+t)*D+d,i<N,0).to(tl.float32);s=tl.load(S+(b*T+t)*D+d,i<N,0).to(tl.float32)
 a=(x*c).to(tl.bfloat16).to(tl.float32);z=(rot*s).to(tl.bfloat16).to(tl.float32)
 tl.store(Y+b*SY0+h*SY1+t*SY2+d*SY3,a+z,i<N)

def rope_one(x,cos,sin):
 b,h,t,d=x.shape
 # TensorIterator canonicalizes dense degenerate dimensions (e.g. T=1),
 # while preserving the physical order of non-contiguous dense inputs.
 y=torch.empty(x.shape,device=x.device,dtype=x.dtype) if x.is_contiguous() else torch.empty_like(x)
 n=x.numel()
 key=(x.device.index,n,h,t,d,x.stride(),y.stride())
 if key in _rope_launchers:
  _rope_launchers[key](x,cos,sin,y)
 else:
  compiled=rope_kernel[(tr.cdiv(n,256),)](x,cos,sin,y,n,h,t,d,*x.stride(),*y.stride(),256,enable_fp_fusion=False)
  _rope_launchers[key]=compiled[(tr.cdiv(n,256),1,1)]
 return y

def rope(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
 assert unsqueeze_dim==1 and cos.is_contiguous() and sin.is_contiguous()
 return rope_one(q,cos,sin),rope_one(k,cos,sin)




def enabled(x):
    return x.is_cuda and x.device.index == torch.cuda.current_device() and x.dtype == torch.bfloat16 and x.ndim >= 2 and x.shape[0] == 1 and not torch.is_grad_enabled()
