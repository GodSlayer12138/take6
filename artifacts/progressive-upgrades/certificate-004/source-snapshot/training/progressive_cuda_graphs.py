"""Optional execution accelerator; captured operations preserve the original policy.

Graphs share a pool within one thread and clone outputs before the next graph.
Inputs use reusable external buffers. This module does not install itself or alter
any running/frozen campaign. Call install explicitly in a separate experiment.
"""
from collections import Counter
import hashlib
import inspect
import threading
import textwrap
import torch

CAPTURE_LOCK=threading.Lock()


class GraphCache:
    def __init__(self,device,max_batch=32768):
        self.device=device;self.max_batch=max_batch;self.graphs={};self.stats=Counter()
        self.pool=torch.cuda.graph_pool_handle();self.buffers={};self.models={}

    def model_signature(self,model):
        parameters=tuple(model.parameters());versions=tuple(p._version for p in parameters)
        if model in self.models:
            previous,digest=self.models[model]
            assert previous==versions,'Captured inference weights must remain frozen'
            return digest
        digest=hashlib.sha256()
        for parameter in parameters:
            digest.update(str((tuple(parameter.shape),parameter.dtype)).encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        signature=digest.hexdigest();self.models[model]=(versions,signature)
        return signature

    def buffer(self,name,size,dtype):
        if name not in self.buffers:
            self.buffers[name]=torch.zeros(size,device=self.device,dtype=dtype)
        assert self.buffers[name].numel()>=size
        return self.buffers[name]

    def execute(self,key,inputs,operation):
        # Each operation's inputs are views of external reusable buffers.
        if key not in self.graphs:
            with CAPTURE_LOCK:
                stream=torch.cuda.current_stream(self.device)
                warm=torch.cuda.Stream(device=self.device);warm.wait_stream(stream)
                with torch.cuda.stream(warm):
                    for _ in range(3):sample=operation()
                stream.wait_stream(warm)
                capacities={(0,torch.float32):self.max_batch*4*252,
                    (1,torch.float32):self.max_batch*4*10*18,
                    (1,torch.int64):self.max_batch*4*10,
                    (2,torch.int64):self.max_batch*4*6,
                    (3,torch.int64):self.max_batch*4,
                    (4,torch.float32):self.max_batch*4,
                    (5,torch.bool):self.max_batch*104}
                outputs=tuple(self.buffer(f'output-{i}-{v.dtype}',capacities[(i,v.dtype)],v.dtype)[:v.numel()].view(v.shape)
                    for i,v in enumerate(sample))
                del sample
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,pool=self.pool,capture_error_mode='relaxed'):
                    captured=operation()
                    for i in range(len(outputs)):outputs[i].copy_(captured[i])
                    del captured
                # Keep captured constants, index tensors and model weights alive.
                self.graphs[key]=(graph,outputs,operation)
                self.stats['captures']+=1
        graph,outputs,_=self.graphs[key];graph.replay();self.stats['replays']+=1
        # Pool storage can be reused by a different graph on the next invocation.
        return tuple(x.clone() for x in outputs)


def install(module,max_batch=32768):
    """Wrap one already loaded Torch environment; never change archived files."""
    if getattr(module,'_cuda_graphs_installed',False):return module._cuda_graph_stats
    local=threading.local();caches=[]
    original_features=module.TorchGame.features;original_step=module.TorchGame.step
    original_utilities=module.split_utilities
    # Hoist the original four-element CPU-to-GPU constant outside capture.
    # All feature arithmetic remains the archived function's exact source.
    source=textwrap.dedent(inspect.getsource(original_features))
    line="globals_=torch.tensor([1,p/10,h/10,(10-h)/10],dtype=torch.float32,device=self.device)"
    assert source.count(line)==1,'Review a changed feature implementation before capture'
    namespace=dict(original_features.__globals__)
    exec(compile(source.replace(line,'globals_=self._graph_globals'),'<captured TorchGame.features>','exec'),namespace)
    captured_features=namespace['features']

    def cache_for(device):
        if not hasattr(local,'cache'):
            local.cache=GraphCache(device,max_batch);caches.append(local.cache)
        return local.cache

    def table_inputs(table,cache):
        b,p,h=table.hands.shape;shape={'hands':(b,p,h),'rows':(b,4,6),'lengths':(b,4),'scores':(b,p),'seen':(b,104)}
        sizes={'hands':max_batch*4*10,'rows':max_batch*4*6,'lengths':max_batch*4,'scores':max_batch*4,'seen':max_batch*104}
        result=object.__new__(module.TorchGame)
        result.device=table.device;result.batch=b;result.players=p;result.points=table.points
        result.batch_ids=table.batch_ids;result.row_ids=table.row_ids
        for name,dims in shape.items():
            source=getattr(table,name);count=source.numel()
            view=cache.buffer('table-'+name,sizes[name],source.dtype)[:count].view(dims)
            view.copy_(source);setattr(result,name,view)
        return result

    @torch.inference_mode()
    def features(table):
        if table.device.type!='cuda' or table.batch>max_batch:return original_features(table)
        cache=cache_for(table.device);static=table_inputs(table,cache)
        _,p,h=table.hands.shape
        static._graph_globals=torch.tensor([1,p/10,h/10,(10-h)/10],dtype=torch.float32,device=table.device)
        return cache.execute(('features',tuple(table.hands.shape)),None,lambda:captured_features(static))

    @torch.inference_mode()
    def step(table,cards):
        if table.device.type!='cuda' or table.batch>max_batch:return original_step(table,cards)
        cache=cache_for(table.device);static=table_inputs(table,cache);shape=tuple(table.hands.shape)
        fixed_cards=cache.buffer('cards',max_batch*4,cards.dtype)[:cards.numel()].view(cards.shape);fixed_cards.copy_(cards)
        # step mutates its table, so warm-up and capture each start from fresh clones.
        def operation():
            temp=object.__new__(module.TorchGame)
            for name in ('device','batch','players','points','batch_ids','row_ids'):setattr(temp,name,getattr(static,name))
            for name in ('hands','rows','lengths','scores','seen'):setattr(temp,name,getattr(static,name).clone())
            costs=original_step(temp,fixed_cards)
            return (costs,)+(tuple(getattr(temp,name) for name in ('hands','rows','lengths','scores','seen')))
        outputs=cache.execute(('step',shape),None,operation)
        for name,value in zip(('hands','rows','lengths','scores','seen'),outputs[1:]):setattr(table,name,value)
        return outputs[0]

    @torch.inference_mode()
    def utilities(model,common,action):
        n,h=action.shape[:2];rounded=((n+63)//64)*64
        if common.device.type!='cuda' or rounded>max_batch:return original_utilities(model,common,action)
        cache=cache_for(common.device)
        c=cache.buffer('neural-common',max_batch*252,common.dtype)[:rounded*252].view(rounded,252)
        a=cache.buffer('neural-action',max_batch*10*18,action.dtype)[:rounded*h*18].view(rounded,h,18)
        c[:n].copy_(common);a[:n].copy_(action)
        value,=cache.execute(('network',cache.model_signature(model),rounded,h),None,lambda:(original_utilities(model,c,a),))
        return value[:n]

    module.TorchGame.features=features;module.TorchGame.step=step;module.split_utilities=utilities
    module._cuda_graphs_installed=True
    module._cuda_graph_stats=lambda:[dict(c.stats,cache_entries=len(c.graphs)) for c in caches]
    return module._cuda_graph_stats
