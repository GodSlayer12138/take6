"""One GPU context serving frozen planners over authenticated local Windows IPC.

Inference stays sequential and calls the original archived Planner.choose method.
Only transport changes: clients send exactly their public observation and policy ID.
"""
import argparse
from concurrent.futures import Future
import hashlib
import importlib.util
import json
from multiprocessing.connection import Client,Listener
import os
from pathlib import Path
import queue
import sys
import threading
import time
import uuid
from evaluate_small_exploration import ExplorationBridge


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RemoteCampaignBridge(ExplorationBridge):
    def __init__(self,path):
        manifest=load(path);self.planner_ids={e['id'] for e in manifest['entrants'] if e['kind']=='planner'}
        self.manifest_sha=sha(path);state=load(os.environ['NTW_PROGRESSIVE_GPU_SERVICE'])
        assert self.manifest_sha in state['manifests'],'Service must load this exact manifest'
        self.connection=Client(state['address'],family='AF_PIPE',authkey=bytes.fromhex(state['authkey']))
        super().__init__(path)

    def planner_call(self,policy,observation):
        started=time.perf_counter()
        self.connection.send(dict(type='choose',manifest=self.manifest_sha,policy=policy,observation=observation))
        response=self.connection.recv()
        if 'error' in response:raise RuntimeError(response['error'])
        result=response['result'];result['compute_seconds']=result['seconds'];result['seconds']=time.perf_counter()-started
        return result

    def call(self,request):
        if request['type']!='actions':return super().call(request)
        results=[None]*len(request['items']);remote=[];indices=[]
        for i,item in enumerate(request['items']):
            if item['id'] in self.planner_ids:
                r=self.planner_call(item['id'],item['observation']);results[i]={'card':r['card'],'seconds':r['seconds']}
            else:remote.append(item);indices.append(i)
        if remote:
            for i,result in zip(indices,super().call(dict(type='actions',items=remote))):results[i]=result
        return results

    def close(self):
        try:self.connection.close()
        finally:super().close()


def serve(args):
    import progressive_campaign as campaign
    import torch
    torch.set_num_threads(1);models={};manifests={}
    for name in args.manifest:
        path=Path(name).resolve();manifest=load(path);campaign.verify(manifest,path.parent);digest=sha(path)
        manifests[digest]=str(path)
        for entry in manifest['entrants']:
            if entry['kind']!='planner':continue
            module_name='service_planner_'+sha(entry['runtime'])[:16]
            if module_name not in sys.modules:
                spec=importlib.util.spec_from_file_location(module_name,entry['runtime']);module=importlib.util.module_from_spec(spec)
                sys.modules[module_name]=module;spec.loader.exec_module(module)
            models[(digest,entry['id'])]=sys.modules[module_name].Planner(load(entry['path'])['params'],'cpu')
    address=r'\\.\pipe\ntw-progressive-'+uuid.uuid4().hex;authkey=os.urandom(32)
    listener=Listener(address,family='AF_PIPE',authkey=authkey);requests=queue.Queue()
    def receive(connection):
        try:
            while True:
                request=connection.recv();future=Future();future.sent=threading.Event();requests.put((request,future))
                connection.send(future.result())
                future.sent.set()
                if request.get('type')=='shutdown':return
        except (EOFError,BrokenPipeError,OSError):pass
        finally:connection.close()
    def accept():
        while True:
            try:connection=listener.accept()
            except OSError:return
            threading.Thread(target=receive,args=(connection,),daemon=True).start()
    threading.Thread(target=accept,daemon=True).start()
    state_path=Path(args.state).resolve();state_path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(pid=os.getpid(),address=address,authkey=authkey.hex(),manifests=manifests,source_sha256=sha(__file__),
        status='ready',information='Policy ID and own public observation only; archived inference method unchanged')
    state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='ready',state=str(state_path),policies=len(models))),flush=True)
    calls=0
    try:
        while True:
            request,future=requests.get()
            if request.get('type')=='shutdown':
                future.set_result(dict(stopped=True));future.sent.wait(timeout=3);break
            try:
                assert request['type']=='choose'
                result=models[(request['manifest'],request['policy'])].choose(request['observation'])
                future.set_result(dict(result=result));calls+=1
            except Exception as error:future.set_result(dict(error=f'{type(error).__name__}: {error}'))
    finally:
        listener.close();state.update(status='stopped',calls=calls);state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')


def stop(path):
    state=load(path)
    with Client(state['address'],family='AF_PIPE',authkey=bytes.fromhex(state['authkey'])) as connection:
        connection.send(dict(type='shutdown'));print(json.dumps(connection.recv()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',action='append');parser.add_argument('--state',required=True)
    parser.add_argument('--stop',action='store_true');args=parser.parse_args()
    if args.stop:stop(args.state)
    else:serve(args)
