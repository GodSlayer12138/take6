"""Inspect a campaign runner; optionally stop its verified worker children.

The parent then exits through ProcessPoolExecutor's error path and closes its
trace normally. Resume the same manifest to change worker count without new deals.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
from datetime import datetime
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


class Entry(c.Structure):
    _fields_=[('size',w.DWORD),('usage',w.DWORD),('pid',w.DWORD),('heap',c.c_size_t),
        ('module',w.DWORD),('threads',w.DWORD),('parent',w.DWORD),('priority',w.LONG),
        ('flags',w.DWORD),('exe',w.WCHAR*260)]


class Unicode(c.Structure):
    _fields_=[('length',w.USHORT),('maximum',w.USHORT),('buffer',c.c_void_p)]


kernel=c.WinDLL('kernel32',use_last_error=True);native=c.WinDLL('ntdll')
kernel.CreateToolhelp32Snapshot.argtypes=[w.DWORD,w.DWORD];kernel.CreateToolhelp32Snapshot.restype=w.HANDLE
kernel.Process32FirstW.argtypes=[w.HANDLE,c.POINTER(Entry)];kernel.Process32NextW.argtypes=[w.HANDLE,c.POINTER(Entry)]
kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
kernel.CloseHandle.argtypes=[w.HANDLE];kernel.TerminateProcess.argtypes=[w.HANDLE,w.UINT]
kernel.GetExitCodeProcess.argtypes=[w.HANDLE,c.POINTER(w.DWORD)]
native.NtQueryInformationProcess.argtypes=[w.HANDLE,w.ULONG,c.c_void_p,w.ULONG,c.POINTER(w.ULONG)]
native.NtQueryInformationProcess.restype=w.LONG


def command(handle):
    size=w.ULONG();native.NtQueryInformationProcess(handle,60,None,0,c.byref(size))
    if not size.value:return ''
    buffer=c.create_string_buffer(size.value)
    if native.NtQueryInformationProcess(handle,60,buffer,size,c.byref(size))!=0:return ''
    value=Unicode.from_buffer(buffer);start=c.addressof(buffer)
    if not (start<=value.buffer and value.buffer+value.length<=start+len(buffer)):return ''
    return c.wstring_at(value.buffer,value.length//2)


def processes():
    handle=kernel.CreateToolhelp32Snapshot(2,0)
    if handle==c.c_void_p(-1).value:raise c.WinError(c.get_last_error())
    result=[];entry=Entry();entry.size=c.sizeof(entry)
    try:
        found=kernel.Process32FirstW(handle,c.byref(entry))
        while found:
            if entry.exe.lower()=='python.exe':
                process=kernel.OpenProcess(0x1000,False,entry.pid)
                if process:
                    try:result.append(dict(pid=entry.pid,parent=entry.parent,command=command(process)))
                    finally:kernel.CloseHandle(process)
            found=kernel.Process32NextW(handle,c.byref(entry))
    finally:kernel.CloseHandle(handle)
    return result


def main(args):
    if not re.fullmatch(r'(development|certificate)-\d{3}',args.run):raise ValueError('Only named campaign runners are supported')
    folder=ROOT/'artifacts/progressive-upgrades'/args.run
    if args.stop_workers:
        assert not (folder/'results.json').exists(),'Do not interrupt a completed evaluation'
        if args.run.startswith('certificate-'):
            state=json.loads((folder.parent/'state.json').read_text(encoding='utf-8'))
            assert state.get('pending_certificate',{}).get('run')==args.run,'Certificate must remain pending for exact-manifest resume'
    entries=processes();needle=f'artifacts/progressive-upgrades/{args.run}/manifest.json'
    roots=[r for r in entries if 'training/progressive_campaign.py' in r['command'].replace('\\','/') and needle in r['command'].replace('\\','/')]
    assert len(roots)==1,f'Expected exactly one matching campaign runner, found {len(roots)}'
    root=roots[0];marker=f'parent_pid={root["pid"]}'
    workers=[r for r in entries if r['parent']==root['pid'] and 'spawn_main' in r['command'] and marker in r['command'].replace(' ','')]
    report=dict(run=args.run,parent=root,workers=workers,action='inspect',time=datetime.now().astimezone().isoformat())
    if args.stop_workers:
        assert workers,'No verified worker children'
        report['action']='stop-workers-for-manifest-resume';report['stopped']=[]
        handles=[]
        try:
            for worker in workers:
                handle=kernel.OpenProcess(0x1001,False,worker['pid'])
                if not handle:raise c.WinError(c.get_last_error())
                handles.append((worker,handle))
                assert command(handle)==worker['command'],'Process changed during verification'
            for worker,handle in handles:
                status=w.DWORD();kernel.GetExitCodeProcess(handle,c.byref(status))
                if status.value==259 and not kernel.TerminateProcess(handle,130):
                    kernel.GetExitCodeProcess(handle,c.byref(status))
                    if status.value==259:raise c.WinError(c.get_last_error())
                report['stopped'].append(worker['pid'])
        finally:
            for _,handle in handles:kernel.CloseHandle(handle)
        with (ROOT/'artifacts/progressive-upgrades'/args.run/'worker-resume-events.jsonl').open('a',encoding='utf-8') as file:
            file.write(json.dumps(report)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--stop-workers',action='store_true')
    main(parser.parse_args())
