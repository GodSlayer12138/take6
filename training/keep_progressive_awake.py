"""Temporary AC-only idle-sleep request while one verified evaluation is alive.

No persistent power settings are changed. Windows releases the request when this
thread exits; display sleep and explicit user sleep commands remain available.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
from datetime import datetime
import json
import os
from pathlib import Path
import manage_progressive_workers as process


class Power(c.Structure):
    _fields_=[('ac',w.BYTE),('battery_flag',w.BYTE),('battery_percent',w.BYTE),('status',w.BYTE),
        ('remaining',w.DWORD),('full',w.DWORD)]


def main(args):
    root=Path(__file__).resolve().parents[1]
    manifest=Path(args.manifest).resolve()
    assert manifest.is_relative_to(root/'artifacts/progressive-upgrades')
    state=json.loads(Path(args.service).resolve().read_text(encoding='utf-8')) if args.service else None
    driver='training/run_progressive_inline.py' if args.inline else 'training/run_progressive_with_service.py'
    needle=str(manifest.relative_to(root)).replace('\\','/')
    runners=[p for p in process.processes() if driver in p['command'].replace('\\','/')
        and needle in p['command'].replace('\\','/')]
    assert len(runners)==1,'Expected exactly one verified live evaluation runner'
    kernel=process.kernel
    kernel.WaitForMultipleObjects.argtypes=[w.DWORD,c.POINTER(w.HANDLE),w.BOOL,w.DWORD]
    kernel.WaitForMultipleObjects.restype=w.DWORD
    kernel.SetThreadExecutionState.argtypes=[w.DWORD];kernel.SetThreadExecutionState.restype=w.DWORD
    kernel.GetSystemPowerStatus.argtypes=[c.POINTER(Power)];kernel.GetSystemPowerStatus.restype=w.BOOL
    handles=[];record=manifest.parent/'awake-request.json';active=None
    try:
        targets=[(runners[0]['pid'],runners[0]['command'])]
        if state:targets.append((state['pid'],'training/progressive_inference_service.py'))
        for pid,expected in targets:
            handle=kernel.OpenProcess(0x101000,False,pid)
            if not handle:raise c.WinError(c.get_last_error())
            handles.append(handle);assert expected in process.command(handle),'Process identity changed'
        array=(w.HANDLE*len(handles))(*handles)
        while kernel.WaitForMultipleObjects(len(handles),array,False,0)==258:
            power=Power();plugged=bool(kernel.GetSystemPowerStatus(c.byref(power))) and power.ac==1
            if active!=plugged:
                assert kernel.SetThreadExecutionState(0x80000001 if plugged else 0x80000000)
                active=plugged
                value=dict(status='active' if active else 'released-on-battery',helper_pid=os.getpid(),
                    runner_pid=runners[0]['pid'],service_pid=state['pid'] if state else None,time=datetime.now().astimezone().isoformat(),
                    scope='Temporary AC-only system-idle request; no persistent power-plan changes')
                record.write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value),flush=True)
            result=kernel.WaitForMultipleObjects(len(handles),array,False,30000)
            if result!=258:
                if result==0xffffffff:raise c.WinError(c.get_last_error())
                break
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        for handle in handles:kernel.CloseHandle(handle)
        value=dict(status='released',helper_pid=os.getpid(),time=datetime.now().astimezone().isoformat())
        record.write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--service');mode.add_argument('--inline',action='store_true')
    main(parser.parse_args())
