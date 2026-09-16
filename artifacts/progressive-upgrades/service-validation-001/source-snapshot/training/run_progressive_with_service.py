"""Run a fresh frozen campaign manifest using a shared, sequential GPU service."""
import argparse
import json
import os
from pathlib import Path
import progressive_campaign as campaign
from progressive_inference_service import RemoteCampaignBridge

BASE_VERIFY=campaign.verify
SOURCES=['training/run_progressive_with_service.py','training/progressive_inference_service.py']


def initialize(path):
    campaign.arena.Bridge=RemoteCampaignBridge;campaign.arena.initialize(path)


def verify(manifest,folder):
    BASE_VERIFY(manifest,folder)
    for name in SOURCES:
        assert name in manifest['source_sha256'],'Create a fresh manifest that freezes the service transport'
        assert campaign.sha(campaign.ROOT/name)==manifest['source_sha256'][name],name


def main(args):
    manifest=Path(args.manifest).resolve();state=Path(args.service).resolve();service=campaign.load(state)
    assert service['status']=='ready' and campaign.sha(manifest) in service['manifests']
    assert service['source_sha256']==campaign.sha(campaign.ROOT/'training/progressive_inference_service.py')
    execution=dict(kind='single-gpu-service',sources={name:campaign.sha(campaign.ROOT/name) for name in SOURCES},
        policy='Original archived Planner.choose calls; authenticated local IPC only; decision time includes queue and transport',
        source_verification='Transport source hashes checked before games and again before result publication')
    record=manifest.parent/'execution.json'
    if record.exists():assert campaign.load(record)==execution
    else:campaign.dump(record,execution)
    os.environ['NTW_PROGRESSIVE_GPU_SERVICE']=str(state)
    campaign.initialize=initialize;campaign.verify=verify
    campaign.run(manifest,args.workers)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--service',required=True)
    parser.add_argument('--workers',type=int,default=8);main(parser.parse_args())
