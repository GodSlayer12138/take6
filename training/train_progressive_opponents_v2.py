"""Broaden opponent imitation with the archived four-player league.

The original proxy data and split stay intact. Newly added full deals use the
same split hash; this remains development training, with fresh campaign tests.
"""
import hashlib
import json
import numpy as np
import torch
import train_progressive_opponents as base
from small_player_env import SmallGame

OUT=base.e.ROOT/'artifacts/progressive-upgrades/opponent-proxies-v2'


def dataset():
    path=OUT/'dataset.pt'
    if path.exists():return torch.load(path,map_location='cpu',weights_only=True)
    source=base.e.ROOT/'artifacts/local-tournament/classic4/blocks.jsonl'
    records=[]
    for line in source.read_text().splitlines():
        records.extend(r for r in json.loads(line) if any(k in r['seats'] for k in base.KINDS))
    values={k:[] for k in ('features','mask','action','kind','deal_seed')};p=4
    for offset in range(0,len(records),64):
        batch=records[offset:offset+64];b=len(batch)
        assert all(len(r['seats'])==4 and r['mode']=='classic4' for r in batch)
        cards=np.stack([np.random.default_rng(r['deal_seed']).permutation(104)+1 for r in batch])
        game=object.__new__(SmallGame);game.batch=b;game.players=p
        game.hands=np.sort(cards[:,:p*10].reshape(b,p,10),axis=-1)
        game.rows=np.zeros((b,4,6),dtype=np.int64);game.rows[:,:,0]=cards[:,-1:-5:-1]
        game.lengths=np.ones((b,4),dtype=np.int64);game.scores=np.zeros((b,p),dtype=np.float32)
        game.seen=np.zeros((b,104),dtype=bool);game.seen[np.arange(b)[:,None],game.rows[:,:,0]-1]=True
        owners=np.array([[base.KINDS.index(k) if k in base.KINDS else -1 for k in r['seats']] for r in batch]);take=owners>=0
        seeds=np.broadcast_to(np.array([r['deal_seed'] for r in batch])[:,None],(b,p))[take]
        for turn in range(10):
            played=np.array([r['actions'][turn] for r in batch])+1
            if turn<9:
                f=game.features()[take];h=10-turn;n=len(f)
                padded=np.zeros((n,10,270),dtype=np.float32);padded[:,:h]=f
                values['features'].append(padded);values['mask'].append(np.broadcast_to(np.arange(10)<h,(n,10)).copy())
                values['action'].append((game.hands==played[:,:,None]).argmax(axis=-1)[take])
                values['kind'].append(owners[take]);values['deal_seed'].append(seeds)
            game.step(played)
        np.testing.assert_array_equal(game.scores,np.array([r['bullheads'] for r in batch]))
        if offset%512==0:print(json.dumps(dict(phase='league-data',games=min(offset+64,len(records)))),flush=True)
    additions={k:torch.from_numpy(np.concatenate(v)) for k,v in values.items()}
    original_path=base.e.ROOT/'artifacts/progressive-upgrades/opponent-proxies-v1/dataset.pt'
    original=torch.load(original_path,map_location='cpu',weights_only=True)
    result={k:torch.cat((original[k],additions[k])) for k in additions}
    del original,additions,values
    unique=result['deal_seed'].unique().tolist()
    split={seed:int.from_bytes(hashlib.sha256(f'proxy-v1:{seed}'.encode()).digest()[:4],'little')%5==0 for seed in unique}
    result['validation']=torch.tensor([split[s] for s in result['deal_seed'].tolist()])
    torch.save(result,path)
    base.dump(OUT/'data-provenance.json',dict(original_dataset_sha256=base.e.sha(original_path),
        league_trace_sha256=base.e.sha(source),league_games=len(records),states=len(result['action']),unique_deals=len(unique),
        validation_states=int(result['validation'].sum()),training_validation_deals_disjoint=True,
        role='Historical four-player league repurposed as development; no campaign certificates used'))
    return result


if __name__=='__main__':
    base.OUT=OUT;OUT.mkdir(parents=True,exist_ok=True)
    base.CONFIG=dict(base.CONFIG,seed=90927001,extra_data='Archived classic4 league, all four target search opponents',
        extra_source_sha256=base.e.sha(__file__))
    base.dataset=dataset;base.main()
