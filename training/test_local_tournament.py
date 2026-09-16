"""Bridge/export checks complementing the full-game validation and trace audit."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import unittest
import numpy as np
import torch
import run_local_tournament as arena

class LocalTournamentTests(unittest.TestCase):
    def test_exported_parameters_preserved(self):
        manifest=json.loads((arena.OUT/'manifest.json').read_text(encoding='utf-8'))
        tested=0
        for entry in manifest['entrants']:
            if not entry.get('exported_from'):continue
            source=torch.load(arena.ROOT/entry['exported_from'],map_location='cpu',weights_only=True)['state_dict']
            model=json.loads(Path(entry['path']).read_text(encoding='utf-8'))
            for json_key,prefix in [('layers','layers'),('stateLayers','state_layers'),('actionLayers','action_layers'),('valueLayers','value_layers')]:
                if json_key not in model:continue
                for i,layer in enumerate(model[json_key]):
                    for parameter in ('weight','bias'):
                        np.testing.assert_array_equal(np.asarray(layer[parameter],dtype=np.float32),source[f'{prefix}.{i}.{parameter}'].numpy())
            tested+=1
        self.assertEqual(tested,15)

    def test_serial_spawned_worker_same_actions(self):
        manifest=str(arena.OUT/'manifest.json')
        arena.initialize(manifest)
        task=('classic4',0,0,['alpha-2000','dirv-10000','ntw-champion','mcs'],60823001)
        serial=arena.run_block(task)
        with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),initializer=arena.initialize,initargs=(manifest,)) as executor:
            parallel=executor.submit(arena.run_block,task).result()
        for a,b in zip(serial,parallel):
            for key in ('actions','shares','bullheads','seats','deal_seed'):self.assertEqual(a[key],b[key])
        arena.BRIDGE.close()

if __name__=='__main__':unittest.main()
