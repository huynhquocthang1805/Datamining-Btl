"""Copy the final checkpoints to models/ (seed chosen by VALIDATION loss, never test).

    python tools/export_models.py
"""
import json
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

PICK = {'losloop': [('main', 'gru'), ('main', 'tgcn'), ('improve', 'tgcn_res_time')],
        'sztaxi': [('main', 'gru'), ('main', 'tgcn'), ('improve', 'tgcn_res_time')]}


def main():
    out = ROOT / 'models'; out.mkdir(exist_ok=True)
    index = []
    for ds, items in PICK.items():
        for group, name in items:
            cands = sorted((ROOT / 'results' / ds / group / name).glob('seed_*/best.pt'))
            if not cands: continue
            best = min(cands, key=lambda p: torch.load(p, map_location='cpu', weights_only=True)['best_val_mse_scaled'])
            dst = out / f'{ds}_{name}.pt'
            shutil.copy2(best, dst)
            m = json.loads((best.parent / 'metrics.json').read_text())
            index.append({'file': dst.name, 'dataset': ds, 'model': name, 'source_run': str(best.parent.relative_to(ROOT)),
                          'test_RMSE_all_horizons': m['overall']['RMSE'], 'test_MAE_all_horizons': m['overall']['MAE']})
            print('exported', dst.name, '<-', best.parent.relative_to(ROOT))
    (out / 'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
