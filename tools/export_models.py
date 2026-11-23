"""Copy trained checkpoints to models/.

* comparison models: the seed with the lowest VALIDATION loss (never test)
* final model (chosen among configurations by validation RMSE, see report 5.5):
  all 3 seeds of GRU-RT (Los-loop) / GRU (SZ-taxi), used together as an ensemble by predict.py

    python tools/export_models.py
"""
import json
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

SINGLE = {'losloop': [('main', 'gru'), ('main', 'tgcn'), ('improve', 'tgcn_selfsep'), ('improve', 'tgcn_res_time')],
          'sztaxi': [('main', 'tgcn')]}
FINAL = {'losloop': ('improve', 'gru_res_time'), 'sztaxi': ('main', 'gru')}   # chosen by validation RMSE


def runs(ds, group, name):
    out = sorted((ROOT / 'results' / ds / group / name).glob('seed_*/best.pt'))
    if group == 'improve':  # seed 42 of some variants lives in variants/
        out += sorted((ROOT / 'results' / ds / 'variants' / name).glob('seed_*/best.pt'))
    return [p for p in out if (p.parent / 'metrics.json').is_file()]   # finished runs only


def main():
    out = ROOT / 'models'; out.mkdir(exist_ok=True)
    index = []
    for ds in SINGLE:
        for group, name in SINGLE[ds]:
            c = runs(ds, group, name)
            if not c: continue
            best = min(c, key=lambda p: torch.load(p, map_location='cpu', weights_only=True)['best_val_mse_scaled'])
            dst = out / f'{ds}_{name}.pt'; shutil.copy2(best, dst)
            m = json.loads((best.parent / 'metrics.json').read_text())
            index.append({'file': dst.name, 'role': 'comparison (best-validation seed)', 'dataset': ds, 'model': name,
                          'source_run': str(best.parent.relative_to(ROOT)),
                          'test_RMSE_all_horizons': m['overall']['RMSE'], 'test_MAE_all_horizons': m['overall']['MAE']})
            print('exported', dst.name, '<-', best.parent.relative_to(ROOT))
        group, name = FINAL[ds]
        for c in runs(ds, group, name):
            seed = c.parent.name.split('_')[-1]
            dst = out / f'{ds}_{name}_seed{seed}.pt'; shutil.copy2(c, dst)
            m = json.loads((c.parent / 'metrics.json').read_text())
            index.append({'file': dst.name, 'role': 'FINAL model, ensemble member', 'dataset': ds, 'model': name,
                          'source_run': str(c.parent.relative_to(ROOT)),
                          'test_RMSE_all_horizons': m['overall']['RMSE'], 'test_MAE_all_horizons': m['overall']['MAE']})
            print('exported', dst.name)
    (out / 'index.json').write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
