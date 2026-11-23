"""Assemble reports/BAO_CAO_TGCN.md from the draft parts, filling every numeric
table from reports/tables/*.csv so that no number is copied by hand.

    python tools/build_report.py && python tools/md2docx.py reports/BAO_CAO_TGCN.md reports/BAO_CAO_TGCN.docx
"""
import json
import re
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from tgcn.plotstyle import label

REP = ROOT / 'reports'; TAB = REP / 'tables'
PARTS = ['_draft_part0.md', '_draft_part1.md', '_draft_part2.md', '_draft_part_critique.md',
         '_draft_part3.md', '_draft_part_model.md', '_draft_part4.md', '_draft_part5.md', '_draft_part6.md']
SECTION_ORDER = None


def md(df, index=False):
    cols = ([df.index.name or ''] if index else []) + list(map(str, df.columns))
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for i, r in df.iterrows():
        vals = ([str(i)] if index else []) + [str(v) for v in r.values]
        out.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(out)


def f3(x):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.3f}'


def peak_table():
    df = pd.read_csv(TAB / 'peak_subsets_losloop.csv')
    order = ['naive', 'linear', 'gru', 'gcn', 'tgcn', 'tgcn_selfsep', 'gru_res_time', 'tgcn_res_time']
    cols = ['tất cả', 'ngoài cao điểm', 'cao điểm', 'thông thoáng', 'kẹt xe (v<40)', 'biến động mạnh (Δ>10)']
    rows = []
    for m in order:
        s = df[df.model == m]
        if s.empty: continue
        r = {'Mô hình': label(m)}
        for h in ('15 phút', '60 phút'):
            for c in cols:
                v = s[(s.horizon == h) & (s.subset == c)]['MAE']
                r[f'{h}: {c}'] = f3(float(v.iloc[0])) if len(v) else '—'
        rows.append(r)
    t = pd.DataFrame(rows)
    share = df[(df.model == 'naive') & (df.horizon == '15 phút')].set_index('subset')['share_%']
    note = ('*Tỉ lệ mẫu test: ' + ', '.join(f'{c} {share.get(c, np.nan):.0f}%' for c in cols[1:]) + '.*')
    a = t[['Mô hình'] + [f'15 phút: {c}' for c in cols]].rename(columns=lambda c: c.replace('15 phút: ', ''))
    b = t[['Mô hình'] + [f'60 phút: {c}' for c in cols]].rename(columns=lambda c: c.replace('60 phút: ', ''))
    return '**MAE – dự báo trước 15 phút**\n\n' + md(a) + '\n\n**MAE – dự báo trước 60 phút**\n\n' + md(b) + '\n\n' + note


def ablation_table():
    df = pd.read_csv(TAB / 'ablation_losloop.csv')
    t = pd.DataFrame({'Cấu hình (seed 42)': df.config,
                      'RMSE TB': df.RMSE_all.map(f3), 'MAE TB': df.MAE_all.map(f3),
                      '15 phút': [f'{a:.2f} / {b:.2f}' for a, b in zip(df.RMSE_15min, df.MAE_15min)],
                      '60 phút': [f'{a:.2f} / {b:.2f}' for a, b in zip(df.RMSE_60min, df.MAE_60min)],
                      'ΔRMSE vs T-GCN': df['ΔRMSE_all_vs_TGCN_%'].map(lambda v: f'{v:+.1f}%'),
                      'ΔMAE vs T-GCN': df['ΔMAE_all_vs_TGCN_%'].map(lambda v: f'{v:+.1f}%'),
                      'epoch chọn': df.best_epoch})
    return md(t)


def tuning_table():
    df = pd.read_csv(TAB / 'tuning_losloop.csv')
    names = {'T-GCN L=12, h=64 (mặc định)': 'T-GCN, L = 12, hidden = 64 (mặc định)',
             'tgcn_seq6': 'T-GCN, L = 6 (30 phút)', 'tgcn_seq24': 'T-GCN, L = 24 (120 phút)',
             'tgcn_h32': 'T-GCN, hidden = 32', 'tgcn_h100': 'T-GCN, hidden = 100',
             'tgcn_res_time L=12, h=64': 'T-GCN-RT, L = 12', 'tgcn_res_time_seq6': 'T-GCN-RT, L = 6',
             'tgcn_res_time_seq24': 'T-GCN-RT, L = 24'}
    order = list(names)
    df['k'] = df.config.map(lambda c: order.index(c) if c in order else 99)
    df = df.sort_values('k')
    t = pd.DataFrame({'Cấu hình (seed 42)': df.config.map(lambda c: names.get(c, c)),
                      'tham số': df.n_params.map(lambda v: '—' if pd.isna(v) else f'{int(v):,}'),
                      'RMSE TB': df.RMSE_all.map(f3), 'MAE TB': df.MAE_all.map(f3),
                      '15 phút': [f'{a:.2f} / {b:.2f}' for a, b in zip(df.RMSE_15min, df.MAE_15min)],
                      '60 phút': [f'{a:.2f} / {b:.2f}' for a, b in zip(df.RMSE_60min, df.MAE_60min)],
                      'epoch chọn': df.best_epoch,
                      'thời gian train (phút)': df.train_s.map(lambda v: '—' if pd.isna(v) else f'{v / 60:.0f}')})
    return md(t)


def main_table(ds='losloop', masked=False):
    df = pd.read_csv(TAB / f'main_results_{ds}{"_masked" if masked else ""}.csv')
    keep = [m for m in df.model if not m.startswith('ensemble') or m in
            ('ensemble_gru_res_time', 'ensemble_gru_res_time_rawloss', 'ensemble_mix', 'ensemble_tgcn_selfsep')]
    df = df[df.model.isin(keep)]
    t = pd.DataFrame({'Mô hình': df.model.map(label), 'seed': df.n_seeds,
                      'RMSE val (chọn)': df['VAL_RMSE (chọn mô hình)'],
                      'RMSE TB': df.overall_RMSE, 'MAE TB': df.overall_MAE,
                      '15 phút RMSE': df['15min_RMSE'], '15 phút MAE': df['15min_MAE'],
                      '60 phút RMSE': df['60min_RMSE'], '60 phút MAE': df['60min_MAE']})
    return md(t)


def fi_table():
    df = pd.read_csv(TAB / 'feature_importance_losloop.csv')
    kinds = ['lịch sử của chính nút', 'láng giềng 1-hop', 'nút ngẫu nhiên xa (≥4 hop)', 'đặc trưng thời gian (giờ, thứ)']
    rows = []
    for m in dict.fromkeys(df.model):
        s = df[df.model == m].set_index('feature')
        lag = df[(df.model == m) & (df.group == 'lag')].set_index('lag')['ΔRMSE']
        r = {'Mô hình': label(m), 'RMSE gốc': f3(s.RMSE_base.iloc[0]),
             'lag 5 phút': f3(lag.get(1.0, np.nan)), 'lag 15 phút': f3(lag.get(3.0, np.nan)),
             'lag 60 phút': f3(lag.get(12.0, np.nan))}
        for k in kinds:
            r[k] = f3(s['ΔRMSE'].get(k, np.nan)) if k in s.index else '—'
        rows.append(r)
    return '*ΔRMSE = RMSE tăng thêm khi xáo trộn nhóm đầu vào (cột không gian: RMSE tại nút đích, TB 40 nút).*\n\n' + md(pd.DataFrame(rows))


def seed_table():
    df = pd.read_csv(TAB / 'rel_seeds_losloop.csv')
    t = pd.DataFrame({'Mô hình': df.model.map(label), 'seed': df.seeds,
                      'RMSE': [f'{a:.3f} ± {b:.3f}' for a, b in zip(df.RMSE_mean, df.RMSE_std)],
                      'CV RMSE': df['RMSE_CV_%'].map(lambda v: f'{v:.2f}%'),
                      'MAE': [f'{a:.3f} ± {b:.3f}' for a, b in zip(df.MAE_mean, df.MAE_std)],
                      'epoch chọn': df.best_epochs})
    return md(t)


def pairwise_table():
    df = pd.read_csv(TAB / 'rel_pairwise_losloop.csv')
    t = pd.DataFrame({'A': df.A.map(label), 'B': df.B.map(label),
                      'RMSE A': df.RMSE_A.map(f3), 'RMSE B': df.RMSE_B.map(f3),
                      'ΔRMSE (A−B)': df['ΔRMSE (A−B)'].map(lambda v: f'{v:+.3f}'),
                      'KTC 95%': [f'[{a:+.3f}; {b:+.3f}]' for a, b in zip(df.CI95_low, df.CI95_high)],
                      'DM p': df.DM_p.map(lambda v: f'{v:.4f}' if v >= 1e-4 else '< 0.0001'),
                      'Kết luận': [('A tốt hơn' if d < 0 else 'B tốt hơn') if p < 0.05 else 'không khác biệt'
                                   for d, p in zip(df['ΔRMSE (A−B)'], df.DM_p)]})
    return md(t)


def conformal_table():
    df = pd.read_csv(TAB / 'rel_conformal_losloop.csv')
    df = df[df['mode'] == 'per_horizon']
    t = pd.DataFrame({'Mô hình': df.model.map(label), 'tầm': df.horizon_min.map(lambda v: f'{v} phút'),
                      'độ phủ tổng (%)': df['coverage_all_%'].map(lambda v: f'{v:.1f}'),
                      'cao điểm (%)': df['coverage_peak_%'].map(lambda v: f'{v:.1f}'),
                      'ngoài cao điểm (%)': df['coverage_offpeak_%'].map(lambda v: f'{v:.1f}'),
                      'độ rộng TB (mph)': df.mean_width.map(lambda v: f'{v:.1f}')})
    return md(t)


def paper_table():
    df = pd.read_csv(TAB / 'paper_comparison_losloop.csv')
    names = ['ha_window', 'linear', 'gcn', 'gru', 'tgcn', 'naive', 'gru_res_time', 'ensemble_gru_res_time']
    rows = []
    for m in names:
        s = df[df.model == m]
        if s.empty: continue
        for who in ('paper', 'ours'):
            if f'{who}_RMSE' not in s or s[f'{who}_RMSE'].isna().all(): continue
            r = {'Mô hình': label(m) + (' (SVR)' if m == 'linear' and who == 'paper' else ''),
                 'nguồn': 'bài báo' if who == 'paper' else 'chúng tôi'}
            for _, x in s.iterrows():
                r[x.horizon.replace('min', ' phút')] = (f"{x[f'{who}_RMSE']:.2f} / {x[f'{who}_MAE']:.2f} / "
                                                         f"{x[f'{who}_Accuracy']:.3f} / {x[f'{who}_R2']:.3f}")
            rows.append(r)
    return md(pd.DataFrame(rows))


def model_card():
    df = pd.read_csv(TAB / 'model_card_tgcn_res_time_losloop.csv')
    t = pd.DataFrame({'Tensor': df.tensor.map(lambda s: f'`{s}`'), 'Kích thước': df['shape'],
                      'Số phần tử': df.n.map(lambda v: f'{v:,}'), 'Học?': df.learned.map(lambda v: 'có' if v else 'không'),
                      'Vai trò': df.role})
    return md(t) + f"\n\nTổng số tham số học được: **{int(df[df.learned].n.sum()):,}**."


def predict_example():
    ck = ROOT / 'models' / 'losloop_tgcn_res_time.pt'
    if not ck.is_file():
        return '*(chạy `python tools/export_models.py` trước)*'
    out = subprocess.run([sys.executable, str(ROOT / 'predict.py'), '--checkpoint', str(ck), '--input',
                          str(ROOT / 'data/losloop/speed.csv'), '--end-time', '2012-03-07 07:30', '--compare',
                          '--sensors', '4'], capture_output=True, text=True, cwd=ROOT,
                         env={**__import__('os').environ, 'TGCN_THREADS': '1'}).stdout
    return '```\n' + out.strip() + '\n```'


def oracle_table():
    df = pd.read_csv(TAB / 'oracle_bias_losloop.csv')
    df['name'] = df.config.str.split('/').str[-1]
    g = df.groupby('name').agg(n=('seed', 'count'), val=('test_RMSE_val_selected', 'mean'),
                               test=('test_RMSE_test_selected', 'mean'), bias=('optimistic_bias_%', 'mean'),
                               bmax=('optimistic_bias_%', 'max'), ev=('best_epoch_val', 'mean'),
                               eo=('oracle_epoch_test', 'mean')).reset_index()
    order = ['gcn', 'gru', 'tgcn', 'gru_res_time', 'tgcn_res_time', 'tgcn_selfsep', 'tgcn_selfsep_res_time']
    g = g[g.name.isin(order)].set_index('name').reindex([o for o in order if o in set(g.name)]).reset_index()
    t = pd.DataFrame({'Mô hình': g.name.map(label), 'run': g.n,
                      'RMSE test – chọn theo val': g.val.map(f3), 'RMSE test – chọn theo test': g.test.map(f3),
                      'độ lạc quan TB': g.bias.map(lambda v: f'{v:.2f}%'), 'tối đa': g.bmax.map(lambda v: f'{v:.2f}%'),
                      'epoch (val / test)': [f'{a:.0f} / {b:.0f}' for a, b in zip(g.ev, g.eo)]})
    return md(t)


def oracle_stats():
    df = pd.read_csv(TAB / 'oracle_bias_losloop.csv')
    return df['optimistic_bias_%'].mean(), df['optimistic_bias_%'].max()


def paper_protocol_table():
    df = pd.read_csv(TAB / 'paper_protocol.csv')
    rows = []
    for _, r in df.iterrows():
        mins = '15 phút'
        if r.model in ('naive', 'ha_window'):
            rows.append({'Bộ dữ liệu': r.dataset, 'Mô hình': label(r.model), 'epoch train': '—',
                         'Chúng tôi (RMSE / MAE)': f'{r.RMSE:.3f} / {r.MAE:.3f}',
                         'Bài báo (RMSE / MAE)': '— (không báo cáo)' if r.model == 'naive'
                         else f'{r.paper_RMSE:.3f} / {r.paper_MAE:.3f}',
                         'Accuracy / R² (chúng tôi | bài báo)': '—'})
            continue
        rows.append({'Bộ dữ liệu': r.dataset, 'Mô hình': label(r.model), 'epoch train': f'{int(r.epochs)} (tốt nhất: {int(r.best_epoch)})',
                     'Chúng tôi (RMSE / MAE)': f'{r.RMSE:.3f} / {r.MAE:.3f}',
                     'Bài báo (RMSE / MAE)': f'{r.paper_RMSE:.3f} / {r.paper_MAE:.3f}',
                     'Accuracy / R² (chúng tôi | bài báo)': f'{r.Accuracy:.3f} / {r.R2:.3f} | {r.paper_Accuracy:.3f} / {r.paper_R2:.3f}'})
    return md(pd.DataFrame(rows))


def final_table():
    df = pd.read_csv(TAB / 'main_results_losloop.csv').set_index('model')
    cand = ['gru', 'tgcn', 'tgcn_selfsep', 'gru_res_time', 'tgcn_res_time', 'tgcn_selfsep_res_time',
            'gru_res_time_rawloss']
    rows = []
    for m in cand:
        if m not in df.index: continue
        r = df.loc[m]
        rows.append({'Cấu hình (3 seed)': label(m), 'RMSE validation ↓ (tiêu chí chọn)': r['VAL_RMSE (chọn mô hình)'],
                     'RMSE test': r.overall_RMSE, 'MAE test': r.overall_MAE})
    for m in ('ensemble_gru_res_time', 'ensemble_tgcn_selfsep', 'ensemble_mix'):
        if m in df.index:
            r = df.loc[m]
            rows.append({'Cấu hình (3 seed)': label(m) if m != 'ensemble_tgcn_selfsep' else 'Ensemble T-GCN tách self (3 seed)',
                         'RMSE validation ↓ (tiêu chí chọn)': '—', 'RMSE test': r.overall_RMSE, 'MAE test': r.overall_MAE})
    return md(pd.DataFrame(rows))


FILL = {'FINAL_TABLE': final_table, 'ORACLE_TABLE': oracle_table, 'PAPER_PROTOCOL_TABLE': paper_protocol_table,'PEAK_TABLE': peak_table, 'ABLATION_TABLE': ablation_table, 'TUNING_TABLE': tuning_table,
        'MAIN_TABLE': main_table, 'FI_TABLE': fi_table, 'SEED_TABLE': seed_table,
        'PAIRWISE_TABLE': pairwise_table, 'CONFORMAL_TABLE': conformal_table, 'PAPER_TABLE': paper_table,
        'MODEL_CARD_TABLE': model_card, 'PREDICT_EXAMPLE': predict_example,
        'SZ_MAIN_TABLE': lambda: main_table('sztaxi'), 'SZ_MASKED_TABLE': lambda: main_table('sztaxi', True)}


def main():
    text = '\n\n'.join((REP / p).read_text(encoding='utf-8') for p in PARTS if (REP / p).is_file())
    for key, fn in FILL.items():
        if re.search(rf'^{key}$', text, flags=re.M):
            try:
                text = re.sub(rf'^{key}$', lambda _: fn(), text, flags=re.M)
            except FileNotFoundError as e:
                print('missing table for', key, e)
    try:
        mean_b, max_b = oracle_stats()
        text = text.replace('ORACLE_MEAN%', f'{mean_b:.2f}%').replace('ORACLE_MAX%', f'{max_b:.2f}%')
    except FileNotFoundError:
        pass
    left = sorted(set(re.findall(r'^[A-Z_]{6,}$', text, flags=re.M)) | set(re.findall(r'[A-Z]{4,}_[A-Z_]{3,}', text)))
    if left:
        print('UNFILLED placeholders:', left)
    (REP / 'BAO_CAO_TGCN.md').write_text(text, encoding='utf-8')
    print('wrote', REP / 'BAO_CAO_TGCN.md', len(text.splitlines()), 'lines')


if __name__ == '__main__':
    main()
