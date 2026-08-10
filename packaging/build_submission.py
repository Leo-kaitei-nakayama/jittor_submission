"""
Build the competition submission package.

Reads packaging/submission_meta.json, renders 提交说明文档.pdf from
packaging/doc_template.html, assembles the required directory layout, and
zips it as  contest<track>_<队名>_<名次>.zip.

Required layout (per the organizers' 提交格式要求):

    contest2_<队名>_<名次>.zip
    ├── code/                 与 A 榜一致的完整代码
    ├── requirements.txt      Python 依赖及版本
    └── 提交说明文档.pdf

Usage:
    pip install weasyprint
    python packaging/build_submission.py [--out-dir dist]
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGING = os.path.join(REPO, 'packaging')

# Everything in the repo root that is NOT part of the code deliverable.
EXCLUDE_TOP = {'.git', '.gitignore', 'packaging', 'dist', 'PROVENANCE.txt',
               'requirements.txt', 'results', 'datasets', 'pretrained',
               'dataset_train', 'dataset_test_noisy', 'datalist',
               'eval_gt', 'eval_noisy', 'eval_mesh_normalized'}

TRACK_CN = {'1': '一', '2': '二'}


def die(msg):
    print(f'ERROR: {msg}', file=sys.stderr)
    sys.exit(1)


def load_meta(path=None):
    path = path or os.path.join(PACKAGING, 'submission_meta.json')
    with open(path, encoding='utf-8') as f:
        meta = json.load(f)

    todo = [k for k, v in meta.items()
            if isinstance(v, str) and v.startswith('TODO_')]
    if todo:
        die('submission_meta.json 中以下字段仍未填写:\n  - ' + '\n  - '.join(todo))

    if meta['track'] not in TRACK_CN:
        die(f"track 必须是 '1' 或 '2'，当前为 {meta['track']!r}")
    return meta


def ignore_pycache(_dir, names):
    return {n for n in names if n == '__pycache__' or n.endswith('.pyc')}


def copy_code(stage_code, meta):
    os.makedirs(stage_code, exist_ok=True)
    for name in sorted(os.listdir(REPO)):
        if name in EXCLUDE_TOP:
            continue
        src = os.path.join(REPO, name)
        dst = os.path.join(stage_code, name)
        if os.path.islink(src):
            print(f'  [skip symlink] {name}')
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=ignore_pycache)
        else:
            shutil.copy2(src, dst)

    # requirements.txt lives at the package root; keep a copy inside code/ too
    # so README.md's `pip install -r requirements.txt` works from either place.
    shutil.copy2(os.path.join(REPO, 'requirements.txt'),
                 os.path.join(stage_code, 'requirements.txt'))

    if not meta.get('include_all_checkpoints', False):
        prune_checkpoints(stage_code, meta)


def prune_checkpoints(stage_code, meta):
    """Keep only the checkpoints the documented pipeline actually needs."""
    keep = {
        os.path.join('experiments', 'asdn', meta['best_ckpt']),
        os.path.join('experiments', 'asdn', 'train_log.csv'),
        os.path.join('experiments', 'classify', meta['classify_ckpt']),
    }
    exp = os.path.join(stage_code, 'experiments')
    if not os.path.isdir(exp):
        return
    removed = 0
    for root, _, files in os.walk(exp):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, stage_code)
            if rel not in keep:
                os.remove(full)
                removed += 1
    for rel in sorted(keep):
        if not os.path.exists(os.path.join(stage_code, rel)):
            die(f'checkpoint 缺失: {rel}（请检查 submission_meta.json）')
    print(f'  [checkpoints] 保留 {len(keep)} 个，删除 {removed} 个')


def render_pdf(out_pdf, meta):
    try:
        from weasyprint import HTML
    except ImportError:
        die('未安装 weasyprint。请先运行:  pip install weasyprint')

    with open(os.path.join(PACKAGING, 'doc_template.html'), encoding='utf-8') as f:
        html = f.read()

    rank = str(meta['rank']).strip()
    ckpt_note = ('本次训练产出的全部 checkpoint 与训练日志'
                 if meta.get('include_all_checkpoints', False) else
                 f"A 榜最优权重 {meta['best_ckpt']}、其依赖的 "
                 f"{meta['classify_ckpt']} 及训练日志 train_log.csv"
                 "（评委可跳过重训直接复现推理结果；从零训练路径见 5.2 节）")

    # The A-board leaderboard only publishes the combined score, so CD/P2S are
    # normally unavailable. Say so rather than printing a bare dash, which a
    # reviewer could misread as an official zero.
    cd, p2s = str(meta['score_cd']).strip(), str(meta['score_p2s']).strip()
    if cd in ('', '-', '—', 'N/A') or p2s in ('', '-', '—', 'N/A'):
        breakdown = ('官方榜单仅公布总分，未公布 CD / P2S 分项'
                     '（本地验证集上的分项见 5.5 节）')
    else:
        breakdown = f'CD {cd} ／ P2S {p2s}'

    fields = {
        'TRACK': meta['track'],
        'TRACK_CN': TRACK_CN[meta['track']],
        'TEAM_NAME': meta['team_name'],
        'RANK_PLAIN': rank.lstrip('0') or '0',
        'SCORE_TOTAL': meta['score_total'],
        'SCORE_BREAKDOWN': breakdown,
        'SUBMIT_DATE': meta['submit_date'],
        'CONTACT_NAME': meta['contact_name'],
        'CONTACT_WECHAT': meta['contact_wechat'],
        'CONTACT_PHONE': meta['contact_phone'],
        'BEST_CKPT': meta['best_ckpt'],
        'BEST_NITERS': str(meta['best_niters']),
        'CLASSIFY_CKPT': meta['classify_ckpt'],
        'CKPT_NOTE': ckpt_note,
        'BUILD_DATE': datetime.date.today().isoformat(),
    }
    for k, v in fields.items():
        html = html.replace('{{' + k + '}}', str(v))

    leftover = [tok for tok in ('{{',) if tok in html]
    if leftover:
        i = html.index('{{')
        die(f'模板中仍有未替换的占位符: {html[i:i + 40]!r}')

    HTML(string=html, base_url=PACKAGING).write_pdf(out_pdf)
    print(f'  [pdf] {out_pdf} ({os.path.getsize(out_pdf) / 1024:.0f} KB)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=os.path.join(REPO, 'dist'))
    parser.add_argument('--meta', default=None,
                        help='另一个 meta JSON 的路径（默认 packaging/submission_meta.json）')
    args = parser.parse_args()

    meta = load_meta(args.meta)
    rank = str(meta['rank']).strip()
    zip_stem = f"contest{meta['track']}_{meta['team_name']}_{rank}"

    out_dir = os.path.abspath(args.out_dir)
    stage = os.path.join(out_dir, zip_stem)
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)

    print(f'Building {zip_stem}.zip')
    copy_code(os.path.join(stage, 'code'), meta)
    shutil.copy2(os.path.join(REPO, 'requirements.txt'),
                 os.path.join(stage, 'requirements.txt'))
    render_pdf(os.path.join(stage, '提交说明文档.pdf'), meta)

    zip_path = os.path.join(out_dir, zip_stem + '.zip')
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(stage):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for fn in sorted(files):
                full = os.path.join(root, fn)
                zf.write(full, os.path.relpath(full, stage))

    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f'\nDone: {zip_path}  ({size_mb:.1f} MB)')
    print('Contents:')
    subprocess.run([sys.executable, '-c',
                    'import sys,zipfile;'
                    'z=zipfile.ZipFile(sys.argv[1]);'
                    'names=z.namelist();'
                    'print("\\n".join("  "+n for n in sorted(names)[:40]));'
                    'print(f"  ... {len(names)-40} more") if len(names)>40 else None',
                    zip_path])


if __name__ == '__main__':
    main()
