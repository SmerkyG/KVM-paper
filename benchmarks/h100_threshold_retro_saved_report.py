"""Combine existing first-response reports after checking paired raw inputs."""
import argparse
import json
from pathlib import Path


def first_response_table(path):
    text=path.read_text(); section=text.split('## First response\n',1)[1].split('\n## ',1)[0]
    lines=[line for line in section.splitlines() if line.startswith('|')]
    header=[x.strip() for x in lines[0].strip('|').split('|')]
    rows={}
    for line in lines[2:]:
        values=[x.strip() for x in line.strip('|').split('|')]
        if values[0].startswith('**'):continue
        rows[values[0]]={name:float(value) for name,value in zip(header[1:-1],values[1:-1])}
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threshold-report',type=Path,required=True);p.add_argument('--retro-report',type=Path,required=True)
    p.add_argument('--threshold-raw',type=Path,required=True);p.add_argument('--retro-raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    t=json.loads(a.threshold_raw.read_text());r=json.loads(a.retro_raw.read_text())
    if t['stage']!='complete' or r['stage']!='complete' or t['mode']!='two-tier' or r['mode']!='retroinfer':
        raise ValueError('expected complete threshold and RetroInfer runs')
    if t.get('threshold_routing',{}).get('mass_multiplier')!=16 or t['model_name']!=r['model_name']:
        raise ValueError('routing or model mismatch')
    tasks=tuple(r['expected']);old={(x['task'],x['source_row']):x for x in r['records']}
    new={(x['task'],x['source_row']):x for x in t['records'] if x['task'] in tasks}
    if len(tasks)!=8 or len(old)!=256 or set(old)!=set(new):raise ValueError('expected eight matched 32-example tasks')
    for key in old:
        for field in ('prompt_sha256','input_token_sha256','answers'):
            if old[key][field]!=new[key][field]:raise ValueError(f'input mismatch {key}/{field}')
    threshold=first_response_table(a.threshold_report);retro=first_response_table(a.retro_report)
    if not set(tasks)<=set(threshold) or set(tasks)!=set(retro):raise ValueError('report task mismatch')
    lines=['# Llama 3.1 32K RULER: two-tier threshold versus RetroInfer','',
           'Eight selected tasks with 32 matched examples each, BF16 weights and KV. '
           'Scores use only the first response; the mean weights tasks equally. '
           'This selected-task mean is not a 13-task RULER average. LoD uses sparse '
           'prefill and two-tier c=16 threshold decode. RetroInfer uses dense prefill '
           'and its released GPU cache with 1.8% retrieval and 23.2% estimated summaries.', '',
           '| Task | Dense | LoD threshold | RetroInfer | RetroInfer minus LoD, points |',
           '|---|---:|---:|---:|---:|']
    values=[]
    for task in tasks:
        d=threshold[task]['Dense'];l=threshold[task]['Two-tier threshold'];v=retro[task]['RetroInfer']
        if abs(d-retro[task]['Dense'])>0.01:raise ValueError('dense summary mismatch')
        values.append((d,l,v));lines.append(f'| {task} | {d:.2f}% | {l:.2f}% | {v:.2f}% | {v-l:+.2f} |')
    means=[sum(row[i] for row in values)/len(values) for i in range(3)]
    lines.append(f'| **Selected-task average** | **{means[0]:.2f}%** | **{means[1]:.2f}%** | **{means[2]:.2f}%** | {means[2]-means[1]:+.2f} |')
    lines += ['', 'The displayed averages are computed from the source reports’ two-decimal '
              'task scores, so they can differ by at most a few thousandths from '
              'averages computed directly from unrounded scores. Small differences '
              'are uncertain with 32 examples per task. The remaining five tasks '
              'have not been run with RetroInfer.', '',
              f'[Full LoD 13-task report]({a.threshold_report.name}) · '
              f'[Original RetroInfer report]({a.retro_report.name}) · '
              f'[Threshold raw results]({a.threshold_raw.name}) · '
              f'[RetroInfer raw results]({a.retro_raw.name})', '']
    a.output.write_text('\n'.join(lines));print('\n'.join(lines))


if __name__=='__main__':main()
