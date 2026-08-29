import json

with open('/home/z/my-project/soilactivity/examples/example01.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']
    new_src = []
    for i, line in enumerate(src):
        # Skip wrongly placed lines from previous attempt
        if '# Unfolder returns' in line:
            continue
        if 'res_mlem.activity_3d = res_mlem.activity_3d.transpose' in line:
            continue
        if 'res_tikh.activity_3d = res_tikh.activity_3d.transpose' in line:
            continue

        new_src.append(line)

        # After MLEM unfold closing line, add transpose
        if 'smooth_sigma=0.5)' in line and i > 0 and 'res_mlem' in ''.join(src[max(0,i-2):i]):
            new_src.append('res_mlem.activity_3d = res_mlem.activity_3d.transpose(1, 2, 0)  # (nz,ny,nx)->(nx,ny,nz)\n')

        # After Tikhonov unfold closing line, add transpose
        if 'smooth_sigma=0.5)' in line and i > 0 and 'res_tikh' in ''.join(src[max(0,i-2):i]):
            new_src.append('res_tikh.activity_3d = res_tikh.activity_3d.transpose(1, 2, 0)\n')

    cell['source'] = new_src

with open('/home/z/my-project/soilactivity/examples/example01.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Done')
