import re
import sys
import matplotlib
matplotlib.use('Agg')

path = sys.argv[1]
blocks = re.findall(r'```python\n(.*?)```', open(path).read(), re.S)
print('{0} blocks'.format(len(blocks)))

env = {'__name__': '__main__'}
for i, b in enumerate(blocks):
    try:
        exec(compile(b, '{0}:b{1}'.format(path, i), 'exec'), env)
    except Exception as e:
        print('BLOCK {0} FAILED: {1}: {2}'.format(i, type(e).__name__, e))
        print(b.strip()[:300])
        raise SystemExit(1)

print('all blocks executed')
