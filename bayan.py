import numpy as np, sys, os
np.random.seed(1)
text = open('data/dialogs.txt', encoding='utf-8').read()
chars = sorted(set(text)); V = len(chars)
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
H, T, lr = 128, 50, 0.1

def fresh():
    return [np.random.randn(H,V)*0.01, np.random.randn(H,H)*0.01,
            np.random.randn(V,H)*0.01, np.zeros((H,1)), np.zeros((V,1))]
params = fresh()
if os.path.exists('model/bayan.npz'):
    d = np.load('model/bayan.npz')
    if list(d['chars']) == chars:
        params = [d[k] for k in ('Wxh','Whh','Why','bh','by')]
        print('تم تحميل النموذج المحفوظ')
Wxh, Whh, Why, bh, by = params

def save():
    np.savez('model/bayan.npz', Wxh=Wxh, Whh=Whh, Why=Why, bh=bh, by=by, chars=np.array(chars))

def loss_fn(inp, tgt, hprev):
    xs, hs, ps = {}, {-1: hprev.copy()}, {}
    loss = 0
    for t in range(len(inp)):
        xs[t] = np.zeros((V,1)); xs[t][inp[t]] = 1
        hs[t] = np.tanh(Wxh@xs[t] + Whh@hs[t-1] + bh)
        y = Why@hs[t] + by
        ps[t] = np.exp(y - y.max()); ps[t] /= ps[t].sum()
        loss += -np.log(ps[t][tgt[t], 0])
    g = [np.zeros_like(p) for p in params]
    dhn = np.zeros_like(hs[0])
    for t in reversed(range(len(inp))):
        dy = ps[t].copy(); dy[tgt[t]] -= 1
        g[2] += dy@hs[t].T; g[4] += dy
        dr = (1 - hs[t]**2) * (Why.T@dy + dhn)
        g[3] += dr; g[0] += dr@xs[t].T; g[1] += dr@hs[t-1].T
        dhn = Whh.T@dr
    for a in g: np.clip(a, -5, 5, out=a)
    return loss, g, hs[len(inp)-1]

def step_h(h, i):
    x = np.zeros((V,1)); x[i] = 1
    return np.tanh(Wxh@x + Whh@h + bh)

def reply(msg, n=80):
    prompt = [c for c in "س: " + msg + "\nج:" if c in c2i]
    h = np.zeros((H,1))
    for c in prompt[:-1]: h = step_h(h, c2i[c])
    ix, out = c2i[prompt[-1]], []
    for _ in range(n):
        h = step_h(h, ix)
        y = Why@h + by; p = np.exp(y - y.max()); p /= p.sum()
        ix = np.random.choice(V, p=p.ravel())
        if i2c[ix] == '\n': break
        out.append(i2c[ix])
    return ''.join(out).strip()

def train(steps):
    mem = [np.zeros_like(p) for p in params]
    p, hprev, smooth = 0, np.zeros((H,1)), np.log(V)*T
    for n in range(steps):
        if p + T + 1 >= len(text): p, hprev = 0, np.zeros((H,1))
        inp = [c2i[c] for c in text[p:p+T]]
        tgt = [c2i[c] for c in text[p+1:p+T+1]]
        loss, g, hprev = loss_fn(inp, tgt, hprev)
        for prm, gr, m in zip(params, g, mem):
            m += gr*gr; prm += -lr*gr/np.sqrt(m + 1e-8)
        p += T; smooth = smooth*0.999 + loss*0.001
        if n % 500 == 0:
            print(n, round(smooth,2), '| صباح الخير ->', reply('صباح الخير'))
            save()
    save()

if len(sys.argv) > 1 and sys.argv[1] == 'chat':
    while True:
        print('بيان:', reply(input('انت: ')))
else:
    train(int(sys.argv[2]) if len(sys.argv) > 2 else 6000)
