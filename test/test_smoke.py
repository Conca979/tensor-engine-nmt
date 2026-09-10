import sys, io, os, time
# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from tensor_engine_nmt.config import HParams
from tensor_engine_nmt.bpe import load_or_train_bpe
from tensor_engine_nmt.dataset import PhoMTDataset
from tensor_engine_nmt.model import Seq2Seq
from tensor_engine_nmt.optimizer import Adam

def run_smoke():
    hp = HParams(
        V=500, e=32, d=64, L=1, B=8,
        k=2000.0, lr=1e-3, beta1=0.9, beta2=0.999, eps_adam=1e-8, clip_norm=5.0,
        max_epochs=1, beam_width=2, max_decode_len=30,
        log_every=5, save_every=9999,
        data_dir='PhoMT_dataset', bpe_dir='bpe_vocab_smoke', ckpt_dir='checkpoints',
        bpe_sample_lines=500,
    )

    t0 = time.time()
    print('Training BPE (500 lines, V=500)...', flush=True)
    bpe = load_or_train_bpe(hp.bpe_dir, hp.data_dir, hp.V, hp.bpe_sample_lines)
    print('BPE done in %.1fs' % (time.time()-t0), flush=True)

    test_en = 'Hello world how are you'
    ids = bpe.encode(test_en, add_end=True)
    decoded = bpe.decode(ids)
    print('Encode test: %r -> %s ids -> %r' % (test_en, len(ids), decoded), flush=True)

    print('Loading dataset...', flush=True)
    ds = PhoMTDataset(bpe, hp.data_dir, 'train', max_len=50)

    model = Seq2Seq(hp)
    optim = Adam(model, hp)
    print('Model params: %d' % model.param_count(), flush=True)

    losses = []
    t1 = time.time()
    print('Training 20 steps...', flush=True)
    for step, batch in enumerate(ds.iterate(batch_size=hp.B, shuffle=False)):
        if step >= 20: break
        X, Xlen = batch['X'], batch['Xlen']
        Yin, Yout, Ylen = batch['Yin'], batch['Yout'], batch['Ylen']
        model.zero_grad()
        loss_val, _ = model.forward(X, Xlen, Yin, Yout, Ylen, step)
        model.backward()
        gnorm = optim.step()
        losses.append(loss_val)
        if (step+1) % 5 == 0:
            print('  step %3d  loss=%.4f  gnorm=%.3f' % (step+1, loss_val, gnorm), flush=True)

    elapsed = time.time() - t1
    print('20 steps in %.1fs  (%.2f s/step)' % (elapsed, elapsed/20), flush=True)

    first5 = np.mean(losses[:5])
    last5  = np.mean(losses[-5:])
    print('first-5 avg loss: %.4f   last-5 avg loss: %.4f' % (first5, last5), flush=True)

    if last5 < first5 * 1.5:
        print('SMOKE TEST PASSED', flush=True)
    else:
        print('SMOKE TEST FAILED', flush=True)
        sys.exit(1)

if __name__ == '__main__':
    run_smoke()
