import torch

from taf.fastclient import train_many
from taf.models import MNISTCNN
from taf.utils import get_flat


def test_vmap_matches_seq():
    """In float64 the parallel and sequential trainers must agree to rounding;
    in float32 they differ only by summation order (amplified by SGD)."""
    torch.manual_seed(0)
    model = MNISTCNN().double()
    g0 = get_flat(model)
    sizes = [37, 64, 130, 5, 90]                   # uneven, incl. partial batches
    datasets = [(torch.randn(n, 1, 28, 28, dtype=torch.float64), torch.randint(0, 10, (n,)))
                for n in sizes]
    kw = dict(epochs=2, lr=0.05, bs=32)
    fast = train_many(model, g0, datasets, gen=torch.Generator().manual_seed(7),
                      chunk=2, trainer="vmap", **kw)   # chunk < K exercises bucketing
    ref = train_many(model, g0, datasets, gen=torch.Generator().manual_seed(7),
                     trainer="seq", **kw)
    rel = ((fast - ref).norm(dim=1) / ref.norm(dim=1)).max()
    assert rel < 1e-10, rel
