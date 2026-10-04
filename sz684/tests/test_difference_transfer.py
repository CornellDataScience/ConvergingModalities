import numpy as np
import torch
from difference_transfer import fit_ridge,difference_scores


def test_ridge_recovers_affine_map_and_differences_cancel_intercept():
    generator=torch.Generator().manual_seed(81)
    x=torch.randn(70,8,generator=generator,dtype=torch.float64)
    matrix=torch.randn(8,6,generator=generator,dtype=torch.float64)
    y=x@matrix+torch.arange(6,dtype=torch.float64)
    fitted=fit_ridge(x,y,alpha=0.)
    torch.testing.assert_close(fitted,matrix)
    torch.testing.assert_close((x[1:]-x[:-1])@fitted,y[1:]-y[:-1])


def test_difference_retrieval_identity_and_reversed_direction():
    targets=torch.eye(7)
    correct=difference_scores(targets,targets)
    reversed_scores=difference_scores(-targets,targets)
    assert correct['top1']==1 and correct['mean_cosine']==1
    assert reversed_scores['top1']==0 and reversed_scores['mean_cosine']==-1


def test_cached_solver_equals_refitting_shuffled_map():
    generator=torch.Generator().manual_seed(82)
    x=torch.randn(25,6,generator=generator)
    y=torch.randn(25,4,generator=generator)
    p=np.random.default_rng(83).permutation(25)
    xc=x-x.mean(0);yc=y-y.mean(0)
    solver=torch.linalg.solve(xc.T@xc+torch.eye(6),xc.T)
    torch.testing.assert_close(solver@yc[p],fit_ridge(x,y[p]),atol=1e-6,rtol=1e-5)
