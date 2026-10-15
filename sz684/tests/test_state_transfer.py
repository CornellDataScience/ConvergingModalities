import numpy as np
import torch
from state_analysis import score_queries,select_scale,clustered_bootstrap,macro
from state_data import make_folds


def test_state_ap_matches_hand_calculation_and_excludes_query():
    # Two nouns, three independently indexed images per state. State/noun clusters
    # have orthogonal centers; target-directed queries retrieve all target examples.
    y=torch.zeros(2,3,2,4,dtype=torch.float64)
    y[0,:,0,0]=1;y[0,:,1,1]=1;y[1,:,0,2]=1;y[1,:,1,3]=1
    result=score_queries(y[:,:,1],y)
    np.testing.assert_allclose(result['ap'],1)
    assert result['noun'].all() and result['joint'].all()
    np.testing.assert_allclose(result['within_auc'],1)
    source=score_queries(y[:,:,0],y)
    # Relevant target scores tie with the other noun; AP cannot be perfect.
    assert np.all(source['ap']<1)


def test_average_precision_against_independent_python_rank_loop():
    gen=torch.Generator().manual_seed(72)
    y=torch.nn.functional.normalize(torch.randn(3,4,2,8,generator=gen,dtype=torch.float64),dim=-1)
    q=y[:,:,0];actual=score_queries(q,y)['ap'].ravel();expected=[]
    for category in range(3):
        for i in range(4):
            entries=[]
            for other in range(3):
                for j in range(4):
                    for state in range(2):
                        if (state==0 and j==i) or (state==1 and j==(i+1)%4):continue
                        entries.append((float(q[category,i]@y[other,j,state]),other==category and state==1))
            entries.sort(key=lambda x:-x[0]);found=0;precisions=[]
            for rank,(_,relevant) in enumerate(entries,1):
                if relevant:found+=1;precisions.append(found/rank)
            expected.append(np.mean(precisions))
    np.testing.assert_allclose(actual,expected)


def test_shared_noun_bootstrap_and_relation_macro():
    rel=np.array([0,0,1,1]);noun=np.array([0,1,0,2]);effects=np.array([np.tile([[1.],[1.],[3.],[3.]],(1,4))])
    boot,cat,per,weights,queries=clustered_bootstrap(effects,rel,noun,9,100,2)
    np.testing.assert_allclose(boot,2);np.testing.assert_allclose(cat,2)
    assert macro(effects[0],rel,2)==2
    assert weights.shape==(100,3) and queries.shape==(100,4,4)


def test_global_noun_groups_cover_each_relation():
    nouns=[str(i) for i in range(20)];support={'a':nouns[:12],'b':nouns[8:]}
    assignment,_=make_folds(nouns,support,91)
    assert set(assignment)==set(nouns)
    for group in range(5):
        for allowed in support.values():assert any(assignment[n]==group for n in allowed)
