import torch
import dgl


def to_dense(edge_idx):
    num_nodes = edge_idx.max().item() + 1
    dense = torch.zeros([num_nodes, num_nodes], device=edge_idx.device)
    for i in range(edge_idx.shape[1]):
        e1 = edge_idx[0][i]
        e2 = edge_idx[1][i]
        dense[e1][e2] += 1
    return dense


def to_dense_adj(edge_index, edge_attr=None, num_nodes=None):
    if edge_attr is None:
        edge_attr = torch.ones(edge_index[0].shape[0])
    if num_nodes is None:
        num_nodes = torch.stack(edge_index, dim=0).max().item() + 1

    if len(edge_attr.shape) == 1:
        dense = torch.zeros([num_nodes, num_nodes], dtype=edge_attr.dtype)
    elif len(edge_attr.shape) == 2:
        dense = torch.zeros([num_nodes, num_nodes, edge_attr.shape[1]], dtype=edge_attr.dtype)
    else:
        raise ValueError('The shape of edge_attr is invalid.')

    dense[edge_index] = edge_attr
    # assert (torch.equal(dense, dense.transpose(1, 0)))
    return dense


def check_edge_max_equal_num_nodes(g):
    edge_idx = torch.stack([g.edges()[0], g.edges()[1]], dim=0)
    assert (g.num_nodes() == edge_idx.max().item() + 1)


def check_dense(g, dense_used):
    edge_idx = torch.stack([g.edges()[0], g.edges()[1]], dim=0)
    dense = to_dense(edge_idx)
    assert (dense_used.equal(dense))
    assert (dense_used.max().item() <= 1)
    assert (len(dense_used.shape) == 2)
    assert (dense_used.equal(dense_used.transpose(0, 1)))


def check_repeated_edges(g):
    edge_idx = torch.stack([g.edges()[0], g.edges()[1]], dim=0)
    edge_unique = torch.unique(edge_idx, dim=1)
    assert (edge_unique.shape[1] == edge_idx.shape[1])


def power(power, graph_matrix):
    if isinstance(power, list):
        left = power[0]
        right = power[1]
    else:
        left = 1
        right = power
    if left <= 0 or right <= 0 or left > right:
        raise ValueError('Invalid power {}'.format(power))

    bases = []
    graph_matrix_n = torch.eye(graph_matrix.shape[0], dtype=graph_matrix.dtype)
    # assert (torch.equal(graph_matrix_n, graph_matrix_n.transpose(1, 0)))
    for _ in range(left - 1):
        graph_matrix_n = torch.matmul(graph_matrix_n, graph_matrix)
    for _ in range(left, right + 1):
        graph_matrix_n = torch.matmul(graph_matrix_n, graph_matrix)
        # assert (torch.allclose(graph_matrix_n, graph_matrix_n.transpose(1, 0)))
        bases = bases + [graph_matrix_n.unsqueeze(-1)]
    return bases


def power_norm(adj, config):
    bases = []
    deg = adj.sum(1)
    for i, eps in enumerate(config.norm):
        sym_basis = deg.pow(eps).unsqueeze(-1)
        graph_matrix = torch.matmul(sym_basis, sym_basis.transpose(0, 1)) * adj
        # assert (torch.equal(graph_matrix, graph_matrix.transpose(1, 0)))
        bases = bases + power(config.power[i], graph_matrix)
    return bases


def power_norm2(adj, config):
    bases = []
    deg = adj.sum(1)
    for i, eps in enumerate(config.norm2):
        d_left = deg.pow(eps).diag()
        d_right = deg.pow(-0.5 - eps).diag()
        graph_matrix = torch.matmul(torch.matmul(d_left, adj), d_right)
        bases = bases + power(config.power[i], graph_matrix)
    return bases


def power_eigen(adj, config):
    bases = []
    eig_val, eig_vec = torch.linalg.eigh(adj)
    eig_val_nosign = eig_val.abs()
    eig_val_nosign = torch.where(eig_val_nosign > 1e-6, eig_val_nosign, torch.zeros_like(eig_val_nosign))  # Precision limitation
    for i, power in enumerate(config.power):
        eig_val_pow = eig_val_nosign.pow(config.eigen * power)
        graph_matrix = torch.matmul(eig_vec, torch.matmul(eig_val_pow.diag_embed(), eig_vec.transpose(1, 0)))
        # assert (torch.allclose(graph_matrix, graph_matrix.transpose(1, 0)))
        bases = bases + [graph_matrix.unsqueeze(-1)]
    return bases


def pair_augment(g, bases, config):
    if len(g.ndata['feat'].shape) == 1:
        nfeat = g.ndata['feat'].unsqueeze(-1)
    else:
        nfeat = g.ndata['feat']
    nfeat = nfeat - nfeat.min() + 1.0
    for neps in config.aug_n:
        _nfeat_pow = nfeat.pow(neps)
        nfeat_pow = torch.matmul(_nfeat_pow, _nfeat_pow.transpose(1, 0)).unsqueeze(-1)
        bases = bases + nfeat_pow * config.aug_coeff
    return bases

def build_local_encoding(g, config):
    # check_edge_max_equal_num_nodes(g)  # with self_loop added, this should be held
    # check_repeated_edges(g)
    edge_idx = g.edges()
    adj = to_dense_adj(edge_idx)  # Graphs may have only one node.
    # check_dense(g, adj)

    bases = [torch.eye(adj.shape[0], dtype=adj.dtype).unsqueeze(-1)]

    if config.get('norm') is not None:
        bases = bases + power_norm(adj, config)

    if config.get('eigen') is not None:
        bases = bases + power_eigen(adj, config)

    if config.get('norm2') is not None:
        bases = bases + power_norm2(adj, config)

    bases = torch.cat(bases, dim=-1)

    assert (g.ndata['feat'].shape[0] == g.num_nodes())
    if config.get('aug_n') is not None:
        bases = pair_augment(g, bases, config)

    new_g = dgl.graph(torch.ones_like(adj, dtype=adj.dtype).nonzero(as_tuple=True))
    assert (new_g.num_nodes() == g.num_nodes())
    # new_g = DGLHeteroGraph(new_edge_idx, ['_U'], ['_E'])
    new_g.ndata['feat'] = g.ndata['feat']
    # new_g.ndata['_ID'] = g.ndata['_ID']

    if 'feat' in g.edata.keys():
        edge_attr = g.edata.pop('feat')
        assert (edge_attr.min() > 0)
        # print(edge_attr)
        edge_attr_dense = to_dense_adj(edge_idx, edge_attr=edge_attr)
        if len(edge_attr_dense.shape) == 2:
            new_g.edata['feat'] = edge_attr_dense.view(-1)
            efeat = edge_attr_dense.unsqueeze(-1)
        else:
            new_g.edata['feat'] = edge_attr_dense.view(-1, edge_attr.shape[-1])
            efeat = edge_attr_dense
        if config.get('aug_e') is not None:
            for eeps in config.aug_e:
                bases = bases + efeat.pow(eeps).sum(-1, keepdim=True) * config.aug_coeff

    if config.get('symmetrize', 'N') == 'Y':
        bases = (bases + bases.transpose(1, 0)) / 2.0

    if config.get('acc_cut', 'N') == 'Y':
        # bases = bases * 1000000.0 // 1.0 / 1000000.0
        bases = (bases * 1000000.0).int() / 1000000.0

    if config.get('basis_norm', 'N') == 'Y':
        std = torch.std(bases, 0, keepdim=True, unbiased=False)
        mean = torch.mean(bases, 0, keepdim=True)
        bases = (bases - mean) / (std + 1e-5)
        # bases = bases - mean

    # assert (torch.equal(bases, bases.transpose(1, 0)))
    bases = bases.view(-1, bases.shape[-1]).contiguous()

    new_g.edata['bases'] = bases
    # new_g.edata['_ID'] = g.edata['_ID']
    # print(new_g)

    if config.get('sparse') is not None:
        sparse_adj = torch.where(torch.rand_like(adj) <= config.sparse, 1.0, adj)
        sparse_edge_idx = sparse_adj.nonzero(as_tuple=True)

        sparse_g = dgl.graph(sparse_edge_idx)
        assert (sparse_g.num_nodes() == g.num_nodes())
        # new_g = DGLHeteroGraph(new_edge_idx, ['_U'], ['_E'])
        sparse_g.ndata['feat'] = g.ndata['feat']
        # new_g.ndata['_ID'] = g.ndata['_ID']
        if 'feat' in new_g.edata.keys():
            if len(edge_attr_dense.shape) == 2:
                sparse_g.edata['feat'] = new_g.edata['feat'].view(new_g.num_nodes(), new_g.num_nodes())[sparse_edge_idx].contiguous()
            else:
                sparse_g.edata['feat'] = new_g.edata['feat'].view(new_g.num_nodes(), new_g.num_nodes(), -1)[sparse_edge_idx].contiguous()
        sparse_g.edata['bases'] = new_g.edata['bases'].view(new_g.num_nodes(), new_g.num_nodes(), -1)[sparse_edge_idx].contiguous()
        # print(new_g.edata['bases'].shape[0], ' -> ', sparse_g.edata['bases'].shape[0])
        new_g = sparse_g

    return new_g
