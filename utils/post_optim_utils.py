from torch_geometric.utils import to_dense_adj
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

def compute_RMSD(a, b):
    return torch.sqrt((((a - b) ** 2).sum(axis=-1)).mean())

def post_optimize_loss_function(epoch, x, predict_compound_coords, compound_pair_dis_constraint,
                                LAS_distance_constraint_mask=None, mode=0):
    dis = (x - predict_compound_coords).norm(dim=-1)
    dis_clamp = torch.clamp(dis, max=10)
    if mode == 0:
        interaction_loss = dis_clamp.sum()
    elif mode == 1:
        interaction_loss = (dis_clamp ** 2).sum()
    elif mode == 2:
        # probably not a good choice. x^0.5 has infinite gradient at x=0. added 1e-5 for numerical stability.
        interaction_loss = ((dis_clamp.abs() + 1e-5) ** 0.5).sum()
    else:
        raise NotImplementedError()

    config_dis = torch.cdist(x, x)
    if LAS_distance_constraint_mask is not None:
        configuration_loss = 1 * (
            ((config_dis - compound_pair_dis_constraint).abs())[LAS_distance_constraint_mask]).sum()
        configuration_loss += 2 * ((1.22 - config_dis).relu()).sum()
    else:
        configuration_loss = 1 * ((config_dis - compound_pair_dis_constraint).abs()).sum()
    loss = configuration_loss
    return loss, (interaction_loss.item(), configuration_loss.item())

def post_optimize_compound_coords(reference_compound_coords, predict_compound_coords,
                                  total_epoch=1000, LAS_edge_index=None, mode=0):
    if LAS_edge_index is not None:
        LAS_distance_constraint_mask = to_dense_adj(LAS_edge_index).squeeze(0).to(torch.bool)
    else:
        LAS_distance_constraint_mask = None
    compound_pair_dis_constraint = torch.cdist(reference_compound_coords, reference_compound_coords)
    x = predict_compound_coords.clone()
    x.requires_grad = True
    optimizer = torch.optim.Adam([x], lr=0.1)
    loss_list = []
    rmsd_list = []
    for epoch in range(total_epoch):
        optimizer.zero_grad()
        loss, (interaction_loss, configuration_loss) = post_optimize_loss_function(epoch, x, predict_compound_coords,
                                                                                   compound_pair_dis_constraint,
                                                                                   LAS_distance_constraint_mask=LAS_distance_constraint_mask,
                                                                                   mode=mode,
                                                                                   )
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        rmsd = compute_RMSD(reference_compound_coords, x.detach()) 
        rmsd_list.append(rmsd.item())
    return x.detach(), loss_list[-1], rmsd_list[-1]

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
  
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize:
            Chem.SanitizeMol(mol)
        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol

def write_mol(reference_file, coords, output_file):
    mol = read_molecule(reference_file, sanitize=True, remove_hs=True)
    if mol is None:
        raise Exception()
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    w = Chem.SDWriter(output_file)
    w.SetKekulize(False)
    w.write(mol)
    w.close()
    return mol
