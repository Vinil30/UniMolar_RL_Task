# Asset manifest

Every data file shipped with this task, where it came from, and what may be
done with it. No third-party code is redistributed here; the only external
asset is the dataset.

## Provenance

All splits derive from **QM9** (`dsgdb9nsd`), the set of 133,885 small organic
molecules with B3LYP/6-31G(2df,p) optimised geometries and computed
properties, published as a Scientific Data descriptor by Ramakrishnan, Dral,
Rupp and von Lilienfeld (2014) and deposited on figshare as collection
`10.6084/m9.figshare.c.978904`. The task's regression target is the
HOMO-LUMO gap, converted from Hartree to electronvolts.

## Licence

Every article in that collection - including
`Data for 133885 GDB-9 molecules` (`dsgdb9nsd.xyz.tar.bz2`), the file these
splits are built from - is released under **CC0 1.0 Universal**, a public
domain dedication. Confirmed against the figshare API rather than a secondary
source, since one widely-repeated web summary misattributes a
non-commercial licence to this dataset.

    https://creativecommons.org/publicdomain/zero/1.0/

CC0 places no restriction on commercial use, redistribution or modification,
and imposes no attribution requirement. The citation below is therefore
offered as scholarly practice, not as a licence condition. Nothing in this
task carries a copyleft or non-commercial obligation.

## Assets

| File | Derivation |
| --- | --- |
| `task_inputs/pretrain.npz` | 50,000 molecules, geometry only |
| `task_inputs/train.npz`, `train_labels.csv`, `train_smiles.txt` | 1,000 labelled molecules with bond graphs and SMILES |
| `task_inputs/val.npz`, `val_smiles.txt` | 2,000 molecules; labels held out under `tests/` |
| `task_inputs/test.npz` | 3,000 rows over 2,700 molecules, 300 of them repeated under an independent rigid motion and atom permutation |
| `task_inputs/denoise.npz` | 2,000 conformations perturbed by Gaussian noise of 0.15 A per coordinate; reference geometry held out |
| `task_inputs/mask.npz` | 2,000 perturbed conformations with half the heavy atoms hidden; true elements held out |
| `task_inputs/topology_eval.npz` | 1,000 molecules as bond graphs with no coordinates |
| `task_inputs/sequence_eval_smiles.txt` | 1,000 SMILES strings |

Molecules with an identical internal geometry (mirror images and exact
duplicates) were collapsed to one representative before splitting, so no
molecule appears in two splits. Every conformation ships in a random frame
with its atoms randomly permuted.

## Citation

Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, O. Anatole von
Lilienfeld. Quantum chemistry structures and properties of 134 kilo
molecules. *Scientific Data* 1, 140022 (2014). doi:10.1038/sdata.2014.22
