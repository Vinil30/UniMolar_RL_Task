Most molecular models use either SMILES strings or bond graphs to represent molecules. These describe atomic connectivity. They do not distinguish different conformers of the same molecule. They also miss spatial contacts between atoms that are close in space but far apart in the bond graph. This task tests whether a 3D representation gives an advantage when labelled data is scarce and a large pool of unlabelled conformations is available.

Use only 1,000 labelled molecules for supervised training. This small labelled set is intentional. The goal is to see whether self-supervised pre-training on the larger unlabelled collection helps downstream property prediction.

All input data lives under /app/task_inputs/. The molecular geometries are B3LYP-optimised conformations. The regression target is the HOMO-LUMO gap in eV.

The molecular arrays are stored in ragged form. Each .npz file contains:

atoms: atomic numbers for the concatenated atoms.
coords: an (total_atoms, 3) floating-point coordinate array.
offsets: molecule boundaries, where molecule i spans rows offsets[i] through offsets[i+1].

The molecules contain H, C, N, O and F. An atomic number of 0 marks a hidden atom type.

Each conformation may use a different coordinate frame and atom ordering. Treat absolute coordinates and positions as arbitrary.

Input files:

pretrain.npz: 50,000 unlabelled molecular conformations, geometry only.
train.npz, train_labels.csv, train_smiles.txt: 1,000 labelled molecules. train.npz also includes bonds and bond_offsets for covalent connectivity.
val.npz, val_smiles.txt: 2,000 molecules with bonds and SMILES strings. Labels are withheld for evaluation.
test.npz: 3,000 molecules, geometry only, no labels.
denoise.npz: 2,000 conformations with independent Gaussian noise (σ = 0.15 Å) added to each coordinate. Original geometries are withheld.
mask.npz: 2,000 conformations where half the heavy-atom identities are replaced with 0, using the same noise as the denoising split. mask_positions and mask_offsets mark the hidden atoms. Their identities are withheld.
topology_eval.npz: 1,000 molecules with atoms and bond connectivity, no coordinates.
sequence_eval_smiles.txt: 1,000 SMILES strings, nothing else.
3D model

Build a 3D molecular encoder whose output is invariant under translation, rotation, and reflection of the input frame. Route geometry through rotation-invariant quantities. Raw Cartesian coordinates must not leak into the representation.

Pre-train the encoder on the 50,000 unlabelled conformations with two self-supervised objectives:

Recover clean atom positions from perturbed coordinates.
Predict the identity of atoms whose element type was hidden.

After pre-training, fine-tune the 3D encoder on the 1,000 labelled molecules to predict the HOMO-LUMO gap. The fine-tuned model's mean absolute error on the gap prediction (gap_mae) must be 0.35 eV or lower.

Baseline models

Train two separate supervised baselines on the same labelled training molecules:

A bond-graph model using connectivity, no geometry.
A SMILES model using the molecular string alone.

Train both baselines as real predictive models. Give both a competitive training effort and a fair number of optimization steps.

The modality-specific evaluation sets add extra checks. Keep the 3D model limited to information it can actually access:

topology_eval.npz has no coordinates. Evaluate it with the bond-graph model only.
sequence_eval_smiles.txt is evaluated only with the SMILES model.
Required output files

Write all results to /app/results/.

predictions.json

One JSON object with these seven keys:

test_gap: 3,000 predictions for test.npz, in file order.
val_gap_3d: 2,000 validation predictions from the 3D model.
val_gap_2d: 2,000 validation predictions from the bond-graph model.
val_gap_1d: 2,000 validation predictions from the SMILES model.
topology_gap: 1,000 predictions from the bond-graph model on topology_eval.npz.
sequence_gap: 1,000 predictions from the SMILES model on sequence_eval_smiles.txt.
hidden_atom_types: one predicted atomic number per hidden atom in the masked split, ordered per mask_positions. There are 7,984 values, each drawn from {1, 6, 7, 8, 9}. Overall mask accuracy (mask_accuracy) must be 0.88 or higher.
denoised_coords.npz

Store:

coords: refined coordinates for the denoising split.
offsets: identical to the offsets in denoise.npz.

The coordinate array must match the input coordinates in shape.

Evaluation aligns each predicted molecule to the withheld reference geometry using optimal rigid alignment, then computes pooled RMSD. You may return coordinates in any frame. Only the internal geometry needs to improve. After alignment, denoise_rmsd must be 0.19 Å or lower.

The refined geometry must also stay close to the noisy input. After the same alignment procedure, its distance from the input must be below 1.5 Å.

Model checkpoints

Write:

unimol_model.pt: state dict for the fine-tuned 3D encoder.
gnn2d_model.pt: state dict for the bond-graph baseline.

These must be distinct networks with meaningful parameter counts. The encoder needs at least 20,000 parameters and two weight matrices. The graph baseline needs at least 5,000 parameters and two weight matrices.

ablation.json

Store an object of the form:

json
{"runs": [...]}