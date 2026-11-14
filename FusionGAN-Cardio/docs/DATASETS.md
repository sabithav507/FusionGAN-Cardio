# Dataset ingestion and manifest contract

All raw datasets must be obtained through their original sources and local research-use licenses. No protected image data are bundled. Set source directories using CLI flags; default manifest paths in `configs/full.yaml` are illustrative. Raw EchoNet AVI and CAMUS NIfTI are decoded/extracted and saved as local PNG frames.

## Standard CSV schema

| Column | Type | Required | Meaning |
|---|---|---|---|
| `path` | str | yes | Absolute image PNG path or relative to the manifest CSV |
| `split` | train/val/test | yes | Patient/study-disjoint partition, synthetic ONLY in train |
| `subject_id` | str | yes | Patient identifier when known; otherwise unique study key (weaker) |
| `study_id` | str | yes | Study/video key for aggregation; never span partitions |
| `y` | int or string | yes | TMED AS `0,1,2`; EchoNet EF `0,1`; CAMUS `-1` when unset |
| `view` | int or string | yes | `0=PLAX,1=PSAX,2=A2C,3=A4C,4=Other` |
| `is_synthetic` | bool | yes | False in raw manifests; True only for accepted train additions |
| `mask_path` | str | CAMUS | Binary/semantic integer label image with values 0..3 |
| `phase` | ED/ES | optional | CAMUS cardiac phase |
| `ef` | float | optional | Video/patient continuous EF for separate analysis |

The parser checks exact image-file existence, repeated paths, unknown views/classes, split names, synthetic/test leakage, and subject/study crossover. It does not infer real patients from anonymous filename patterns. If IDs were lost from the original dataset, patient-leakage guarantees cannot be restored by code.

### TMED metadata CSV

```
image_path,study_id,patient_id,disease,view,split
study101/frame07.png,study101,patient35,no_as,PLAX,train
study101/frame09.png,study101,patient35,no_as,A4C,train
study235/frame01.png,study235,patient51,early_as,PSAX,val
```

Study-level diagnosis is copied to each selected **known-view** image as a proxy label. A full echocardiographic examination with Doppler information is required for clinical AS grading. Source TMED-2 provides 599 studies from 577 patients, not 599 distinct patients; its official example split is 360/119/120 studies, while the manuscript reports 359/120/120. A paper-exact split and its final image counts cannot be reconstructed without original IDs.

The `Other` label is a heterogeneous collection of acquisition views. The source release includes unlabeled-view images; this program **requires actual view annotations for conditioned training** and does not hallucinate a view for an unlabeled frame. For a no-Other sensitivity run, create a filtered training/validation/test manifest that excludes `view=4` consistently.

### EchoNet-Dynamic

Expected local folders:

```
EchoNet-Dynamic/
  FileList.csv
  Videos/
    0X100009310A3BD7FC.avi
    ...
```

Reads `FileName`, `EF`, `Split` columns; samples five near-uniformly spaced frames (fewer if invalid), records frame index and EF, retains the **official video-level partition**, maps all frames to A4C view 3. It does **not** estimate EF from individual images.

### CAMUS

Endpoint file stems expected:

```
patient0001/
  patient0001_2CH_ED.nii.gz
  patient0001_2CH_ED_gt.nii.gz
  patient0001_2CH_ES.nii.gz
  patient0001_2CH_ES_gt.nii.gz
  patient0001_4CH_ED.nii.gz
  patient0001_4CH_ED_gt.nii.gz
  ...
  Info_2CH.cfg
  Info_4CH.cfg
```

`.mhd` files can be used with SimpleITK. The converter preserves integer mask labels in separate `.png` images. If readable official `subgroup_*.txt` train/test/validation files are found, they define partitions; otherwise a patient-grouped, seeded 400/100 split is assigned (for 500 patients). Official dataset editions are **not pixel-identical** and published official partitions may differ; record the release version and checksum in the paper.

### Data integrity requirements

- Keep a frozen CSV copy and checksums for each experiment.
- Avoid late resplitting after model selection; use the same partitions across methods/seeds.
- Never mix CAMUS segmentation target masks or EchoNet labels into TMED-2 disease labels.
- Frozen structure encoder can learn from CAMUS/EchoNet TRAIN only; its embeddings act as a constraint, not as disease ground truth.
- Only the classifier sees synthetic images; independent evaluation partitions remain real.
- 112x112 source TMED images are upsampled to 256x256 for the target architecture, which **does not add native resolution**.
