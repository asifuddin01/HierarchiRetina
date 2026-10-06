# Released predictions (paper version; unchanged since v1.0)

These files hold the model outputs behind every end-to-end number in the paper. Run
`scripts/reproduce_paper_results.py` to recompute the numbers and compare each one with the
value printed in the paper. Only numpy and pandas are needed, and the run takes about 10 s on a
laptop.

| File | Rows | Content |
|---|---|---|
| `cascade_test_predictions.csv` | 58,689 | Cascade outputs for every test image. **No reference labels.** |
| `stage3_oof_predictions.csv` | 18,570 | LG-DRG out-of-fold probabilities on the development images, with their reference grade (anonymous rows, no image IDs). The Stage III decision thresholds are fitted on this file only. |
| `SHA256SUMS` | | SHA-256 checksums of the two files above. |

## `cascade_test_predictions.csv`

| Column | Meaning |
|---|---|
| `image_id` | File name without extension, as distributed by the dataset provider. |
| `dataset` | `eyepacs`, `ddr`, `aptos`, `fgadr`, `messidor` (Messidor-2) or `idrid`. |
| `stage1_prob` | Stage I probability that the image shows DR (Grades 1 to 5). |
| `stage1_passed` | 1 if `stage1_prob` ≥ 0.2391715943813324, the operating point fixed on validation data. The image then went through Stages II and III. |
| `stage3_p_grade1` … `stage3_p_grade5` | Five-fold LG-DRG ensemble probabilities for Grades 1 to 5 (Grade 5 = ungradable). Empty for images stopped at Stage I. |
| `stage3_grade_argmax` | Stage III grade by argmax. |
| `stage3_grade` | Stage III grade by the consecutive CORN decode with the OOF-fitted thresholds t = (0.38, 0.60, 0.19), t_g = 0.10. |
| `final_grade` | Cascade output: `stage3_grade` for routed images, 0 for images stopped at Stage I. This is the grade scored in the paper. |
| `final_grade_argmax` | The same with the argmax decode. |

## Reference labels

The labels belong to the dataset providers and are not redistributed here. To reproduce the
results, build a CSV with one row per test image and two columns:

- `image`: the image ID, with or without its file extension.
- `grade`: the reference grade, 0 to 4, or 5 for ungradable.

The test images are exactly the `image_id` values in `cascade_test_predictions.csv`, so no
re-splitting is needed. Take each grade from the provider's official label file:

| Dataset | Test images | Official source of the grades |
|---|---|---|
| EyePACS | 53,576 | Kaggle *Diabetic Retinopathy Detection*, test-set solution file (`image`, `level`) |
| DDR | 4,105 | DDR grading split, `test.txt` (Grade 5 = ungradable) |
| APTOS 2019 | 366 | Kaggle *APTOS 2019 Blindness Detection*, `train.csv` (`id_code`, `diagnosis`); a random 10% held out, as listed here |
| Messidor-2 | 262 | Adjudicated DR grades of Krause *et al.* (2018); a random 15% held out, as listed here |
| FGADR | 277 | Seg-set grading labels; a random 15% held out, as listed here |
| IDRiD | 103 | Disease-grading test labels (`Retinopathy grade`) |

The script accepts common column names for the ID (`image`, `image_id`, `id_code`, `id`, …) and
for the grade (`grade`, `level`, `diagnosis`, `label`, …). The repository's
`data/test/test_grade.csv` already has the expected format.

## Licence

These prediction files are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), because the underlying datasets
restrict commercial use. The code in this repository is MIT-licensed.
