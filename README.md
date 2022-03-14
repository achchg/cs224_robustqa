# CS224N default final project (2022 RobustQA track)

## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val`


# Modification for MAML DistillBERT implementation
- `maml.py`: MAML implementation for DistillBERT
  - class `MAML()` with the following methods different from `train.py`:
    - `outer_evaluate`: evaluation method for meta-step (outer-loop)
    - `task`: define task (support/query)
    - `get_task_dataset`: get support and query datasets and make embeddings
    - `inner_loop_train`: MAML inner-loop gradient descent update
    - `inner_loop_query`: define query set for each task and calculate the gradient of loss with query set
    - `inner_loop`: the overall inner_loop step
- Train a MAML: `python maml.py --do-train --eval-every 2000 --run-name maml`
- additional arguments in `args.py`:
  - `oodomain`: whether to use OOD set for training
  - `oodomain_train-dir`: oodomain_train set path
  - `indomain_eval`: whether to use the overall indomain_val during meta-step model evaluation and update
  - `outer_num_epochs`: # of epochs for the outer-loop training
  - `outer_lr`: outer-loop learning rate
  - `n_task`: # of tasks used for inner-loop training
  - `p_oodomain_task`: proportion of task pool from OOD set
  - `k_shot`: # of sample in the support and query set
