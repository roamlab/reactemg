# ReactEMG: Zero-Shot, Low-Latency Intent Detection via sEMG

[Project Page](https://reactemg.github.io/) | [arXiv](https://arxiv.org/abs/2506.19815) | [Video](https://youtu.be/AKT8hMvVCGY)

[Runsheng Wang](http://runshengwang.com/)<sup>1</sup>,
[Xinyue Zhu](https://reactemg.github.io/)<sup>1</sup>,
[Ava Chen](https://avachen.net/),
[Jingxi Xu](https://jxu.ai/),
[Lauren Winterbottom](https://reactemg.github.io/),
[Dawn M. Nilsen](https://www.vagelos.columbia.edu/profile/dawn-nilsen-edd)<sup>2</sup>,
[Joel Stein](https://www.neurology.columbia.edu/profile/joel-stein-m-d)<sup>2</sup>,
[Matei Ciocarlie](https://www.me.columbia.edu/faculty/matei-ciocarlie)<sup>2</sup>

<sup>1</sup>Equal contribution,
<sup>2</sup>Co-Principal Investigators

Columbia University

<div style="margin:50px; text-align: justify;">
<img style="width:100%;" src="assets/demo.gif">

ReactEMG is a zero-shot, low-latency EMG framework that segments forearm signals in real time to predict hand gestures at every timestep, delivering calibration-free, high-accuracy intent detection ideal for controlling prosthetic and robotic devices.

## :package: Installation
Clone the repo with `--recurse-submodules` and install our conda (mamba) environment on an Ubuntu machine with a NVIDIA GPU. We use Ubuntu 24.04 LTS and Python 3.11. 

```bash
mamba env create -f environment.yml
```

Install [PyTorch](<https://pytorch.org/get-started/locally/>) in the conda environment, then install wandb via pip:

```bash
pip install wandb
```

Lastly, install minLoRA via:

```bash
cd minLoRA && pip install -e .
```

minLorRA was built for editable install with `setup.py develop`, which is deprecated. Consider enabling `--use-pep517` and use `setuptools ≥ 64` when working with `pip ≥ 25.3`.

## :floppy_disk: Datasets

### 1. ROAM-EMG 
We are open-sourcing our sEMG dataset, **ROAM-EMG**.  
- **Scope:** Using the Thalmic Myo armband, we recorded eight-channel sEMG signals from 28 participants as they performed hand gestures in four arm postures, followed by two grasping tasks and three types of arm movement. Full details of the dataset are provided in our paper and its supplementary materials.
- **Download:** [Dropbox Link](<https://www.dropbox.com/scl/fi/19zvl12vn27wsnzsmw0vx/ROAM_EMG.zip?rlkey=x6gtygdfz24i8efdswr1exii7&st=ljzuoire&dl=0>)  

### 2. Pre-processed public datasets  
For full reproducibility, we also provide pre-processed versions of all public EMG dataset used in the paper. The file structures and data formats have been aligned with ROAM-EMG. We recommend organizing all datasets under the `data/` folder (automatically created with the command below) in the root directory of the repo. To download all datasets (including ROAM-EMG): 

```bash
curl -L -o data.zip "https://www.dropbox.com/scl/fi/isj4450alriqjfstkna2s/data.zip?rlkey=n5sf910lopskewzyae0vgn6j7&st=vt89hfpj&dl=1" && unzip data.zip && rm data.zip
```

## :hammer_and_wrench: Training

### Logging

We use W&B to track experiments. Decide whether you want metrics online or offline:

```bash
# online (default) – set once in your shell
export WANDB_PROJECT=my-emg-project
export WANDB_ENTITY=<your-wandb-username>

# or completely disable
export WANDB_MODE=disabled
```

### Pre-training with public datasets

Use the following command to pre-train our model on EMG-EPN-612 and other public datasets:

```bash
python3 main.py \
  --embedding_method linear_projection \
  --use_input_layernorm \
  --task_selection 0 1 2 \
  --offset 30 \
  --share_pe \
  --num_classes 3 \
  --use_warmup_and_decay \
  --dataset_selection pub_with_epn \
  --window_size 600 \
  --val_patient_ids s1 \
  --epn_subset_percentage 1.0 \
  --model_choice any2any \
  --inner_window_size 600 \
  --exp_name <RUN_ID>
```

Replace <RUN_ID> with your desired name, and the script will save checkpoints to `model_checkpoints/<RUN_ID>_<timestamp>_<machine_name>/epoch_<N>.pth`, where `<timestamp>` records the run’s start time and `<machine_name>` identifies the host. Ensure you have write permission where you launch the job.

You may also initialize weights from a saved checkpoint by adding `--saved_checkpoint_pth path/to/epoch_X.pth` to the training command. If you wish to fine-tune a model via LoRA, provide the flag `--use_lora 1`, in addition to the locally saved checkpoint path.

To train EPN-only models for evaluation purposes, set `--dataset_selection epn_only`

If this is your first time using W&B on your machine, you will be prompted to provide credentials:

```text
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```

Enter `2` to use your W&B account, and follow the prompts to provide your API key.

### Fine-tuning on ROAM-EMG

Fine-tuning follows a leave-one-subject-out (LOSO) protocol. The helper script `finetune_runner.sh` trains a separate model for every subject in the ROAM-EMG dataset. Open `finetune_runner.sh` and set `saved_checkpoint_pth` to be your pre-trained checkpoint path, and start LOSO fine-tuning via:

```bash
source finetune_runner.sh
```

## :bar_chart: Evaluation

We evaluate model performance on two metrics:
- **Raw accuracy**: Per-timestep correctness across the entire EMG recording
- **Transition accuracy**: Event-level score that captures accuracy and stability

During evaluation, we run the model exactly as how it would run online: windows slide forward in real time and predictions are aggregated live. This gives a realistic view of online performance instead of an offline, hindsight-only score.

### Run the evaluation
```bash
python3 event_classification.py \
  --eval_task predict_action \
  --files_or_dirs ../data/ROAM_EMG \
  --allow_relax 0 \
  --buffer_range 200 \
  --stride 1 \
  --lookahead 50 \
  --weight_max_factor 1.0 \
  --likelihood_format logits \
  --samples_between_prediction 20 \
  --maj_vote_range future \
  --saved_checkpoint_pth <path_to_your_pth_checkpoint> \
  --epn_eval 0 \
  --verbose 1 \
  --model_choice any2any
```
To remove all smoothing, set `--stride 20`, `--lookahead 0`, `--samples_between_prediction 1`, and `--maj_vote_range single`.

To evaluate EPN-only models, set `--files_or_dirs ../data/EMG-EPN-612` and `--epn_eval 1`.

### Output

The evaluation code produces three outputs under `output/`:
- Summary txt: Overall raw & transition accuracy (mean ± std), event counts, and a tally of failure reasons.
- Per-file JSON: Metrics plus full ground-truth & prediction sequences for each file.
- PNG plots: 3-panel figure: 8-channel EMG, ground-truth labels, and model predictions over time.

## :memo: Citation
If you find this codebase useful, consider citing:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2506.19815,
  doi = {10.48550/ARXIV.2506.19815},
  url = {https://arxiv.org/abs/2506.19815},
  author = {Wang,  Runsheng and Zhu,  Xinyue and Chen,  Ava and Xu,  Jingxi and Winterbottom,  Lauren and Nilsen,  Dawn M. and Stein,  Joel and Ciocarlie,  Matei},
  keywords = {Robotics (cs.RO),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {ReactEMG: Zero-Shot,  Low-Latency Intent Detection via sEMG},
  publisher = {arXiv},
  year = {2025},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## :email: Contact
For questions or support, please email Runsheng at runsheng.w@columbia.edu

## :scroll: License
This project is released under the MIT License; see the [License](LICENSE) file for full details.

## :handshake: Acknowledgments
This work was supported in part by an Amazon Research Award and the Columbia University Data Science Institute Seed Program. Ava Chen was supported by NIH grant 1F31HD111301-01A1. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of the sponsors. We would like to thank Katelyn Lee, Eugene Sohn, Do-Gon Kim, and Dilara Baysal for their assistance with the hand orthosis hardware. We thank Zhanpeng He and Gagan Khandate for their helpful feedback and insightful discussions.

