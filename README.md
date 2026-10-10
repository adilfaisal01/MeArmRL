# MeArmRL — train a robot arm with reinforcement learning

Course repository for the MATE connectors arm. You will write a reward function,
train a PPO policy in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) with
[skrl](https://skrl.readthedocs.io/), and compare it against a baseline.

**You do not need a GPU to start.** Writing and testing reward code is plain
tensor math and runs on any laptop — Linux, macOS, or Windows. Only the actual
*training* needs an NVIDIA GPU.

## Start here

**Step 1 — check your machine.** This tells you which steps below apply:

```bash
python scripts/doctor.py
```

**Step 2 — set up the CPU environment and explore the rewards** (everyone: no
GPU, no container):

```bash
bash scripts/lab_setup.sh                 # Linux / macOS / Git-Bash
# Windows:  scripts\lab_setup.cmd
python labs/week01_intro_isaaclab/harness/playground.py
```

**Step 3 — train.** Only if you have (access to) an NVIDIA GPU:

- **Local RTX GPU?** `python scripts/bootstrap_gpu.py` sets up the container for
  you in one command (checks, logs in, builds, smoke-tests).
- **Mac or no local GPU?** Train on the university cluster — see `slurm/README.md`
  and the cluster section below.

**Then do the lab:** [`labs/week01_intro_isaaclab/README.md`](labs/week01_intro_isaaclab/README.md)

| Your machine | Step 3 looks like |
| --- | --- |
| Linux or Windows + NVIDIA RTX GPU | `python scripts/bootstrap_gpu.py`, then train locally |
| macOS, or no NVIDIA GPU | build the SIF and submit to the cluster |

> **Stuck?** Every tool here prints the exact command to fix things — start with
> `python scripts/doctor.py`. Maintainers: `AGENTS.md`, `docs/class-onboarding.md`.

## Training the MATE connectors arm (e2e RL)

The arm task is `Mearmrl-Reach-v0`: move the 4-DOF MATE connectors arm's
end-effector (`effector_claw_right`) to a randomly sampled pose. The arm USD
(`source/MeArmRL/MeArmRL/robot_description/mate_connectors/isaac/assets/mate_connectors.usd`)
is committed and verified; no conversion step is needed to train.

> **First time on this machine?** `python scripts/bootstrap_gpu.py` checks your
> GPU/Docker setup, logs in to NGC, builds the image, and smoke-tests it -- one
> command. See `docker/README.md`.

All commands below are prefixed with `/IsaacLab/isaaclab.sh -p` and **only
exist inside the container** — they are not host paths. Run them either via a
one-liner from the `docker/` directory:

```bash
cd docker
docker compose --profile dev run --rm dev /IsaacLab/isaaclab.sh -p scripts/zero_agent.py ...
```

or after dropping into an interactive shell (`docker compose --profile dev run
--rm dev`, see `docker/README.md` for build and X11/GUI setup; the source
tree is bind-mounted so edits take effect without a rebuild).

```bash
# quick sanity check: the arm builds, articulation has 14 joints, env steps
/IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Mearmrl-Reach-v0 --num_envs=4 --headless

# train (PPO via skrl); checkpoints land in logs/skrl/mearmrl-reach/<timestamp>/checkpoints/
/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task=Mearmrl-Reach-v0 --num_envs=1024 --headless --seed=42

# play a trained checkpoint (add X11 mounts and drop --headless to watch it)
/IsaacLab/isaaclab.sh -p scripts/skrl/play.py --task=Mearmrl-Reach-v0 --num_envs=8 \
    --checkpoint=/MeArmRL/logs/skrl/mearmrl-reach/<timestamp>/checkpoints/agent_4800.pt --headless
```

Headless training without an interactive shell — `docker compose up` runs the
`mearmrl` service, configured via env vars (defaults: `Mearmrl-Reach-Student-v0`, 1024
envs, seed 42, 4800 iterations; see `docker/.env.example`):

```bash
cd docker
docker compose up
TASK=Mearmrl-Reach-Student-v0 NUM_ENVS=512 MAX_ITERATIONS=2000 docker compose up
```

Notes:

- `Mearmrl-Reach-v0` drives the 4 actuated joints (`revolute_base`,
  `revolute_left`, `revolute_right`, `rev_end_effector1`); the four-bar
  linkage followers are mimic-driven by PhysX. Rewards track the claw pose
  against a resampled pose command.
- Actuator gains in `mate_connectors_config.py` (stiffness 5000, damping 200)
  are intentionally far above datasheet values: PhysX implicit drives act
  orders of magnitude weaker than configured on these gram-scale links.
- `Template-Mearmrl-v0` is the untouched cartpole template — useful as a
  known-good control when debugging the arm env.
- To regenerate the USD from the URDF (rarely needed; requires a GPU):
  `IsaacLab/isaaclab.sh -p source/MeArmRL/MeArmRL/robot_description/mate_connectors/isaac/convert_urdf_to_usd.py --headless`

### Training on the university HPC cluster

The repo ships Singularity + SLURM job scripts for the university's cluster (see
`slurm/README.md`). With no local NVIDIA GPU, build the image and run your
training there — the job bind-mounts your `source/MeArmRL`, so config edits take
effect without rebuilding the SIF.

## Advanced / reference material

> **Students: you can stop reading here.** Everything below is either reference
> for the arm environment or boilerplate inherited from the Isaac Lab project
> template (manual install, Omniverse extension setup, Pylance tuning). The
> Week-1 lab needs none of it — you use the container (or the cluster) and the
> CPU harness.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/MeArmRL

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/MeArmRL/MeArmRL/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/MeArmRL"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```