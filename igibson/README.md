#  Time Horizon Based Hierarchical Motion Planning for Navigation and Manipulation


### Requirements
- CUDA compatible GPU for visualizations and CUDA installed
- Anaconda
- Minimum system requirements on the [Installation webpage](http://svl.stanford.edu/igibson/docs/installation.html#installing-the-environment) for iGibson.

### Installation
It's recommended to use Linux as things are tested with Ubuntu 18.04, but it should be fine with a newer Ubuntu version as well.

- Install dependencies
```
apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    g++ \
    libegl-dev
```
- make sure you have cuda installed on your machine and it's on your PATH variable
- check that `nvidia-smi` and `nvcc --version` works well on your device and you don't run into errors while running those.
- Then compile iGibson from the source repository and use a conda enviroment to install igibson on your machine
```
git clone https://github.com/krsandeep98/iGibson --recursive
cd iGibson
git checkout ig-develop
git submodule update

conda create -n igibson python=3.6
conda activate igibson

pip install -e . # This step takes about 4 minutes
```
- Check if igibson is installed on your machine by running `import igibson` inside python in your conda environment
- Download assets and demo data for running the experiments by the following commands
```
python -m igibson.utils.assets_utils --download_assets
python -m igibson.utils.assets_utils --download_demo_data
```

The installation guide on the [iGibson Documentation](http://svl.stanford.edu/igibson/docs/) webpage is a more complete guide, but the steps mentioned above should get the job done for running the experiments here.

### Experiments
<!-- <img src="./docs/images/igibsonlogo.png" width="500"> <img src="./docs/images/igibson.gif" width="250">  -->
After installing iGibson from the installation guide on the [iGibson Documentation](http://svl.stanford.edu/igibson/docs/) webpage, we can get to running different robots finishing navigation and manipulation tasks using the following instructions:

- Make sure to activate your virtual environment `conda activate igibson`
- Head over to `igibson/examples/demo` folder and there you would find multiple demos and you can experiment with those to get a good idea of the wide variety of things which you can do in iGibson
- for running a navigation task, run the following command
```
  python realtime_loop_env_interactive_example_dynamic_nav_rrg_with_changed_global_clock.py
```
- for running a manipulation task, run this instead
```
  python manipulation_realtime_loop_env_interactive_example_dynamic_nav_rrg_with_changed_global_clock.py
```
- 
<!-- iGibson is a simulation environment providing fast visual rendering and physics simulation based on Bullet. iGibson is equipped with fifteen fully interactive high quality scenes, hundreds of large 3D scenes reconstructed from real homes and offices, and compatibility with datasets like CubiCasa5K and 3D-Front, providing 8000+ additional interactive scenes. Some of the features of iGibson include domain randomization, integration with motion planners and easy-to-use tools to collect human demonstrations. With these scenes and features, iGibson allows researchers to train and evaluate robotic agents that use visual signals to solve navigation and manipulation tasks such as opening doors, picking up and placing objects, or searching in cabinets. -->

