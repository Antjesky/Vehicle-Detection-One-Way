# Projekt Vihicle-Detection-One-Way

This repository is for [Visual Studio Code] and aims to extract vehicles from a video stream, recognise them and classify them by vehicle type, including trucks, cars, mopeds and buses.
VEHICLE-DETECTION-ONE-WAY uses the DETR model from Facebook, which uses a Transformer-based architecture. It is supplemented by the instance segmentation part and is thus supposed to be able to recognise and segment vehicles in images by identifying and marking each vehicle as a separate instance.

The repository is based on the work of HuggingFace Transformers, a Python library that implements several state-of-the-art AI algorithms.  The repository builds on the Transformers tutorials https://github.com/NielsRogge/Transformers-Tutorials.

# DETR model from Facebook

The DERT model is explained at
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Evaluating_DETR_on_COCO_validation_2017.ipynb

# MaskFormer instance segmentation model

The Maskformer model is described in detail at
https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MaskFormer
https://github.com/NielsRogge/mmdetection/tree/master/configs/maskformer


# Licence

All licence conditions contained therein apply to this repository.



# python-template

Precondition:
Windows users can follow the official microsoft tutorial to install python, git and vscode here:

- ​​https://docs.microsoft.com/en-us/windows/python/beginners
- german: https://docs.microsoft.com/de-de/windows/python/beginners

## Visual Studio Code

This repository is optimized for [Visual Studio Code](https://code.visualstudio.com/) which is a great code editor for many languages like Python and Javascript. The [introduction videos](https://code.visualstudio.com/docs/getstarted/introvideos) explain how to work with VS Code. The [Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial) provides an introduction about common topics like code editing, linting, debugging and testing in Python. There is also a section about [Python virtual environments](https://code.visualstudio.com/docs/python/environments) which you will need in development. There is also a [Data Science](https://code.visualstudio.com/docs/datascience/overview) section showing how to work with Jupyter Notebooks and common Machine Learning libraries.

The `.vscode` directory contains configurations for useful extensions like [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens0) and [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python). When opening the repository, VS Code will open a prompt to install the recommended extensions.

## Development Setup

Open the [integrated terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) and run the setup script for your OS (see below). This will install a [Python virtual environment](https://docs.python.org/3/library/venv.html) with all packages specified in `requirements.txt`.

### Linux and Mac Users

1. run the setup script: `./setup.sh` or `sh setup.sh`
2. activate the python environment: `source .venv/bin/activate`
3. run example code: `python src/hello.py`
4. install new dependency: `pip install sklearn`
5. save current installed dependencies back to requirements.txt: `pip freeze > requirements.txt`

### Windows Users

1. run the setup script `.\setup.ps1`
2. activate the python environment: `.\.venv\Scripts\Activate.ps1`
3. run example code: `python src/hello.py`
4. install new dependency: `pip install sklearn`
5. save current installed dependencies back to requirements.txt: `pip freeze > requirements.txt`

Troubleshooting:

- If your system does not allow to run powershell scripts, try to set the execution policy: `Set-ExecutionPolicy RemoteSigned`, see https://www.stanleyulili.com/powershell/solution-to-running-scripts-is-disabled-on-this-system-error-on-powershell/
- If you still cannot run the setup.ps1 script, open it and copy all the commands step by step in your terminal and execute each step
