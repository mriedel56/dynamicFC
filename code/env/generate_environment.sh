# create a miniconda environment
conda create -n dynamicFC pip python=3.11.3
source activate dynamicFC
 
file_path="$(dirname -- "${BASH_SOURCE[0]}")"
file_path="$(cd -- "$file_path" && pwd)"
project_path=$(dirname $(dirname $file_path))
 
python -m pip install -r $project_path/code/env/requirements.txt
 
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
touch $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
touch $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
 
echo export PYTHONPATH=$PYTHONPATH:$project_path/code >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo export project_path=$project_path >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
 
echo unset PYTHONPATH >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo unset project_path >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh