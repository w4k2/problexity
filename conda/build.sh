#! /bin/bash -e 

# before running anaconda should be installed on computer


rm -rf problexity/

# create temporary environment (current user env can cause conflits with packages we want to install)
conda create -y -n tmp-conda-upload python">=3.10"
eval "$(conda shell.bash hook)"
conda activate tmp-conda-upload
conda install -y conda-build anaconda-client grayskull -c conda-forge

# build conda package from the latest pypi release
grayskull pypi problexity 
sed -i 's/igraph/python-igraph/' problexity/meta.yaml # for some reason name of this package is different in pypi and in anaconda
conda-build problexity -c conda-forge

# upload package to anaconda
echo "login to anaconda"
anaconda whoami &> /tmp/conda_user.txt
CURRENT_ANACONDA_USER=`cat /tmp/conda_user.txt | cut -d$'\n' -f2`
if [ "$CURRENT_ANACONDA_USER" = "Anonymous User" ]; then
    anaconda login
fi
CONDA_INSTALLATION_DIR=`which conda | xargs dirname | xargs dirname`
LATEST_PACKAGE=`ls -t $CONDA_INSTALLATION_DIR/conda-bld/noarch/problexity-* | head -1`
anaconda upload -u w4k2 $LATEST_PACKAGE

# cleanup
conda deactivate
conda env remove -y -n tmp-conda-upload