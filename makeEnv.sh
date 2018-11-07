set -e
rm -rf env
virtualenv -p $(which python3) env
. ./env/bin/activate
pip install -r requirements.txt
cd gym-unicycle
pip install -e .
cd ../
echo "done"
