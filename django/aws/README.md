# AWS
1. https://aws.amazon.com/
2. Click on EC2 for launching a new virtual server.
3. Select “US West Orgeon” from the drop-down in the top right hand corner. 
4. Click the “Launch Instance” button.
5. Click “Community AMIs”. 
6. Enter “ami-dfb13ebf”
7. Scroll down and select the “g2.2xlarge” hardware.

~~~
aws configure list
aws configure
aws s3 ls s3://fruitclassifierdata
aws s3 cp s3://fruitclassifierdata /home/ec2-user --recursive

sudo pip install virtualenv
sudo virtualenv newenv
sudo source newenv/bin/activate

sudo pip install -q -U pip setuptools wheel
sudo pip install django
sudo pip3 install scikit-learn
sudo pip3 install Pillow
sudo pip3 install seaborn
sudo pip3 install matplotlib
sudo pip install h5py
sudo pip3 install --upgrade keras
sudo pip install --upgrade tensorflow

deactivate

python3 04_cnn.py
~~~
