# AWS
1. https://aws.amazon.com/
2. Click on EC2 for launching a new virtual server.
3. Select “US West Orgeon” from the drop-down in the top right hand corner. 
4. Click the “Launch Instance” button.
5. Click “Community AMIs”. 
6. Enter “ami-dfb13ebf”
7. Scroll down and select the “g2.2xlarge” hardware.

~~~
Login as ec2-user

/* Optional 
aws configure list
aws configure
aws s3 ls s3://fruitclassifierdata
aws s3 cp s3://fruitclassifierdata /home/ec2-user --recursive

sudo pip install virtualenv
sudo virtualenv -p /usr/bin/python2.7 newenv
source newenv/bin/activate
*/

sudo pip install -q -U pip setuptools wheel --> Doesn't seems to be working. It actually remove pip

sudo pip install django
sudo pip install scikit-learn
sudo pip install Pillow
sudo pip install seaborn
sudo pip install matplotlib
sudo pip install h5py
sudo pip install --upgrade keras
sudo pip install --upgrade tensorflow

screen
sudo python manage.py runserver 0.0.0.0:8080

//deactivate

~~~
### b9_project\b9_project\settings.py
1. Ensure that DEBUG = True
2. ALLOWED_HOSTS = ['Host address or IP']
