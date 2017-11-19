# Django Web Server

# Setup
~~~
pip3 install django
pip3 install theano tensorflow keras
pip3 install h5py
pip3 install pillow
pip3 install scikit-learn
pip3 install matplotlib
pip3 install seaborn
~~~

# Startup
~~~
cd b9_project
python manage.py runserver 8080

Using TensorFlow backend.
loading model
2017-11-19 17:21:44.195086: I C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
model ready
Using TensorFlow backend.
loading model
2017-11-19 17:21:46.832861: I C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
model ready
Performing system checks...

System check identified no issues (0 silenced).

You have 13 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions.
Run 'python manage.py migrate' to apply them.
November 19, 2017 - 17:21:48
Django version 1.11.7, using settings 'b9_project.settings'
Starting development server at http://127.0.0.1:8080/
Quit the server with CTRL-BREAK.
~~~

# Access the Demo Page
* [http://127.0.0.1:8080/fruitclass/](http://127.0.0.1:8080/fruitclass/)
