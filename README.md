# learning-machine-learning
various machine learning exercises

# Create a virtual environment
python -m venv .venv \
pip install -r requirements.txt

# Training network for digit ocr
Run training with mnist data: \
tensorflow-mnist-digit-ocr.py

# Network diagnostics using tensorboard
run: tensorboard --logdir logs/fit
<div align="left">
  <img src="./doc/tensorboard.png"  width="10%" height="10%"/>
</div>

# Test with own image
Create a test image using the iOS App Paint 98 as shown below.\
The generated 28 x 28 picture can then be classified with tensorflow-digit-ocr-predict.py path_to_image
<div align="left">
  <img src="./doc/IMG_0357.PNG"  width="10%" height="10%"/>
  <img src="./doc/IMG_0358.PNG"  width="10%" height="10%"/>
  <img src="./doc/IMG_0359.PNG"  width="10%" height="10%"/>
  <img src="./doc/IMG_0360.PNG"  width="10%" height="10%"/>
</div>

Output should look as follows:\
<div align="left">
  <img src="./doc/predict-output.png"  width="25%" height="25%"/>
</div>

