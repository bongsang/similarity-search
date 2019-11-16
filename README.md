<h1>
<a href="https://www.linkedin.com/in/bongsang/">
<img src="https://media.licdn.com/dms/image/C5103AQE834J0PWkG5g/profile-displayphoto-shrink_200_200/0?e=1579132800&v=beta&t=Vc3geyDnIDnn1HiFVT26-VP-qqyJZH_BGfZPtcJpk9U" width=80 align=middle></a>
Geological Multi Classification
</h1>

## Solution for Amazon
<img src="tests/andesite1.jpg" height=150> <img src="tests/gneiss1.jpg.jpg" height=150>

## Usage: Training
#### Everything is automated. 
- You don't need to any manual configuration. Just enter python main.py
```python
python main.py  # donwload, dataset configuration, training and model saving
python main.py --epochs=100
python main.py --epochs=100 --batch_size=64
main_training.py
```

## Usage: Predicting
### Please place test images that you want to inspect in tests directory
```python
python main_predict.py # There are some images for testing in tests directory default.
python main.py --test_path=your/directory  # You can change default test directory
```

## Training results
![accuracy](results/1573767030_accuracy.jpg)
![loss](results/1573767030_loss.jpg)


# Thank you!
