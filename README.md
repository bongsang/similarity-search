<h1>
<a href="https://www.linkedin.com/in/bongsang/">
<img src="https://media.licdn.com/dms/image/C5103AQE834J0PWkG5g/profile-displayphoto-shrink_200_200/0?e=1579132800&v=beta&t=Vc3geyDnIDnn1HiFVT26-VP-qqyJZH_BGfZPtcJpk9U" width=80 align=middle></a>
Geological Multi Classification
</h1>


## Amazon
<img src="tests/andesite1.jpg" height=150> <img src="tests/gneiss1.jpg.jpg" height=150> <img src="tests/marble1.jpg" height=150>

## Usage: Training
```python
python main.py
python main.py --epochs=100
python main.py --epochs=100 --batch_size=64
# You can seen various options in main.py source code.
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
