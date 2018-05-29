import numpy as np
import csv

labels=[]
dataset = []
def load_data():
    with open('plus_images.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data = {"image": row[1:], "label":[1,0,0,0]}
            dataset.append(data)

    with open('minus_images.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data = {"image": row[1:], "label":[0,1,0,0]}
            dataset.append(data)
            

    with open('divide_images.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data = {"image": row[1:], "label":[0,0,1,0]}
            dataset.append(data)

    with open('multiply_images.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data = {"image": row[1:], "label":[0,0,0,1]}
            dataset.append(data)
            
            
load_data()
dataset = np.array(dataset)
np.random.shuffle(dataset)

np.save("dataset", dataset)