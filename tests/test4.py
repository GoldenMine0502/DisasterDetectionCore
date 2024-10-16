import os

folders = os.listdir('../../datasets/incidents/images')

names = sorted(list(folders))

classes = ['fire', 'smoke', 'bicycle', 'blizzard', 'blocked']

for name in names:
    print(name)