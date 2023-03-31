"""
Reads all the story files in a folder and writes it into one text file.
The two folder paths are for the cnn and daily mail articles respectively.
The folders were downloaded from https://cs.nyu.edu/~kcho/DMQA/.
"""
import os

path1 = "dailymail_stories\dailymail_stories\dailymail\stories"
path2 = "cnn_stories\cnn_stories\cnn\stories"
os.chdir(path1)

with open(os.path.join(path1, "articles.txt"), 'a', encoding='utf-8') as write_file:

    for f in os.listdir():
        if f.endswith(".story"):
            with open(f, 'r', encoding='utf-8') as read_file:
                txt = read_file.read()
                write_file.write(txt)
    write_file.close()
