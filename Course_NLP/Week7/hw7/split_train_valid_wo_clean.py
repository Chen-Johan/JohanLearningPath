import random

"""Split the training data into training and validation sets"""

def split_train_valid(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
    random.shuffle(lines)
    
    num_lines = len(lines)
    num_train = int(num_lines * 0.8)
    
    train_lines = lines[:num_train]
    valid_lines = lines[num_train:]
    
    with open("train.csv", "w", encoding="utf-8") as f_train:
        f_train.writelines(train_lines)
    
    with open("valid.csv", "w", encoding="utf-8") as f_valid:
        f_valid.writelines(valid_lines)
    
    return

if __name__ == "__main__":
    split_train_valid("data.csv")


    