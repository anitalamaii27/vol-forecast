def make_folds(split, n, step=63):
    folds = []
    start = split
    while start < n:
        end = min(start + step, n)
        folds.append((start, end))
        start = end 
    return folds


if __name__ == "__main__":
    folds = make_folds(3370, 4213)
    print(len(folds), folds[0], folds[-1])