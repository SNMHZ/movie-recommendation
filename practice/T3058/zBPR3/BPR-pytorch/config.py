# dataset name 
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

# paths
main_path = '/opt/ml/input/data/train/'

train_rating = main_path + 'train_ratings.csv'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = './models/'
BPR_model_path = model_path + 'NeuMF.pth'
