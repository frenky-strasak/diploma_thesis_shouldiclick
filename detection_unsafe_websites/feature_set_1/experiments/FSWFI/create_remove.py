
def load_all_feature():
    path = '/home/frenky/Documents/Skola/Magistr/diplomka/shouldiclick_thesis/2018_process/old/experiments/feature_names.txt'
    all_features = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line == '':
                continue
            all_features.append(line)
    return all_features

# root_folder = 'xbboost_1'
root_folder = 'randomforest_1'



def load_good_features():
    path = root_folder + '/best_features_only_names.txt'
    best_features = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line == '':
                continue
            best_features.append(line)
    return best_features


all_features = load_all_feature()
print('all features: {}'.format(len(all_features)))
best_features = load_good_features()
print('all features: {}'.format(len(best_features)))


with open(root_folder + '/removed_features.txt', 'w') as f:
    for feature in all_features:
        if feature not in best_features:
            f.write(feature + '\n')
