
def load_feature_name():
    features_310 = '/home/frenky/Documents/Skola/Magistr/diplomka/shouldiclick_thesis/process/urlscan/final_urlscan/new/all/extract/feature_names_310.txt'
    feature_list = []
    with open(features_310) as f:
        for line in f:
            line = line.rstrip()
            if line == '':
                continue
            feature_list.append(line)

    assert len(feature_list) == 310
    return feature_list


def filter_name(text: str):
    text = text.replace('_', '')
    return text

def show_absolute(type_d, key):
    for feature_name, f_index in type_d[key]:
        # print(feature_name)
        json_key_1, type, json_key_2 = feature_name.split('__')
        if json_key_2 == 'len':
            json_key_2 = ''
            json_key_3 = ''
        else:
            json_key_3 = json_key_2
            json_key_2 = 'urls'

        json_key_1 = filter_name(json_key_1)
        json_key_2 = filter_name(json_key_2)

        print('\hline')
        print('{} & {} & {} & {}\\\\'.format(f_index, json_key_1, json_key_2, json_key_3))


def show_categorical(type_d, key):
    for feature_name, f_index in type_d[key]:
        # print(feature_name)

        json_key_1, type, json_key_2, value = feature_name.split('__')

        json_key_1 = filter_name(json_key_1)
        json_key_2 = filter_name(json_key_2)
        value = filter_name(value)


        print('\hline')
        print('{} & {} & {} & {}\\\\'.format(f_index, json_key_1, json_key_2, value))


def show_numerical(type_d, key):
    temp_d = {}
    for feature_name, f_index in type_d[key]:
        # print(feature_name)

        json_key_1, type, json_key_2, value = feature_name.split('__')
        new_key = '__'.join([json_key_1, json_key_2])
        if temp_d.get(new_key, None) is None:
            temp_d[new_key] = [f_index]
        else:
            temp_d[new_key].append(f_index)

    for key, index_list in temp_d.items():
        try:
            json_key_1, json_key_2 = key.split('__')
        except:
            print(key)
            raise KeyError
        json_key_1 = filter_name(json_key_1)
        json_key_2 = filter_name(json_key_2)

        print('\hline')
        print('{} & {} & {} \\\\'.format(','.join(index_list), json_key_1, json_key_2))



index = 1
feature_names = load_feature_name()
type_d = {'categorical': [], 'numerical': [], 'absolute': []}
for feature_name in feature_names:
    split_name = feature_name.split('__')
    type = split_name[1]
    type_d[type].append((feature_name, str(index)))
    index += 1




# show_absolute(type_d, 'absolute')
show_numerical(type_d, 'numerical')
# show_categorical(type_d, 'categorical')