import pickle
import numpy as np

def load_articles(obj):
    print('Dataset: ', obj)
    print("loading news articles")

    # Load training and original test data
    train_dict = pickle.load(open('data/news_articles/' + obj + '_train.pkl', 'rb'))
    test_dict = pickle.load(open('data/news_articles/' + obj + '_test.pkl', 'rb'))

    # Load adversarial test sets A, B, C, D
    restyle_dict_a = pickle.load(open('data/adversarial_test/' + obj + '_test_adv_A.pkl', 'rb'))  # Objective
    restyle_dict_b = pickle.load(open('data/adversarial_test/' + obj + '_test_adv_B.pkl', 'rb'))  # Neutral
    restyle_dict_c = pickle.load(open('data/adversarial_test/' + obj + '_test_adv_C.pkl', 'rb'))  # Emotionally Triggering
    restyle_dict_d = pickle.load(open('data/adversarial_test/' + obj + '_test_adv_D.pkl', 'rb'))  # Sensational

    # Extract training and test data
    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']

    # Extract adversarial test sets
    x_test_a = restyle_dict_a['news']
    x_test_b = restyle_dict_b['news']
    x_test_c = restyle_dict_c['news']
    x_test_d = restyle_dict_d['news']

    return x_train, x_test, x_test_a, x_test_b, x_test_c, x_test_d, y_train, y_test

def load_reframing(obj):
    print("loading news augmentations")
    print('Dataset: ', obj)

    restyle_dict_train1_1 = pickle.load(open('data/reframings/' + obj + '_train_objective.pkl', 'rb'))
    restyle_dict_train1_2 = pickle.load(open('data/reframings/' + obj + '_train_neutral.pkl', 'rb'))
    restyle_dict_train2_1 = pickle.load(open('data/reframings/' + obj + '_train_emotionally_triggering.pkl', 'rb'))
    restyle_dict_train2_2 = pickle.load(open('data/reframings/' + obj + '_train_sensational.pkl', 'rb'))

    finegrain_dict1 = pickle.load(open('data/veracity_attributions/' + obj + '_fake_standards_objective_emotionally_triggering.pkl', 'rb'))
    finegrain_dict2 = pickle.load(open('data/veracity_attributions/' + obj + '_fake_standards_neutral_sensational.pkl', 'rb'))

    x_train_res1 = np.array(restyle_dict_train1_1['rewritten'])
    x_train_res1_2 = np.array(restyle_dict_train1_2['rewritten'])
    x_train_res2 = np.array(restyle_dict_train2_1['rewritten'])
    x_train_res2_2 = np.array(restyle_dict_train2_2['rewritten'])

    y_train_fg, y_train_fg_m, y_train_fg_t = finegrain_dict1['orig_fg'], finegrain_dict1['mainstream_fg'], finegrain_dict1['tabloid_fg']
    y_train_fg2, y_train_fg_m2, y_train_fg_t2 = finegrain_dict2['orig_fg'], finegrain_dict2['mainstream_fg'], finegrain_dict2['tabloid_fg']

    replace_idx = np.random.choice(len(x_train_res1), len(x_train_res1) // 2, replace=False)

    x_train_res1[replace_idx] = x_train_res1_2[replace_idx]
    x_train_res2[replace_idx] = x_train_res2_2[replace_idx]
    y_train_fg[replace_idx] = y_train_fg2[replace_idx]
    y_train_fg_m[replace_idx] = y_train_fg_m2[replace_idx]
    y_train_fg_t[replace_idx] = y_train_fg_t2[replace_idx]

    return x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t