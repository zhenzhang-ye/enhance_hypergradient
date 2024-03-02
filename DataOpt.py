import os
import numpy as np
import jax.numpy as jnp
from libsvm.svmutil import svm_read_problem

class Dataset:
    def __init__(self, args):
        file_path = os.path.join(args.data_path, args.dataset + '.txt')
        y, x = svm_read_problem(file_path)

        # preprocessing
        self.label, self.feat = self.preprocessing(x, y, args)
        
        # get the test dataset for logistic regression
        if args.outer_function == 'logistic_regression':
            file_path = os.path.join(args.data_path, args.dataset + '_test.txt')
            y, x = svm_read_problem(file_path)
            self.label_t, self.feat_t = self.preprocessing(x, y, args)
        elif args.outer_function in ['quadratic', 'linear']:
            self.A_out = jnp.array(np.random.randn(len(x[0]), len(x[0])))
            self.b_out = jnp.array(np.random.randn(len(x[0])))

        self.x_init = jnp.array(np.random.randn(self.feat.shape[-1])) * 2.
        if args.y_scale == "-1_1":
            self.y_init = jnp.array(np.random.rand(self.feat.shape[-1])) * 2 - 1
        elif args.y_scale == "3_6":
            self.y_init = jnp.array(np.random.rand(self.feat.shape[-1])) * 3 + 3

    def preprocessing(self, x, y, args):
        label = np.zeros((len(y)))
        feat = np.zeros((len(x), len(x[0])))
        count_val = 0
        for i, feature in enumerate(x):
            np_feature = np.array(list(feature.items()))[:, -1]
            if np_feature.shape[0] < len(x[0]):
                continue
            label[count_val] = y[i]
            feat[count_val, np.fromiter(feature.keys(), dtype=int)-1] = np_feature
            count_val += 1
        label = jnp.array(label[:count_val])
        feat = jnp.array(feat[:count_val])
        if 'liver' in args.dataset:
            label = 2 * label - 1
        if args.demo:
            return label[:15], feat[:15]
        else:
            return label, feat