import unittest
import numpy as np
from reval import relative_validation, config
from reval.relative_validation import RelativeValidation


class TestReval(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.s = config.CLASSIFIER
        cls.c = config.CLUSTERING
        cls.nrand = config.RNDLABELS_ITER
        cls.reval_cls = RelativeValidation(cls.s, cls.c, cls.nrand)

    def test_revaltraining(self):
        data_tr = np.array([[0] * 10,
                            [1] * 10] * 20)
        misclass, model, labels = self.reval_cls.train(data_tr)
        # self.assertSequenceEqual([misclass] + [lab.tolist() for lab in labels.values()],
        #                          [0.0, [1, 0] * 20, [1, 0] * 20])
        self.assertSequenceEqual([misclass] + [labels.tolist()],
                                 [0.0, [1, 0] * 20])
        self.assertEqual(type(model), type(self.s))

    def test_revaltest(self):
        data_tr = np.array([[0] * 10,
                            [1] * 10] * 20)
        data_ts = np.array([[0] * 10,
                            [1] * 10] * 10)
        model = self.s.fit(data_tr, [1, 0] * 20)
        misclass, labels = self.reval_cls.test(data_ts, model)
        # self.assertSequenceEqual([misclass] + [lab.tolist() for lab in labels.values()],
        #                          [0.0, [1, 0] * 10, [1, 0] * 10])
        self.assertSequenceEqual([misclass] + [labels.tolist()],
                                 [0.0, [1, 0] * 10])

    def test_rndlabels(self):
        data_tr = np.array([[0] * 10,
                            [1] * 10] * 20)
        labels_tr = np.array([0, 1] * 20)
        data_ts = np.array([[0] * 10,
                            [1] * 10] * 10)
        labels_ts = np.array([0, 1] * 10)
        misclass_ts = self.reval_cls.rndlabels_traineval(data_tr,
                                                         data_ts,
                                                         labels_tr,
                                                         labels_ts)
        self.assertTrue(misclass_ts > 0)

    def test_khun_munkres_algorithm(self):
        true_lab = np.array([1, 1, 1, 0, 0, 0])
        pred_lab = np.array([0, 0, 0, 1, 1, 1])
        new_lab = relative_validation._kuhn_munkres_algorithm(true_lab, pred_lab)
        self.assertSequenceEqual(new_lab.tolist(), [1, 1, 1, 0, 0, 0])


if __name__ == 'main':
    unittest.main()
