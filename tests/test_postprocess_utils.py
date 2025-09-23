from sahi.postprocess.utils import ObjectPredictionList


class TestPostprocessUtils:
    def setup_method(self):
        self.test_input = [ObjectPredictionList([1, 2, 3, 4])]

    def test_get_item_int(self):
        obj = self.test_input[0]
        assert obj[0].tolist() == 1

    def test_len(self):
        obj = self.test_input[0]
        assert len(obj) == 4

    def test_extend(self):
        import torch

        obj = self.test_input[0]
        obj.extend(ObjectPredictionList([torch.randn(1, 2, 3, 4)]))
        assert len(obj) == 5

    def test_tostring(self):
        obj = self.test_input[0]
        assert str(obj) == str([1, 2, 3, 4])
