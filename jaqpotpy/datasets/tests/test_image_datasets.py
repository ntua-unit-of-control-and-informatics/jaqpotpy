"""
Tests for Jaqpotpy Image Datasets.
"""
import unittest
from jaqpotpy.datasets import TorchImageDataset
from torchvision import transforms
import random


class TestImageDatasets(unittest.TestCase):

    png_path = "C:/Users/jason/centralenv/Jaqpot_Tests/faces/faces"

    @unittest.skip('Local data')
    def test_png(self):
        featurizer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        dataset = TorchImageDataset(path=self.png_path,
                                    y=[random.randint(0, 1) for _ in range(69)],
                                    featurizer=featurizer,
                                    task="classification",
                                    y_name="Random",
                                    images_name="Faces"
                                    )

        dataset.create()
        assert [item[1] for item in list(dataset.data)] == dataset.ys


if __name__ == '__main__':
    unittest.main()