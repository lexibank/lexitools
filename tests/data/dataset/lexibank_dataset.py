import pathlib

from pylexibank import Dataset


class TestDataset(Dataset):
    id = 'dataset'
    dir = pathlib.Path(__file__).parent
