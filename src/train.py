from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0).fit(X, y)

    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    with open('src/model.onnx', 'wb') as f:
        f.write(onx.SerializeToString())


if __name__ == '__main__':
    main()
