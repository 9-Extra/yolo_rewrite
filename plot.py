from matplotlib import pyplot
import numpy
import pandas


def main():
    feature_layer_result = pandas.read_csv("summary/feature_layer_result.csv")
    # names, auroc, fpr95 = feature_layer_result
    print(feature_layer_result)
    pyplot.figure("show")
    pyplot.plot(feature_layer_result["name"], feature_layer_result["auroc"])
    pyplot.show()


if __name__ == '__main__':
    main()
