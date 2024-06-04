from matplotlib import pyplot
import numpy
import pandas


def main():
    feature_layer_result = pandas.read_csv("summary/feature_layer_result.csv")
    # names, auroc, fpr95 = feature_layer_result
    print(feature_layer_result)
    auroc = feature_layer_result["auroc"]
    pyplot.figure("show")
    pyplot.barh(feature_layer_result["name"], auroc)
    pyplot.show()


if __name__ == '__main__':
    main()
