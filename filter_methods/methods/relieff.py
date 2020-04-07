from weka.attribute_selection import ASSearch
from weka.attribute_selection import ASEvaluation
from weka.attribute_selection import AttributeSelection


def relieff(filter_data, feature_names):
    # define search and evaluation for ReliefF
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
    # last param is number of nearest neighbors
    evaluation = ASEvaluation(classname="weka.attributeSelection.ReliefFAttributeEval",
                              options=["-M", "-1", "-D", "1", "-K", "10"])

    # run the ReliefF alg
    relieff = AttributeSelection()
    relieff.search(search)
    relieff.evaluator(evaluation)
    relieff.select_attributes(filter_data)
    results = relieff.selected_attributes

    # weka wrapper returns the class col number with the results, so slice -1
    return [feature_names[i] for i in results[:-1]]

