from weka.attribute_selection import ASSearch
from weka.attribute_selection import ASEvaluation
from weka.attribute_selection import AttributeSelection


def information_gain(filter_data, feature_names):
    # last param determines how many attributes are returned
    # 2nd param controls the score threshold
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
    # has no params
    evaluation = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval", options=[])

    # run the Information Gain alg
    info_gain = AttributeSelection()
    info_gain.search(search)
    info_gain.evaluator(evaluation)
    info_gain.select_attributes(filter_data)
    results = info_gain.selected_attributes

    # weka wrapper returns the class col number with the results, so slice -1
    return [feature_names[i] for i in results[:-1]]

