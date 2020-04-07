from weka.attribute_selection import ASSearch
from weka.attribute_selection import ASEvaluation
from weka.attribute_selection import AttributeSelection


def cfs(filter_data, feature_names):
    # 2nd param changes direction 0 - Backward; 1 - Forward; 2 - Bidirectional
    # last param controls the search termination
    search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
    # parameters only control pool and thread size
    evaluation = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])

    # run the CFS alg
    cfs = AttributeSelection()
    cfs.search(search)
    cfs.evaluator(evaluation)
    cfs.select_attributes(filter_data)
    results = cfs.selected_attributes

    # weka wrapper returns the class col number with the results, so slice -1
    return [feature_names[i] for i in results[:-1]]

