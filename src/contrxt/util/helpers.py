from typing import List, Dict
from pyeda.inter import expr, expr2bdd


# def filter_already_existing(bdd_dict):
#     '''
#     Filter already done classes to avoid doing them twice
#     '''
#     already_existing_class_ids = set()
#     check_kpi_file_exists()
#
#     with open('results/KPI.csv','r') as f:
#         f.readline()
#         for l in f:
#             l = l.strip('\n').split(';')
#             class_id = l[0]
#             already_existing_class_ids.add(class_id)
#
#     exclude_class_ids = [] #'sci.electronics' #'5131', '1420', '7115'
#     already_existing_class_ids.update(exclude_class_ids)
#     for class_id in already_existing_class_ids:
#         bdd_dict.pop(str(class_id))
#
#     print(f'Removed already existing class_ids: {already_existing_class_ids}')
#     return bdd_dict


def union(list1: List, list2: List) -> int:
    '''Returns the total number of features from two lists.

    Parameters
    ----------
    list1, list2 : List
        Lists to unite.

    Returns
    -------
    out: Int
        Length of union between the two lists.

    '''
    list1, list2 = set(list1), set(list2)
    return len(list(list1.union(list2)))


def jaccard_similarity(list1: List, list2: List) -> int:
    '''Returns the Jaccard similarity between two lists.

    Parameters
    ----------
    list1, list2 : List
        Lists to compare.

    Returns
    -------
    out: Int
        Jaccard similarity between the two lists.

    '''
    list1, list2 = set(list1), set(list2)
    intersection = len(list(list1.intersection(list2)))
    union_ = (len(list(list1.union(list2))))
    return intersection / union_


def jaccard_distance(list1: List, list2: List) -> int:
    '''Returns the Jaccard distance between two lists.

    Parameters
    ----------
    list1, list2 : List
        Lists to compare.

    Returns
    -------
    out: Int
        Jaccard distance between the two lists.

    '''
    return 1 - jaccard_similarity(list1, list2)
