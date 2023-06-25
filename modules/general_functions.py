from ImageSet import ImageSet
from typing import List


def separate_to_sublists(list_of_ImageSets: List[ImageSet]):
    """
    Separates a list of ImageSet objects into sublists by their subject names,
    used magnification and used illumination type.
    :param list_of_ImageSets: list of ImageSet objects.
    :return: list of lists containing ImageSet objects.
    """
    list_of_sublists = []

    for imageSet in list_of_ImageSets:

        # Check if list_of_sublists is empty. If yes, create first sublist and
        # automatically add the first ImageSet object to it.
        if not list_of_sublists:

            sublist = [imageSet]
            list_of_sublists.append(sublist)
            continue

        number_of_sublists = len(list_of_sublists)
        for i in range(number_of_sublists):

            sublist = list_of_sublists[i]
            current_name = imageSet.name
            current_ill = imageSet.ill
            current_mag = imageSet.mag
            names_in_sublist = sublist[0].name
            ill_in_sublist = sublist[0].ill
            mag_in_sublist = sublist[0].mag
            if current_name == names_in_sublist and \
                    current_ill == ill_in_sublist and \
                    current_mag == mag_in_sublist:
                sublist.append(imageSet)
                break
            if number_of_sublists - 1 - i == 0:
                additional_list = [imageSet]
                list_of_sublists.append(additional_list)
                break

    return list_of_sublists
