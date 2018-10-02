from dataset_text_samples import labelled_phrases_list


def find_by_label(label):
    for phrase in labelled_phrases_list:
        if label == phrase[1]:
            return phrase[0]