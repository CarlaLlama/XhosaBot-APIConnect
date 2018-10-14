from dataset_text_samples import labelled_response_list


def find_by_label(label):
    for response in labelled_response_list:
        if label == response[0]:
            return response[1]
    return "I don't know what you are saying message"
