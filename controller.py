from firebase_service import get_new_message, push_message
from load_model import load_model, query_model
from response_generator import find_by_label

def init_model:
    # Initiate model
    load_model()


def retreive_message_triggered(userId):
    print("Triggered message received from user: " + userId)
    new_message = get_new_message(userId)
    print("Message received: " + new_message)


    # On new message received make FAQ matcher request
    predicted_label = ""
    if new_message is not None:
        api_response = query_model(new_message)
        if api_response is not None:
            predicted_label = api_response[0].label
            print("Predicted label: " + predicted_label)


    # On FAQ matcher response use response_generator to get answer
    if predicted_label != "":
        output_text = find_by_label(predicted_label)
        push_message("lCsDaad4LSOyCnHHoe81BESaDt92", output_text)
