import urllib3

import feersum_nlu
from feersum_nlu.rest import ApiException
from examples import feersumnlu_host, feersum_nlu_auth_token

# Configure API key authorization: APIKeyHeader
configuration = feersum_nlu.Configuration()

configuration.api_key['X-Auth-Token'] = feersum_nlu_auth_token

configuration.host = feersumnlu_host

api_instance = feersum_nlu.FaqMatchersApi(feersum_nlu.ApiClient(configuration))

instance_name = 'xhosabot_matcher'

create_details = feersum_nlu.FaqMatcherCreateDetails(name=instance_name,
                                                     load_from_store=True)

text_input_0 = feersum_nlu.TextInput("Impawu zokuba n dizokubeleka?")
text_input_1 = feersum_nlu.TextInput("How long should labour last?")


def load_model():
    try:
        print("Create the FAQ matcher:")
        api_response = api_instance.faq_matcher_create(create_details)
        print(" api_response", api_response)

        # Get the classifier's possible labels. Might be inferred from the training data, but guaranteed to be available
        # after training.
        print("Get the labels of named loaded FAQ matcher:")
        api_response = api_instance.faq_matcher_get_labels(instance_name)
        print(" api_response", api_response)

    except ApiException as e:
        print("Exception when calling an FAQ matcher operation: %s\n" % e)
    except urllib3.exceptions.HTTPError:
        print("Connection HTTPError!")


def query_model(text):
    text_input = feersum_nlu.TextInput(text)
    try:
        print("Match a question:")
        api_response = api_instance.faq_matcher_retrieve(instance_name, text_input)
        print(" api_response", api_response)
        return api_response
    except ApiException as e:
        print("Exception when calling an FAQ matcher operation: %s\n" % e)
    except urllib3.exceptions.HTTPError:
        print("Connection HTTPError!")
