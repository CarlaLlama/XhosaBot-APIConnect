import urllib3
import time

import feersum_nlu
from feersum_nlu.rest import ApiException
from examples import feersumnlu_host, feersum_nlu_auth_token
from dataset_text_samples import labelled_phrases_list

# Configure API key authorization: APIKeyHeader
configuration = feersum_nlu.Configuration()

configuration.api_key['AUTH_TOKEN'] = feersum_nlu_auth_token
configuration.api_key['X-Auth-Token'] = feersum_nlu_auth_token

configuration.host = feersumnlu_host

api_instance = feersum_nlu.FaqMatchersApi(feersum_nlu.ApiClient(configuration))

instance_name = 'xhosabot_matcher'

create_details = feersum_nlu.FaqMatcherCreateDetails(name=instance_name,
                                                     desc="FAQ matcher for Xhosa Maternity health chatbot.",
                                                     long_name="XhosaBot_Maternity_FAQs",
                                                     lid_model_file="lid_za",
                                                     load_from_store=False)

# Create NLU Labelled Text Samples from dataset
labelled_phrases_list_parsed = []
for phrase in labelled_phrases_list:
    labelled_phrases_list_parsed.append(
        feersum_nlu.LabelledTextSample(
                            text=phrase[0],
                            label=phrase[1],
                            lang_code=phrase[2]))

word_manifold_list = [feersum_nlu.LabeledWordManifold('eng', 'feers_wm_eng'),
                      feersum_nlu.LabeledWordManifold('xho', 'feers_wm_xho')]

# later use: faq_matcher_get_labels

# Launch Matcher Creation
try:
    print("Create the FAQ matcher:")
    api_response = api_instance.faq_matcher_create(create_details)
    print(" api_response", api_response)
except ApiException as e:
    print("Exception when calling an FAQ matcher create operation:\n" % e)
G
# Attempt add training samples
try:
    print("Add training samples to the FAQ matcher:")
    api_response = api_instance.faq_matcher_add_training_samples(instance_name, labelled_phrases_list_parsed)
    print(" api_response", api_response)
except ApiException as e:
    print("Exception when calling an FAQ matcher add training samples operation:\n" % e)

# Train the Matcher
try:
    train_details = feersum_nlu.TrainDetails(threshold=10.0,
                                             word_manifold_list=word_manifold_list,
                                             immediate_mode=True)

    print("Train the FAQ matcher:")
    instance_detail = api_instance.faq_matcher_train(instance_name, train_details)
    print(" api_response", instance_detail)

    # Training in progress..
    if instance_detail.training_stamp.startswith('ASYNC'):
        # Background training in progress. We'll poll and wait for it to complete.
        previous_id = instance_detail.id

        while True:
            time.sleep(1)
            inst_det = api_instance.faq_matcher_get_details(instance_name)
            if inst_det.id != previous_id:
                break  # break from while-loop when ID updated which indicates training done.

        print('Done.')

    print("Get the labels of named loaded FAQ matcher:")
    api_response = api_instance.faq_matcher_get_labels(instance_name)
    print(" api_response", api_response)

    text_input_0 = feersum_nlu.TextInput("Impawu zokuba n dizokubeleka?")
    text_input_1 = feersum_nlu.TextInput("How long should labour last?")

    print("Match a question:")
    api_response = api_instance.faq_matcher_retrieve(instance_name, text_input_0)
    print(" api_response", api_response)

    print("Match a question:")
    api_response = api_instance.faq_matcher_retrieve(instance_name, text_input_1)
    print(" api_response", api_response)

    '''
        print("Add online training samples to the FAQ matcher:")
        api_response = api_instance.faq_matcher_online_training_samples(instance_name,
                                                                        additional_labelled_text_sample_list)
        print("api_response", api_response)
    '''

except ApiException as e:
    print("Exception when calling an FAQ matcher operation: %s\n" % e)
except urllib3.exceptions.HTTPError as e:
    print("Connection HTTPError! %s\n" % e)

    # later do testing
    '''
    print("Add testing samples to the FAQ matcher:")
    api_response = api_instance.faq_matcher_add_testing_samples(instance_name, labelled_text_sample_testing_list)
    print(" api_response", api_response)
    
    immediate_mode = True  # Set to True to do a blocking train operation.
    '''
