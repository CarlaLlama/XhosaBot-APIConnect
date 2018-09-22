import urllib3
import time

import feersum_nlu
from feersum_nlu.rest import ApiException
from examples import feersumnlu_host, feersum_nlu_auth_token

# Configure API key authorization: APIKeyHeader
configuration = feersum_nlu.Configuration()

# configuration.api_key['AUTH_TOKEN'] = feersum_nlu_auth_token
configuration.api_key['X-Auth-Token'] = feersum_nlu_auth_token  # Alternative auth key header!

configuration.host = feersumnlu_host

api_instance = feersum_nlu.FaqMatchersApi(feersum_nlu.ApiClient(configuration))

instance_name = 'xhosabot_matcher'

create_details = feersum_nlu.FaqMatcherCreateDetails(name=instance_name,
                                                     desc="Test FAQ matcher.",
                                                     long_name="The optional more descriptive name.",
                                                     lid_model_file="lid_za",
                                                     load_from_store=False)

# The training samples.
labelled_text_sample_list = []
labelled_text_sample_list.append(feersum_nlu.LabelledTextSample(text="How claim?",
                                                                label="claim",
                                                                lang_code="eng"))
labelled_text_sample_list.append(feersum_nlu.LabelledTextSample(text="Hoe eis?",
                                                                label="claim",
                                                                lang_code="afr"))

labelled_text_sample_list.append(feersum_nlu.LabelledTextSample(text="How quote?",
                                                                label="quote"))
labelled_text_sample_list.append(feersum_nlu.LabelledTextSample(text="Hoe prys?",
                                                                label="quote"))

labelled_text_sample_delete_list = []
labelled_text_sample_delete_list.append(feersum_nlu.LabelledTextSample(text="Hoe eis?",
                                                                       label="claim",
                                                                       lang_code="afr"))
labelled_text_sample_delete_list.append(feersum_nlu.LabelledTextSample(text="Hoe prys?",
                                                                       label="quote"))

labelled_text_sample_testing_list = []
labelled_text_sample_testing_list.append(feersum_nlu.LabelledTextSample(text="Where do I claim?",
                                                                        label="claim",
                                                                        lang_code="eng"))
labelled_text_sample_testing_list.append(feersum_nlu.LabelledTextSample(text="Waar moet ek eis?",
                                                                        label="claim",
                                                                        lang_code="afr"))

labelled_text_sample_testing_list.append(feersum_nlu.LabelledTextSample(text="Can I put in a claim?",
                                                                        label="quote"))  # text actually on 'claim'.
labelled_text_sample_testing_list.append(feersum_nlu.LabelledTextSample(text="Waar kan ek 'n prys kry?",
                                                                        label="quote"))

additional_labelled_text_sample_list = []
additional_labelled_text_sample_list.append(feersum_nlu.LabelledTextSample(text="How much does a quote cost?",
                                                                           label="quote"))
additional_labelled_text_sample_list.append(feersum_nlu.LabelledTextSample(text="How long does a claim take?",
                                                                           label="claim"))

word_manifold_list = [feersum_nlu.LabeledWordManifold('eng', 'feers_wm_eng'),
                      feersum_nlu.LabeledWordManifold('afr', 'feers_wm_afr'),
                      feersum_nlu.LabeledWordManifold('zul', 'feers_wm_zul')]
# The playground's pre-loaded embeddings include:
# "feers_wm_afr", "feers_wm_eng", "feers_wm_nbl", "feers_wm_xho",
# "feers_wm_zul", "feers_wm_ssw", "feers_wm_nso", "feers_wm_sot",
# "feers_wm_tsn", "feers_wm_ven", "feers_wm_tso"
# and "glove6B50D_trimmed"

text_input_0 = feersum_nlu.TextInput("Waar kan ek 'n eis insit?")
text_input_1 = feersum_nlu.TextInput("How long does a claim take?")

print()

try:
    print("Create the FAQ matcher:")
    api_response = api_instance.faq_matcher_create(create_details)
    print(" api_response", api_response)

    print("Add training samples to the FAQ matcher:")
    api_response = api_instance.faq_matcher_add_training_samples(instance_name, labelled_text_sample_list)
    print(" api_response", api_response)

    print("Add testing samples to the FAQ matcher:")
    api_response = api_instance.faq_matcher_add_testing_samples(instance_name, labelled_text_sample_testing_list)
    print(" api_response", api_response)

    immediate_mode = True  # Set to True to do a blocking train operation.

    train_details = feersum_nlu.TrainDetails(threshold=10.0,
                                             word_manifold_list=word_manifold_list,
                                             immediate_mode=immediate_mode)

    print("Train the FAQ matcher:")
    instance_detail = api_instance.faq_matcher_train(instance_name, train_details)
    print(" api_response", instance_detail)
    print()

    # TRAINING:
    # If timestamp begins with 'ASYNC...' the the training is running in the background and you need to poll until the
    # model ID has updated.
    # if timestamp doesn't begin with ASYNC then the training has completed synchronously and you may continue.

    if instance_detail.training_stamp.startswith('ASYNC'):
        # Background training in progress. We'll poll and wait for it to complete.
        previous_id = instance_detail.id

        while True:
            time.sleep(1)
            inst_det = api_instance.faq_matcher_get_details(instance_name)
            if inst_det.id != previous_id:
                break  # break from while-loop when ID updated which indicates training done.

        print('Done.')

    print("Get the details of all loaded FAQ matchers:")
    api_response = api_instance.faq_matcher_get_details_all()
    print(" api_response", api_response)

    print("Get the details of specific named loaded FAQ matcher:")
    api_response = api_instance.faq_matcher_get_details(instance_name)
    print(" api_response", api_response)
    cm_labels = api_response.cm_labels

    print("Get the labels of named loaded FAQ matcher:")
    api_response = api_instance.faq_matcher_get_labels(instance_name)
    print(" api_response", api_response)

    print("From the model details the cm_labels where =", cm_labels)

    print("Match a question:")
    api_response = api_instance.faq_matcher_retrieve(instance_name, text_input_0)
    print(" type(api_response)", type(api_response))
    print(" api_response", api_response)
    print()

    print("Update the parameters:")
    model_params = feersum_nlu.ModelParams(threshold=0.9, desc="Examples: Test FAQ matcher.", long_name="A longer name.")
    api_response = api_instance.faq_matcher_set_params(instance_name, model_params)
    print(" type(api_response)", type(api_response))
    print(" api_response", api_response)
    print()

    print("Match a question:")
    api_response = api_instance.faq_matcher_retrieve(instance_name, text_input_1)
    print(" api_response", api_response)

    print("Add online training samples to the FAQ matcher:")
    api_response = api_instance.faq_matcher_online_training_samples(instance_name,
                                                                    additional_labelled_text_sample_list)
    print("api_response", api_response)

    print("Match a question:")
    api_response = api_instance.faq_matcher_retrieve(instance_name, text_input_1)
    print(" api_response", api_response)

except ApiException as e:
    print("Exception when calling an FAQ matcher operation: %s\n" % e)
except urllib3.exceptions.HTTPError as e:
    print("Connection HTTPError! %s\n" % e)
