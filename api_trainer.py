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
                                                     desc="FAQ matcher for Xhosa Maternity health chatbot.",
                                                     long_name="The optional more descriptive name.",
                                                     lid_model_file="lid_za",
                                                     load_from_store=False)

# The training samples.
labelled_phrases_list = []
# 1: Why is my baby not sleeping?
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Kutheni umntwana wam engalali?",
                                                            label="lala",
                                                            lang_code="xho"))

# 2: How do I register my baby?
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndingambhalisa njani umntwana wam?",
                                                            label="bhala",
                                                            lang_code="xho"))

# 3: Signs I am in labour
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Impawu zokuba ndizokubeleka?",
                                                            label="zokubeleka",
                                                            lang_code="xho"))

# 4: How do I know if I am having premature labour?
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndazi kanjani ukuba ndizokubeleka phambi kwexesha?",
                                                            label="phambi",
                                                            lang_code="xho"))

# 5: I have pre-eclampsia
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndine pre-eclampsia?",
                                                            label="pre-eclampsia",
                                                            lang_code="xho"))

# 6: How long should labour last?
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Lide kangakanani ixesha lokubeleka?",
                                                            label="ixhesha",
                                                            lang_code="xho"))

# 7: I am having morning sickness
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndinesifo sasekuseni?",
                                                            label="morning_sickness",
                                                            lang_code="xho"))

# 8: I have a UTI during pregnancy
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndine UTI ngelixa ndikhulelwe",
                                                            label="uti",
                                                            lang_code="xho"))

# 9: High-blood pressure during pregnancy
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="I high-blood pressure ngelixa ukhulelwe",
                                                            label="high-blood",
                                                            lang_code="xho"))

# 10: I have an STD
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndinesifo sokwasulelana ngesondo",
                                                            label="std",
                                                            lang_code="xho"))

# 11: I am sick during pregnancy
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndiyagula ngelixa ndikhulelwe",
                                                            label="ndiyagula",
                                                            lang_code="xho"))

# 12: My baby is sick
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Umntwana wam uyagula",
                                                            label="uyagula",
                                                            lang_code="xho"))

# 13: I am in pain, how do I get to a clinic
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndineentlungu, ndingaya njani ekliniki?",
                                                            label="ekliniki",
                                                            lang_code="xho"))

# 14: I have swollen legs
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndinemilenze edumbileyo",
                                                            label="dumbileyo",
                                                            lang_code="xho"))

# 15: What must I feed my baby?
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Yintoni endimele ndondle ngayo umntwana wam?",
                                                            label="ndondle",
                                                            lang_code="xho"))

# 16: I am bleeding during pregnancy
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndiyopha ngexesha lokukhulelwa",
                                                            label="opha",
                                                            lang_code="xho"))

# 17: I am depressed
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndinocinezelelo lwengqondo",
                                                            label="depressed",
                                                            lang_code="xho"))


# 18: How do I get a grant for my baby?
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndingasifumana njani isibonelelo somntwana wam?",
                                                            label="isibonelelo",
                                                            lang_code="xho"))

# 19: Will my baby have HIV?
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ingaba umntwana wam uzobane HIV?",
                                                            label="hiv",
                                                            lang_code="xho"))


# 20: Can I breastfeed with HIV?
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndingancancisa ndine HIV?",
                                                            label="hiv_cancisa",
                                                            lang_code="xho"))

# 21: My baby has nappy rash
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Umntwana wam unorhawuzelelo owenziwa yi-nappy",
                                                            label="nappy_rash",
                                                            lang_code="xho"))

# 22: My baby is not eating
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Umntwana wam akatyi",
                                                            label="akatyi",
                                                            lang_code="xho"))

# 23: Can I have sex during pregnancy?
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndingabelana ngesondo xa dikhulelwe?",
                                                            label="ngesondo",
                                                            lang_code="xho"))

# 24: Should I have a C Section?
labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndingayenza I C-Section?",
                                                            label="c-section",
                                                            lang_code="xho"))

labelled_phrases_list.append(feersum_nlu.LabelledTextSample(text="Ndingayenza uqhaqho?",
                                                            label="c-section",
                                                            lang_code="xho"))







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


faq_matcher_get_labels

text_input_0 = feersum_nlu.TextInput("Waar kan ek 'n eis insit?")
text_input_1 = feersum_nlu.TextInput("How long does a claim take?")

print()

try:
    print("Create the FAQ matcher:")
    api_response = api_instance.faq_matcher_create(create_details)
    print(" api_response", api_response)

    print("Add training samples to the FAQ matcher:")
    api_response = api_instance.faq_matcher_add_training_samples(instance_name, labelled_phrases_list)
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
