import datetime
from google.cloud import firestore

db = firestore.Client()


def push_message(user_id, response):
    db.collection(user_id).add({
        u"messageFromBot": True,
        u"messageSendTime": datetime.datetime.now(),
        u"messageText": response
    })


push_message("56iEUgkELJ8B3rYqDpyB", "Hello")
