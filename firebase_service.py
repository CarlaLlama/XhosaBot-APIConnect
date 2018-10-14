import datetime
from google.cloud import firestore

db = firestore.Client()


def push_message(user_id, response):
    db.collection(user_id).add({
        u"messageFromBot": True,
        u"messageSendTime": datetime.datetime.now(),
        u"messageText": response
    })


def get_new_message(user_id):
    messages = db.collection(user_id)\
        .order_by(u'messageSendTime', direction=firestore.Query.DESCENDING)\
        .limit(1)\
        .get()

    for message in messages:
        message_dict = message.to_dict()
        if not message_dict["messageFromBot"]:
            # message_text = message.to_dict()
            print(message_dict["messageText"])
            return message_dict["messageText"]
        else:
            print("No new message from user.")
            return
