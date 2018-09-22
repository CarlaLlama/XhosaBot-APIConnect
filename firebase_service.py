import firebase
import datetime

db = firebase.database()


def push_message(user_id, response):
    db.child(user_id).push({
        "messageFromBot": True,
        "messageSendTime": "" + datetime.datetime.now(),
        "messageText": "" + response
    })
