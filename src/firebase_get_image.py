import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

import urllib.request

cred = credentials.Certificate('./ServiceKey.json')
firebase_admin.initialize_app(cred,{
    'storageBucket':'dimigonia.appspot.com'
})

#urllib.request.urlretrieve("https://firebasestorage.googleapis.com/v0/b/dimigonia.appspot.com/o/images%2Fpicture?alt=media&token=4156ab5d-89d5-49d2-afb6-390bd9b5f38c","./picture.jpeg")


bucket = storage.bucket()

blob = bucket.blob('images/picture')

blob.upload_from_filename(filename='picture.jpeg')
