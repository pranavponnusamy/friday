from dotenv import load_dotenv
load_dotenv()

import os
import sys
from nylas import Client

nylas = Client(
    os.environ.get('NYLAS_API_KEY'),
    os.environ.get('NYLAS_API_URI')
)

grant_id = os.environ.get("NYLAS_GRANT_ID")
folder_id = os.environ.get("FOLDER_ID")

folder = nylas.folders.list(
  grant_id
)

for x in range(0, len(folder[0])):
    # if folder[0][x].id == "Label_2945227334572139329":
    print(folder[0][x].id)