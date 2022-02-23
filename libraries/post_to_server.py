
#
#  Sends post to relay controller (raspberry pi) to change state of switch on esp8266
#

# importing the requests library
import requests

# api-endpoint
on_url = "http://192.168.1.205:5000/on"
off_url = "http://192.168.1.205:5000/off"

def send_post_on():
    r = requests.get(on_url)
    print(r)

def send_post_off():
    r = requests.get(off_url)
    print(r)