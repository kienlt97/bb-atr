import requests

token = '6570332463:AAE2r8rAq3Jlcc8dMN9mnvCg19hMZ_tXu2A'
method = 'sendMessage'
chat_id = '1489044599'


def sendMessage(message):
    try:
        message ="<b>{}</b>&parse_mode=HTML".format(message)
        url = 'https://api.telegram.org/bot{0}/{1}?chat_id={2}&text={3}'.format(token, method, chat_id, message)
        response = requests.post(url=url).json()
    except Exception as e:
        print("sendMessage e", e)

# if __name__ == '__main__':
#     sendMessage('toi test')