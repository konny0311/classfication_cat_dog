from bs4 import BeautifulSoup as bs
import requests

cnt = 0
words = ['犬','dog','개','狗','cão']
base_url = 'https://www.google.com/search?rlz=1C5CHFA_enJP830JP830&biw=1084&bih=798&tbm=isch&sa=1&ei=G0CTXM22BcmmmAWp0ZPoDg&q={}&oq={}&gs_l=img.3..0l10.102667.103873..104538...0.0..0.114.548.0j5......1....1..gws-wiz-img.....0..0i4.mUUG3Q9dET8#imgrc=B8HFcuio7XfGbM:'
for word in words:
    url = base_url.format(word, word)
    response = requests.get(url)
    soup = bs(response.text, 'html.parser')
    imgs = soup.find_all('img', limit=100) #1度に20枚しか取得できない
    for img in imgs:
        img_url = img['src']
        res = requests.get(img_url)
        with open('./dog_images/dog{}.jpg'.format(str(cnt)), 'wb') as f:
            f.write(res.content)
        cnt += 1

