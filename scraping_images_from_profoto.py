from bs4 import BeautifulSoup as bs
import requests
import re

cnt = 160
base_url = 'https://pro.foto.ne.jp/free/products_list.php/cPath/21_28_70/page/{}'
for i in range(1,7):
    image_base_url = 'https://pro.foto.ne.jp/free/'
    response = requests.get(base_url.format(str(i)))
    soup = bs(response.text, 'html.parser')
    imgs = soup.find_all('img',src=re.compile('^img/images_thumb/'), limit=100)
    for img in imgs:
        img_url = image_base_url + img['src']
        res = requests.get(img_url)
        with open('./dog_images/dog{}.jpg'.format(str(cnt)), 'wb') as f:
            f.write(res.content)
        cnt += 1
