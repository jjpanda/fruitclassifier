from urllib.request import urlopen, Request
import os
import util_00 as util
##From imagenet

header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
url_list = []

def downloader(image_url, file):
    print(image_url)
    if image_url.endswith('.png'):
        full_file_name = str(file) + '.png'
    else:
        full_file_name = str(file) + '.jpg'
    try:
        req = Request(image_url, headers=header)
        raw_img = urlopen(req, timeout=5).read()
        f = open(full_file_name, 'wb')
        f.write(raw_img)
        f.close()
    except Exception as e:
        print('Can not download: ', image_url)
        print(e)
        
current_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_path, 'data', 'download', 'url')
dest_path = os.path.join(current_path, 'data', 'download', 'images')
            
for i in ['apple', 'orange', 'peach', 'pineapple']:
    count = 0
    url_file = os.path.join(src_path, str(i) + '.txt')
    urls = open(url_file).read().split('\n')
    util.checkdir(os.path.join(dest_path, str(i)))
    for url in urls:
        file = os.path.join(dest_path, str(i), str(count))
        downloader(url, file)
        count += 1