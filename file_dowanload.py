"""
purpose:download file from internet using ftp://XXXX
author:qiufen-chen
date:11/11/2019
"""

import requests
from bs4 import BeautifulSoup
import re
import os

def getHTMLText(url):
    """
    purpose:Open the url and read
    :param url:The url has been obtained
    :return:content
    """

    try:
        web_page = requests.get(url, timeout=30)
        web_page.raise_for_status()
        web_page.encoding = "utf-8"
        content = web_page.text
        return content
    except:
        return  ""


def getFileList(soup):
    """
    purpose:Extract the content we want through the structure of the page
    :param soup:Content of web pages
    :return:
    """

    print("Please wait for seconds ......")
    folder = "Download_files"  # Store folder name

    # Determine if the folder exists
    if (os.path.exists(folder) == False):
        # If it doesn't exist, please create a new folder
        os.mkdir(folder)
    os.chdir(folder)

    data = soup.find_all('tr')  # Find all tr tags

    for info in data:
        if(len(re.compile(r'\.[a-zA-Z0-9]+$').findall(info.a.text.strip()))):
            filename = info.a.text.strip()  # Extract file name
            # 拼接网址
            response = urlopen(url + info.a['href'])
            file = response.read()

            # 文件存储
            with open(filename,'wb') as f:
                print("save----> %s" %filename)
                f.write(file)
    print("Download Completed!")


if __name__ == '__main__':
    url = 'http://www.math.pku.edu.cn/teachers/lidf/docs/textrick/index.htm'
    html = getHTMLText(url)
    soup = BeautifulSoup(html, "html.parser")
    getFileList(soup)
















