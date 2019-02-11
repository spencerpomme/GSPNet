# -*- coding: utf-8 -*-
import urllib2
import re
import cookielib
import time
from multiprocessing import Queue, Process, cpu_count

cookie = cookielib.CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookie))
header = {'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',\
          'Content-Type':'application/x-www-form-urlencoded',\
          'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0',\
          'Referer' : 'http://www.baidu.com/'}

def openPage(search_url, type_code):
        '''search the url page'''
        req = urllib2.Request(
        url = search_url,
        headers = header
        )
        try:
            page = opener.open(req).read().decode('unicode_escape')
        except:
            if type_code == 1:
                page = '<span class="nums_text">ERR</span>'
            if type_code == 2:
                page = '<div class="th_footer_l">ERR</span>'
            if type_code == 3:
                page = '<div id="resultInfo">ERR</div>'
            return page
        return page

#新闻搜索数量
def get_news_num(CITY, ADJ):
    url = "http://news.baidu.com/ns?cl=2&rn=20&tn=news&word="+str(CITY)+"%20"+str(ADJ)+"&ie=utf-8"
    try:
        page = urllib2.urlopen(url).read().decode('unicode_escape').encode('utf8')
    except:
        page = '<span class="nums">ERR</span>'
    num = re.findall('<span class="nums">.*?([\d\,]+).*?</span>', page)
    #print url
    #print num
    if num:
        total_num = ''.join(num[0].split(','))
    else:
        total_num = 'NOT FOUND'
    return total_num


#网页搜索数量
def get_webpage_num(CITY, ADJ):
    url = "http://www.baidu.com/s?wd="+str(CITY)+"%20"+str(ADJ)+"&rsv_bp=0&tn=baidu&rsv_spt=3&ie=utf-8&rsv_enter=1&rsv_sug3=10&rsv_sug4=250&rsv_sug1=1&rsv_sug2=0&inputT=2497&rsv_sug=1"
    #page = urllib2.urlopen(url).read().decode('unicode_escape')
    page = openPage(url,1)
    num = re.findall(r'<span class="nums_text">(.*?)</span>', page, re.S)
    if num == []:
        return 0
    else:
        num = num[0].encode("utf8")
        a = re.findall(r'(\d+)', num, re.S)
        total_num = 0
        for i in range(0, len(a)):
            total_num = total_num + int(a[i])*10**(3*(len(a)-1-i))
        return total_num


#贴吧主题数量
def get_tieba_num(CITY, ADJ):
    url = "http://tieba.baidu.com/f?kw="+str(CITY)+"%20"+str(ADJ)+"&ie=utf-8&t=12&fr=news"
    #page = urllib2.urlopen(url).read().decode('unicode_escape')
    page = openPage(url, 2)
    num = re.findall(r'<div class="th_footer_l">(.*?)</span>', page, re.S)
    if num == []:
        return 0
    else:
        num = num[0].encode("utf8")
        a = re.findall(r'(\d+)', num, re.S)
        total_num = 0
        for i in range(0, len(a)):
            total_num = total_num + int(a[i])*10**(3*(len(a)-1-i))
        return total_num

#图片数量
def get_pic_num(CITY, ADJ):
    url = "http://image.baidu.com/i?tn=baiduimage&ct=201326592&lm=-1&cl=2&t=12&word="+str(CITY)+"%20"+str(ADJ)+"&ie=utf-8&fr=news"
    #page = urllib2.urlopen(url).read().decode('unicode_escape')
    page = openPage(url, 3)
    num = re.findall(r'<div id="resultInfo">(.*?)</div>', page, re.S)
    if num == []:
        return 0
    else:
        num = num[0].encode("utf8")
        a = re.findall(r'(\d+)', num, re.S)
        total_num = 0
        for i in range(0, len(a)):
            total_num = total_num + int(a[i])*10**(3*(len(a)-1-i))
        return total_num

#------------------------------开始----------------------------------

city_list = ['中山市']
adj_list = ["金钟水库", "翠丽湖", "长龙坑水库", "石榴坑水库", "马岭水库", "横门水库",
            "浅水湖", "古宥水库", "南头水库", "长坑三级水库", "烟管山水库", "黄布水库",
            "横径水库", "蠄蜞塘水库", "逸仙水库", "田心水库", "古鹤水库", "长江水库", "石塘水库"]

start = time.time()

task_pool = Queue()




news_f = open("news.csv", "a")
webpage_f = open("webpage.csv", "a")
tieba_f = open("tieba.csv", "a")
pic_f = open("pic.csv", "a")

for j in range(0, len(adj_list)):
    news_f.write(","+adj_list[j])
    webpage_f.write(","+adj_list[j])
    tieba_f.write(","+adj_list[j])
    pic_f.write(","+adj_list[j])
news_f.write("\n")
webpage_f.write("\n")
tieba_f.write("\n")
pic_f.write("\n")
news_f.close()
webpage_f.close()
tieba_f.close()
pic_f.close()

for i in range(0, len(city_list)):
    news_f = open("news.csv", "a")
    webpage_f = open("webpage.csv", "a")
    tieba_f = open("tieba.csv", "a")
    pic_f = open("pic.csv", "a")
    news_f.write(city_list[i]+",")
    webpage_f.write(city_list[i]+",")
    tieba_f.write(city_list[i]+",")
    pic_f.write(city_list[i]+",")
    for j in range(0, len(adj_list)):
        news_num = get_news_num(city_list[i], adj_list[j])
        webpage_num = get_webpage_num(city_list[i], adj_list[j])
        tieba_num = get_tieba_num(city_list[i], adj_list[j])
        pic_num = get_pic_num(city_list[i], adj_list[j])
        news_data = str(news_num)+","
        news_f.write(news_data)
        webpage_data = str(webpage_num)+","
        webpage_f.write(webpage_data)
        tieba_data = str(tieba_num)+","
        tieba_f.write(tieba_data)
        pic_data = str(pic_num)+","
        pic_f.write(pic_data)
        print city_list[i].decode('utf8'), adj_list[j].decode('utf8'), "news", news_num
        print city_list[i], adj_list[j], "web", webpage_num
        print city_list[i].decode('utf8'), adj_list[j].decode('utf8'), "tieba", tieba_num
        print city_list[i].decode('utf8'), adj_list[j].decode('utf8'), "picture", pic_num
    news_f.write("\n")
    webpage_f.write("\n")
    tieba_f.write("\n")
    pic_f.write("\n")
    print str(city_list[i]).decode("utf8") + "  DOWN!"
    news_f.close()
    webpage_f.close()
    tieba_f.close()
    pic_f.close()

t = time.time() - start
print 'total time: ' + str(t)
