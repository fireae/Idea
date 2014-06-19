# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:29:01 2014

@author: Administrator
"""

import urllib2
import time
if __name__ == '__main__':
    idx = 0
    while idx < 100:
        #url = 'https://id.plaync.com/account/captcha'
        url = 'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=login&rand=sjrand'
        f = urllib2.urlopen(url)
        data = f.read()
        #save_name = 'E:\\workplace\\python\\captcha\\'+str(idx)+'.png'
        save_name = 'E:\\workplace\\python\\123061\\'+str(idx)+'.png'
        idx = idx + 1
        with open(save_name, 'wb') as code:
            code.write(data)
            print save_name 
            print 'save ok'
        time.sleep(1)

#import os
#import urllib2
#from urlparse import urlsplit
#
#def get(url, file_path, file_name = None, buffer = 16*1024):
#    
#
#def write_file(src, dst, total_len):
#    if not total_len:
#        total_len = 0
#    else:
#        total_len = float(total_len)
#
#    byte_read = 0.0
#    while (1):
#        buf = src.read(buffer)
#        
#    
#def download_file(url_path, save_path, attempt = 5):
#    print url_path
#    print save_path
#    
#    while(attempt > 0):
#        attempt = attempt - 1
#        filename = 