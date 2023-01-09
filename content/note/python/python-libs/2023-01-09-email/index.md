---
title: Python email
author: 王哲峰
date: '2023-01-09'
slug: email
categories:
  - Python
tags:
  - tool
---


```python
# -*- coding: utf-8 -*-


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header


def mail(msg, msg_attached = None, msg_attached_image = None, formater = 'text', is_attach = 'n'):
    if formater == 'text':
        msgs = MIMEText(msg, 
                        'plain', 
                        'utf-8')
    elif formater == 'html':
        msgs = MIMEText(msg, 
                        'html', 
                        'utf-8')
    elif formater == 'attached_text':
        msgs = MIMEMultipart()
    elif formater == 'attached_image':
        msgs = MIMEMultipart('related')
        msgs_alternative = MIMEMultipart('alternative')
        msgs_html = MIMEText(msg, 'html', 'utf-8')
        
    if is_attach == 'n':
        msgs['From'] = Header("163", 'utf-8')
        msgs['To'] = Header("gmail", 'utf-8')
        msgs['Subject'] = Header("Python SMTP 邮件测试", 'utf-8')
    elif is_attach == 'attach_text':
        msgs['From'] = Header("163", 'utf-8')
        msgs['To'] = Header("gmail", 'utf-8')
        msgs['Subject'] = Header("Python SMTP 邮件测试", 'utf-8')
        msgs.attach(msg)
        msgs.attach(msg_attached)
    elif is_attach == 'attach_image':
        msgs.attach(msgs_alternative)
        msgs_alternative.attach(msgs_html)
        msg_attached_image.add_header("Content-ID", "<image1>")
        msgs.attach(msg_attached_image)

    return msgs

def send_mail(msgs, smtp_host, smtp_port, smtp_user, smtp_password, from_addr, to_addr):
    try:
        smtpObj = smtplib.SMTP(smtp_host, smtp_port)
        smtpObj.set_debuglevel(1)
        # smtpObj.connect(smtp_host, smtp_port)
        smtpObj.login(smtp_user, smtp_password)
        smtpObj.sendmail(from_addr, to_addr, msgs.as_string())
        smtpObj.quit()
        print('邮件发送成功！')
    except smtplib.SMTPException:
        print("Error: 无法发送邮件！")



def main():
    # from_addr = input('From: ')         # 输入email地址和口令
    # password = input('Password: ')
    # to_addr = input('To: ')             # 输入收件人地址
    # smtp_server = input('SMTP server: ')# 输入SMTP服务器地址
    from_addr = 'wangzhefengr@163.com'
    to_addr = ['wangzhefengr@163.com']
    smtp_host = 'smtp.163.com'
    smtp_port = '25'
    smtp_user = 'wangzhefengr@163.com'
    smtp_password = input("Password:")

    # msg = 'hello, send by python'

    # msg = """
    # <p>Python 邮件发送测试...</p>
    # <p><a href="http://www.runoob.com">这是一个链接</a></p>
    # """

    # msg = MIMEText('这是菜鸟教程Python 邮件发送测试……', 'plain', 'utf-8')
    # msg_attached = MIMEText(open('./data/test.txt', 'rb').read(), 'base64', 'utf-8')
    # msg_attached["Content-Type"] = 'application/octet-stream'
    # msg_attached["Content-Disposition"] = 'attachment; filename="test.txt"'

    msg = """
    <p>Python 邮件发送测试...</p>
    <p><a href="http://www.runoob.com">菜鸟教程链接</a></p>
    <p>图片演示: </p>
    <p><img src="cid:image1"></p>
    """
    fp = open('./data/test.jpg', 'rb')
    msg_attached_image = MIMEImage(fp.read())
    fp.close()

    msgs = mail(msg, 
                msg_attached = None, 
                msg_attached_image = msg_attached_image, 
                formater = 'attached_image',
                is_attach = 'attach_image')
    send_mail(msgs, 
                smtp_host, smtp_port, smtp_user, smtp_password, 
                from_addr, to_addr)

if __name__ == "__main__":
    main()
```

