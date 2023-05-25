#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 25/05/2017

def send_email(subject, body):
    import smtplib

    gmail_user = "YOURGMAIL@gmail.com"
    gmail_pwd = "YOURPASSWORD"
    recipient = ["WHEREYOUWANTEMAILSENT@ucl.ac.uk"]
    FROM = gmail_user
    TO = recipient if type(recipient) is list else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print('[Email sent]')
    except:
        print("[Failed to send email]")
