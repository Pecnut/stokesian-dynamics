#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 25/05/2017

def send_email(subject, body):
    """Send an email. 

    Is only executed if `send_email = True` in inputs.py.
    Set up to work for Gmail accounts but requires some configuring of your
    account.

    Args:
        subject: Subject of the email
        body: Body text of the email

    Prints success or failure statement.
    """
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
