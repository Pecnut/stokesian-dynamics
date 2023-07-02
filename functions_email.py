#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 25/05/2017

import smtplib


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
    gmail_user = "YOURGMAIL@gmail.com"
    gmail_pwd = "YOURPASSWORD"
    recipient = ["WHEREYOUWANTEMAILSENT@example.com"]
    from_address = gmail_user
    to_address = recipient if isinstance(recipient,list) else [recipient]

    # Prepare actual message
    message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (from_address, ", ".join(to_address), subject, body)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(from_address, to_address, message)
        server.close()
        print('[Email sent]')
    except:
        print("[Failed to send email]")
