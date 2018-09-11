import smtplib;
from email.mime.text import MIMEText;
import numpy as np;
import os;

def inform(txt):
    try:
        path = os.path.dirname(os.path.realpath(__file__));
        address = np.loadtxt(path+os.sep+'email.np',dtype=str);
        s = smtplib.SMTP(address[0],int(address[1]));
        s.login(address[2],address[3]);
        msg = MIMEText("running stopped","plain","utf-8");
        msg["Subject"] = txt;
        msg["From"] = address[4];
        msg["To"] = address[5];
        s.sendmail(address[4],address[5],msg.as_string());
        s.close();
    except Exception,e:
        print('The email function does not work');
        print(e);
        