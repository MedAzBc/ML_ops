import smtplib
from email.mime.text import MIMEText
import sys

def send_email(subject, body):
    sender_email = "azizbchir189@gmail.com"
    receiver_email = "mohamedaziz.b'chir@esprit.tn"
    #password = "211JMT4775"
    password = "zhlp nigs fcyq elph"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    subject = sys.argv[1]
    body = sys.argv[2]
    send_email(subject, body)

