import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def is_smtp_configured() -> bool:
    return bool(settings.SMTP_HOST and settings.SMTP_USER and settings.SMTP_PASSWORD)


def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send an email via SMTP. Returns True on success, False on failure."""
    if not is_smtp_configured():
        logger.error("SMTP not configured — cannot send email")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM_EMAIL or settings.SMTP_USER}>"
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(
                settings.SMTP_FROM_EMAIL or settings.SMTP_USER,
                to_email,
                msg.as_string(),
            )

        logger.info("Email sent to %s: %s", to_email, subject)
        return True
    except Exception as e:
        logger.error("Failed to send email to %s: %s", to_email, e)
        return False


def send_password_reset_email(to_email: str, reset_token: str) -> bool:
    """Send a password reset email with a link containing the reset token."""
    reset_url = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin:0;padding:0;background-color:#0a0a0a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
        <div style="max-width:480px;margin:40px auto;background-color:#141414;border-radius:12px;border:1px solid #262626;overflow:hidden;">
            <!-- Header -->
            <div style="padding:32px 32px 0;text-align:center;">
                <div style="display:inline-block;background-color:#3b82f6;border-radius:8px;padding:8px 12px;margin-bottom:16px;">
                    <span style="color:white;font-size:18px;font-weight:600;">ChapterLens</span>
                </div>
            </div>

            <!-- Body -->
            <div style="padding:24px 32px 32px;">
                <h1 style="color:#fafafa;font-size:20px;font-weight:600;margin:0 0 8px;">Reset Your Password</h1>
                <p style="color:#a1a1aa;font-size:14px;line-height:1.6;margin:0 0 24px;">
                    We received a request to reset the password for your account. Click the button below to set a new password.
                </p>

                <div style="text-align:center;margin:24px 0;">
                    <a href="{reset_url}"
                       style="display:inline-block;background-color:#3b82f6;color:white;text-decoration:none;padding:12px 32px;border-radius:8px;font-size:14px;font-weight:600;">
                        Reset Password
                    </a>
                </div>

                <p style="color:#71717a;font-size:12px;line-height:1.5;margin:24px 0 0;">
                    This link will expire in 15 minutes. If you did not request a password reset, you can safely ignore this email.
                </p>

                <hr style="border:none;border-top:1px solid #262626;margin:24px 0;">

                <p style="color:#52525b;font-size:11px;line-height:1.5;margin:0;">
                    If the button above doesn't work, copy and paste this URL into your browser:<br>
                    <span style="color:#71717a;word-break:break-all;">{reset_url}</span>
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    return send_email(to_email, "Reset Your Password — ChapterLens", html)
