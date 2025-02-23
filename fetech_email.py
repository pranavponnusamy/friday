from dotenv import load_dotenv
load_dotenv()
import json
import datetime
import enum
import os
import sys
import time

from google import genai
from google.genai.errors import ClientError
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from nylas import Client


class EmailCategory(enum.Enum):
    REPLY = "Reply"
    ARCHIVE = "Archive"
    Unsubscribe = "Unsubscribe"
    Forward = "Forward"

# Initialize the Nylas client with your env variables
nylas = Client(
    os.environ.get('NYLAS_API_KEY'),
    os.environ.get('NYLAS_API_URI')
)

grant_id = os.environ.get("NYLAS_GRANT_ID")
one_week_ago = int((datetime.datetime.now() - datetime.timedelta(days=7)).timestamp())

# Fetch messages from the last 7 days (limit 1 for this example)
messages = nylas.messages.list(
    grant_id,
    query_params={
        "limit": 100,
        # "received_after": one_week_ago,
        "in": "INBOX"
    }
)

# Initialize Google GenAI client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize Qdrant vector database client
q_client = QdrantClient(url="http://localhost:6333")
collection_name = "test_collection"  # new variable to control collection name

# Check/create the collection in Qdrant
try:
    q_client.get_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
except Exception:
    q_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )


def categorize_email(subject):
    """Categorize an email subject using GenAI and Qdrant context."""
    try:
        # Embed the subject text
        subject_embedding = client.models.embed_content(
            model="text-embedding-004",
            contents=subject
        )
        query_vector = subject_embedding.embeddings[0].values

        # Query the vector DB
        results = q_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            with_payload=True,
            limit=3
        ).points

        # Build context from retrieved emails
        context = ""
        for res in results:
            p = res.payload
            context += f"Email subject: {p.get('subject','')}. Action: {p.get('action',{})}\n"

        # Compose prompt
        prompt_text = (
            f"Context:\n{context}\n---\nCategorize this email subject: {subject}."
        )

        # Request an enum-based response from GenAI
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt_text,
            config={
                'response_mime_type': 'text/x.enum',
                'response_schema': EmailCategory,
            },
        )
        return response.text

    except ClientError as e:
        print("ClientError encountered:", e)
        return "ResourceExhausted"


def summarize_email(email_body):
    """Summarize an email body using GenAI."""
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"Summarize the following email: {email_body}",
            config={'response_mime_type': 'text/plain'}
        )
        return response.text
    except ClientError as e:
        print("ClientError encountered during summary:", e)
        return "Summary not available."


def suggest_reply(email_body):
    """Suggest a reply for an email using GenAI and similar email context from Qdrant."""
    try:
        # Compute embedding for the email body
        embedding_result = client.models.embed_content(
            model="text-embedding-004",
            contents=email_body
        )
        query_vector = embedding_result.embeddings[0].values

        # Query Qdrant for similar emails (limit to 3)
        results = q_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            with_payload=True,
            limit=3
        ).points

        # Build context from retrieved similar emails
        similar_context = ""
        for res in results:
            p = res.payload
            similar_context += f"Subject: {p.get('subject', '')} | Summary: {p.get('summary', '')}\n"

        # Build prompt that includes the similar emails' context
        prompt_text = (
            f"Based on the following similar emails:\n{similar_context}\n"
            f"Write a brief, professional reply to the email below:\n{email_body}"
        )
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt_text,
            config={'response_mime_type': 'text/plain'}
        )
        return response.text
    except ClientError as e:
        print("ClientError encountered during reply suggestion:", e)
        return ""


# Categories for demonstration (you can customize these)
categories = {
    "Work": [],
    "Personal": [],
    "Spam": [],
    "Newsletter": [],
    "Social": []
}

# Dictionaries to store data
email_points = {}  # Qdrant vectors (points)
nylas_emails = {}  # Original Nylas messages

# Process incoming messages one by one with interactive CLI
for index, message in enumerate(messages[0]):
    email_body = message.body

    # Categorize via LLM (optionally use subject as input to categorize_email)
    category = categorize_email(email_body)
    summary = summarize_email(email_body)

    # Embedding the email body
    embedding_result = client.models.embed_content(
        model="text-embedding-004",
        contents=email_body
    )

    # Extract sender (safely handle `message.from_`)
    sender = ""
    if hasattr(message, "from_") and message.from_:
        sender = message.from_[0].get("email", "")

    # Process recipients (safely handle `message.to`)
    recipients_raw = getattr(message, "to", [])
    recipients = [
        r.get("email", r) if isinstance(r, dict) else r for r in recipients_raw
    ]

    payload = {
        "category": category,
        "summary": summary,
        "subject": getattr(message, "subject", ""),
        "date": getattr(message, "date", ""),
        "sender": sender,
        "recipients": recipients,
        "message_id": getattr(message, "id", ""),
        "action": {},
    }

    # Build the Qdrant point
    point = PointStruct(
        id=index,
        vector=embedding_result.embeddings[0].values,
        payload=payload
    )

    # Save references and upsert point
    email_points[index] = point
    nylas_emails[index] = message
    operation_info = q_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[point],
    )
    print(f"Upsert result for email {index}:", operation_info)

    # (Optional) Add email to categories if needed
    if payload.get("category") not in categories:
        categories[payload.get("category")] = []
    categories[payload.get("category")].append(message)

    # Interactive CLI for this email in real time
    print("\n------------------------")
    print(f"Email [{index}]:")
    print(f"From: {payload.get('sender', '')}")
    print(f"To: {payload.get('recipients', '')}")
    print(f"Subject: {payload.get('subject', '')}")
    print(f"Summary: {payload.get('summary', '')}")
    print(f"Suggested Action: {payload.get('category', 'None')}")

    action = input("Action (archive/reply/forward/skip): ").strip().lower()
    if action in ["skip", ""]:
        print("Skipping email.")
        continue

    action_payload = {"type": action}
    if action == "reply":
        # Use LLM to suggest a reply without prompting the user immediately.
        reply_text = suggest_reply(email_body)
        print("LLM suggested reply:")
        print(reply_text)
        decision = input("Send reply? (yes/edit/skip): ").strip().lower()
        if decision == "edit":
            reply_text = input("Enter your revised reply: ").strip()
            action_payload["reply_text"] = reply_text
            action_payload["send_reply"] = True
        elif decision == "yes":
            action_payload["reply_text"] = reply_text
            action_payload["send_reply"] = True
        else:
            print("Reply draft saved but not sent.")
            action_payload["reply_text"] = reply_text
            action_payload["send_reply"] = False
    elif action == "forward":
        forward_to = input("Enter recipient email for forwarding: ").strip()
        forward_body = input("Enter additional message for forwarding: ").strip()
        action_payload["reply_text"] = forward_body

    # Update point payload and upsert updated record
    point.payload["action"] = action_payload
    op_info = q_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[point],
    )
    print(f"Updated action for email {index}: {op_info}")

    # Reflect action in Gmail
    try:
        nylas_email = nylas_emails.get(index)
        if not nylas_email:
            print("No corresponding Nylas email found.")
            continue

        message_id = getattr(nylas_email, "id", "")
        if action == "archive":
            request_body = {
                "remove_labels": ["INBOX"],
                "starred": False,
                "unread": False,
            }
            current_folders = getattr(nylas_email, "folders", [])
            new_folders = [folder for folder in current_folders if folder != "INBOX"]
            updated_message = nylas.messages.update(
                grant_id,
                nylas_email.id,
                request_body={
                    "folders": new_folders,
                    "starred": False,
                    "unread": False,
                }
            )
            print("Email archived:", updated_message)


        elif action == "reply":
            if action_payload.get("send_reply"):
                reply_text = action_payload.get("reply_text", "")
                # Extract original Message-ID safely
                headers = nylas_email.headers or []  # Fallback to empty list if headers is None
                original_msg_id = None
                for header in headers:
                    if header.get("name", "").lower() == "message-id":
                        original_msg_id = header.get("value")
                        break
                if not original_msg_id:
                    print("Warning: Original Message-ID header not found; threading may not work as expected.")
                    original_msg_id = message_id  # fallback (not ideal for Gmail threading)
                thread_id = nylas_email.thread_id or message_id
                original_subject = getattr(nylas_email, "subject", "")
                subject = original_subject if original_subject.lower().startswith("re:") else "Re: " + original_subject
                reply_data = {
                    "subject": subject,
                    "body": reply_text,
                    "to": [{"email": nylas_email.from_[0]["email"]}] if hasattr(nylas_email, "from_") and nylas_email.from_ else [],
                    "in_reply_to": original_msg_id,
                    "references": original_msg_id,
                    "thread_id": thread_id,
                }
                reply_response = nylas.messages.send(os.environ.get("NYLAS_GRANT_ID"), request_body=reply_data)
                print("Reply sent successfully.", reply_response)
            else:
                print("Draft saved; reply not sent.")



        elif action == "forward":
            forward_to = input("Confirm recipient email for forwarding: ").strip()
            forward_data = {
                "subject": "Fwd: " + (getattr(nylas_email, "subject", "")),
                "body": action_payload.get("reply_text", ""),
                "to": [{"email": forward_to}],
                "attachment_ids": getattr(nylas_email, "file_ids", []),
            }
            forward_response = nylas.messages.send(os.environ.get("NYLAS_GRANT_ID"), request_body=forward_data)
            print("Email forwarded successfully.", forward_response)
        else:
            print("No recognized action performed on Gmail.")
    except Exception as e:
        print("Error reflecting action in Gmail:", e)

print("All emails processed and upserted.\n")
